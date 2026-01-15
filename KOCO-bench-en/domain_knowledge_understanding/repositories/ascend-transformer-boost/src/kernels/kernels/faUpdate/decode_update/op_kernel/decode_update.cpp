/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "kernels/utils/kernel/kernel_utils.h"
#include "kernels/faUpdate/decode_update/tiling/tiling_data.h"
using namespace AscendC;

static constexpr uint32_t BUFFER_NUM = 2;
static constexpr uint32_t NUM7 = 7;
static constexpr uint32_t NUM8 = 8;
static constexpr uint32_t NUM16 = 16;
static constexpr uint32_t NUM64 = 64;
static constexpr uint32_t NUM256 = 256;
static constexpr uint32_t MAX_UB_SIZE = 188 * 1024; //  double buffer, 每块94KB共188KB
static const uint16_t ALIGNED_TO_8 = 8;
static const int32_t ALIGNED_TO_2 = 2;
static const uint32_t SPLIT_TO_2 = 2;

class DecodeUpdate {
public:
    __aicore__ inline DecodeUpdate() {}
    __aicore__ inline void Init(GM_ADDR les, GM_ADDR in, GM_ADDR out, AsdOps::DecodeUpdateTilingData *tdata)
    {
        this->hDim = tdata->hDim;
        this->sp = tdata->sp;
        this->totalLength = tdata->totalLength;

        if (GetBlockIdx() < tdata->formerNum) {
            blockLength = tdata->formerLength;
            this->GmStartOffset = GetBlockIdx() * tdata->formerLength;
        } else {
            blockLength = tdata->tailLength;
            this->GmStartOffset = tdata->formerNum * tdata->formerLength +
                                  (GetBlockIdx() - tdata->formerNum) * tdata->tailLength; // tail block
        }
        uint32_t spAligned = (NUM8 + 7) / NUM8 * NUM8;
        //  用94K的UB大小推算出 tileLength 最大能设置到多少
        uint32_t maxTileLength = (MAX_UB_SIZE - NUM8 * sizeof(uint32_t)) /
                                 (sizeof(float) * spAligned * BUFFER_NUM * (2 * (1 + hDim) + hDim / spAligned) +
                                  hDim * spAligned * sizeof(float));
        if (sp >= NUM8) {
            maxTileLength = maxTileLength / SPLIT_TO_2;
        }
        this->tileLength = min(maxTileLength, blockLength);
        this->curLength = this->tileLength;
        this->lastLength = blockLength % tileLength;
        this->loopCount = blockLength / tileLength + (lastLength == 0 ? 0 : 1);
        this->tileLengthAlig = ((tileLength + NUM7) / NUM8) * NUM8;
        this->lastLengthAlig = ((lastLength + NUM7) / NUM8) * NUM8;
        // 设置全局变量的起始地址与总长度BLOCK_LENGTH; sp, B*s*hc in1 sp; B*s*hc, hd in2
        lesGm.SetGlobalBuffer((__gm__ float *)les, totalLength * sp);
        inGm.SetGlobalBuffer((__gm__ float *)in, totalLength * hDim * sp);
        outGm.SetGlobalBuffer((__gm__ float *)out, totalLength * hDim);
        pipe.InitBuffer(inQueueLes, BUFFER_NUM, tileLengthAlig * sp * sizeof(float));
        pipe.InitBuffer(inQueueIn, BUFFER_NUM, tileLength * hDim * sp * sizeof(float));
        pipe.InitBuffer(outQueueOut, BUFFER_NUM, tileLength * hDim * sizeof(float));
        if (hDim > NUM256 && sp >= NUM8) {
            pipe.InitBuffer(lesexpBuffer, tileLengthAlig * sp * NUM64 * sizeof(float));
        } else {
            pipe.InitBuffer(lesexpBuffer, tileLengthAlig * sp * hDim * sizeof(float));
        }
    }
    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < this->loopCount; i++) {
            if (lastLength && i == this->loopCount - 1) {
                curLength = lastLength;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        //  按照逻辑队列初始化的buffer数量和每块大小，分配对应大小的内存
        LocalTensor<float> LesLocal = inQueueLes.AllocTensor<float>();
        LocalTensor<float> inLocal = inQueueIn.AllocTensor<float>();
        if (curLength % ALIGNED_TO_8 == 0) {
            for (int32_t i = 0; i < sp; i++) {
                DataCopy(LesLocal[curLength * i], lesGm[progress * tileLength + totalLength * i + GmStartOffset],
                         curLength);
                DataCopy(inLocal[curLength * hDim * i],
                         inGm[progress * tileLength * hDim + totalLength * hDim * i + GmStartOffset * hDim],
                         curLength * hDim);
            }
        } else {
            uint32_t curLengthAlig = ((curLength + NUM7) / NUM8) * NUM8;
            for (int32_t i = 0; i < sp; i++) {
                DataCopyPad(LesLocal[curLengthAlig * i], lesGm[progress * tileLength + totalLength * i + GmStartOffset],
                            {static_cast<uint16_t>(1), static_cast<uint32_t>(curLength * sizeof(float)), 0, 0, 0},
                            {true, 0, static_cast<uint8_t>(NUM8 - curLength % NUM8), 0});
                DataCopy(inLocal[curLength * hDim * i],
                         inGm[progress * tileLength * hDim + totalLength * hDim * i + GmStartOffset * hDim],
                         curLength * hDim);
            }
        }

        inQueueLes.EnQue(LesLocal);
        inQueueIn.EnQue(inLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<float> lesLocal = inQueueLes.DeQue<float>();
        LocalTensor<float> inLocal = inQueueIn.DeQue<float>();
        LocalTensor<float> lesexpLocal = lesexpBuffer.Get<float>();
        LocalTensor<float> outLocal = outQueueOut.AllocTensor<float>();

        // broad to 8
        uint32_t curLengthPad = ((curLength + NUM7) / NUM8) * NUM8;
        const uint32_t srcShape[2] = {sp * curLengthPad, 1};
        uint32_t dstShape[2] = {sp * curLengthPad, hDim};
        if (hDim > NUM256 && sp >= NUM8) {
            dstShape[1] = NUM64;
        }

        // 公式： o = sum(lse_exp/sum(lse_exp) * oi)
        PipeBarrier<PIPE_V>();
        Exp(lesexpLocal, lesLocal, curLengthPad * sp); //  exp(les)
        PipeBarrier<PIPE_V>();

        // move lesLocal <-  lesexpLocal[0]
        DataCopy(lesLocal, lesexpLocal, curLengthPad);
        PipeBarrier<PIPE_V>();
        for (int32_t i = 1; i < sp; i++) {
            Add(lesLocal, lesLocal, lesexpLocal[i * curLengthPad], curLength);
            PipeBarrier<PIPE_V>();
        }

        for (int32_t i = 0; i < sp; i++) {
            Div(lesexpLocal[i * curLengthPad], lesexpLocal[i * curLengthPad], lesLocal, curLength);
        }

        // broad到8 优化Mul
        PipeBarrier<PIPE_V>();
        BroadCast<float, ALIGNED_TO_2, 1>(lesexpLocal, lesexpLocal, dstShape, srcShape);
        PipeBarrier<PIPE_V>();
        // [sp,curLengthPad,headDim] [sp,cur,headDim]
        if (hDim > NUM256 && sp >= NUM8) {
            for (int32_t i = 0; i < sp; i++) {
                Mul(inLocal[i * curLength * hDim], inLocal[i * curLength * hDim], lesexpLocal[i * curLengthPad * NUM64],
                    64, curLength * hDim / 64, {1, 1, 1, 8, 8, 0});
            }
        } else {
            for (int32_t i = 0; i < sp; i++) {
                Mul(inLocal[i * curLength * hDim], inLocal[i * curLength * hDim], lesexpLocal[i * curLengthPad * hDim],
                    curLength * hDim);
            }
        }
        PipeBarrier<PIPE_V>();

        // move lesLocal <-  inLocal[0]
        DataCopy(outLocal, inLocal, curLength * hDim);
        PipeBarrier<PIPE_V>();
        for (int32_t i = 1; i < sp; i++) {
            Add(outLocal, outLocal, inLocal[i * curLength * hDim], curLength * hDim);
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_V>();

        outQueueOut.EnQue<float>(outLocal);
        inQueueLes.FreeTensor(lesLocal);
        inQueueIn.FreeTensor(inLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<float> outLocal = outQueueOut.DeQue<float>();
        DataCopy(outGm[GmStartOffset * hDim + progress * tileLength * hDim], outLocal, curLength * hDim);
        outQueueOut.FreeTensor(outLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueLes, inQueueIn, inQueueIdx;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOut;
    TBuf<QuePosition::VECCALC> lesexpBuffer;
    GlobalTensor<float> lesGm;
    GlobalTensor<float> inGm;
    GlobalTensor<float> outGm;
    uint32_t blockLength; //  单核上数据总长度
    uint16_t tileLength;  //  单核循环的非最后一轮数据长度
    uint16_t curLength; //  Tiling 循环数据长度为 curLength，最后一轮为 lastLength，内存申请使用 tileLength，数据量用
                        //  curLength
    uint16_t lastLength; //  单核循环的最后一轮数据长度
    uint32_t loopCount;  //  单核循环次数
    uint32_t hDim;       //  head dimension
    uint32_t sp;
    uint32_t tileLengthAlig;
    uint32_t lastLengthAlig;
    uint32_t totalLength;
    uint32_t GmStartOffset;
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AsdOps::DecodeUpdateTilingData *tilingdata)
{
    tilingdata->formerLength = ((const __gm__ uint32_t *)p_tilingdata)[0];
    tilingdata->tailLength = ((const __gm__ uint32_t *)p_tilingdata)[1];
    tilingdata->formerNum = ((const __gm__ uint32_t *)p_tilingdata)[2];
    tilingdata->tailNum = ((const __gm__ uint32_t *)p_tilingdata)[3];
    tilingdata->hDim = ((const __gm__ uint32_t *)p_tilingdata)[4];
    tilingdata->sp = ((const __gm__ uint32_t *)p_tilingdata)[5];
    tilingdata->totalLength = ((const __gm__ uint32_t *)p_tilingdata)[6];
}

extern "C" __global__ __aicore__ void decode_update(GM_ADDR les, GM_ADDR in, GM_ADDR out, GM_ADDR tiling)
{
    DecodeUpdate op;
    AsdOps::DecodeUpdateTilingData tdata;
    InitTilingData(tiling, &tdata);
    op.Init(les, in, out, &tdata);
    op.Process();
}
