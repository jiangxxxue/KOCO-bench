/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_OP_COMMON_LIB_FASTER_GELU_FORWARD_H
#define ASCEND_OP_COMMON_LIB_FASTER_GELU_FORWARD_H

#include "kernel_operator.h"
#include "kernels/activation/faster_gelu_forward/tiling/tiling_data.h"
#include "kernels/utils/kernel/kernel_utils.h"

using namespace AscendC;
constexpr uint32_t BUFFER_NUM = AsdOps::FASTER_GELU_FORWARD_BUFF_NUM;
constexpr float COEFFICIENTS = -1.702;
constexpr float ADD_SCALAR_ONE = 1.0;

template <typename inType, typename outType, bool highPerformance = true> class FasterGeluForward {
public:
    __aicore__ inline FasterGeluForward() {}
    __aicore__ inline ~FasterGeluForward() {}
    __aicore__ inline void Init(GM_ADDR inputAddr, GM_ADDR outAddr, AsdOps::FasterGeluForwardTilingData &tilingData)
    {
        this->blockLength = tilingData.singleCoreDataLen[GetBlockIdx()];
        this->tileLength = tilingData.maxTileLen;
        this->tileLoopNum = this->blockLength / this->tileLength;
        this->tailTileLength = this->blockLength % this->tileLength;
        for (uint32_t i = 0; i < GetBlockIdx(); ++i) {
            this->gmOffset += tilingData.singleCoreDataLen[i];
        }
        if (GetBlockIdx() == tilingData.usedCoreNum - 1) {
            this->gmOffset -= tilingData.alignDataNum;
        }
        inputGM.SetGlobalBuffer((__gm__ inType *)inputAddr + this->gmOffset, this->blockLength);
        outputGM.SetGlobalBuffer((__gm__ outType *)outAddr + this->gmOffset, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(inType));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(outType));
        pipe.InitBuffer(inFp32Buffer, this->tileLength * sizeof(float));
        pipe.InitBuffer(outFp32Buffer, this->tileLength * sizeof(float));
    }

    template <bool isHalf = true> __aicore__ inline void Process()
    {
        uint64_t offset = 0;
        if constexpr (isHalf) {
            LocalTensor<float> inLocal = inFp32Buffer.Get<float>();
            LocalTensor<float> outLocal = outFp32Buffer.Get<float>();
            for (uint32_t i = 0; i < this->tileLoopNum; i++) {
                CopyInAndCastF32(inLocal, inputGM, inQueueX, offset, this->tileLength);
                Compute(outLocal, inLocal, this->tileLength);
                Cast16AndCopyOut(outLocal, outputGM, outQueueZ, offset, this->tileLength);
                offset = offset + this->tileLength;
            }

            if (this->tailTileLength > 0) {
                CopyInAndCastF32(inLocal, inputGM, inQueueX, offset, this->tailTileLength);
                Compute(outLocal, inLocal, this->tailTileLength);
                Cast16AndCopyOut(outLocal, outputGM, outQueueZ, offset, this->tailTileLength);
            }
        } else {
            for (uint32_t i = 0; i < this->tileLoopNum; ++i) {
                // 将数据从GM -> inQueueX
                CopyIn(inputGM, inQueueX, offset, this->tileLength);
                LocalTensor<float> inLocal = inQueueX.DeQue<float>();
                LocalTensor<float> outLocal = outQueueZ.AllocTensor<float>();
                Compute(outLocal, inLocal, this->tileLength);
                inQueueX.FreeTensor(inLocal);
                outQueueZ.EnQue<float>(outLocal);
                CopyOut(outputGM, outQueueZ, offset, this->tileLength);
                offset = offset + this->tileLength;
            }
            if (this->tailTileLength > 0) {
                CopyIn(inputGM, inQueueX, offset, this->tailTileLength);
                LocalTensor<float> inLocal = inQueueX.DeQue<float>();
                LocalTensor<float> outLocal = outQueueZ.AllocTensor<float>();
                Compute(outLocal, inLocal, this->tailTileLength);
                inQueueX.FreeTensor(inLocal);
                outQueueZ.EnQue<float>(outLocal);
                CopyOut(outputGM, outQueueZ, offset, this->tailTileLength);
            }
        }
    }

    __aicore__ inline void Compute(LocalTensor<float> &zLocal, LocalTensor<float> &xLocal, uint64_t tileLength)
    {
        Muls(zLocal, xLocal, COEFFICIENTS, tileLength);
        PipeBarrier<PIPE_V>();

        // 指数一定要升到fp32
        Exp(zLocal, zLocal, tileLength);
        PipeBarrier<PIPE_V>();
        Adds(zLocal, zLocal, ADD_SCALAR_ONE, tileLength);
        PipeBarrier<PIPE_V>();

        // fast_gelu(x) = x / x1
        if constexpr (highPerformance) {
            Reciprocal(zLocal, zLocal, tileLength);
            PipeBarrier<PIPE_V>();

            Mul(zLocal, xLocal, zLocal, tileLength);
            PipeBarrier<PIPE_V>();
        } else {
            // 除法一定要升到fp32
            Div(zLocal, xLocal, zLocal, tileLength);
            PipeBarrier<PIPE_V>();
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    TBuf<AscendC::TPosition::VECCALC> inFp32Buffer;
    TBuf<AscendC::TPosition::VECCALC> outFp32Buffer;
    GlobalTensor<inType> inputGM;
    GlobalTensor<outType> outputGM;
    uint32_t blockLength{0};
    uint32_t tileLoopNum{0};
    uint32_t tileLength{0};
    uint32_t tailTileLength{0};
    uint64_t gmOffset{0};
};

#endif // ASCEND_OP_COMMON_LIB_FASTER_GELU_FORWARD_H
