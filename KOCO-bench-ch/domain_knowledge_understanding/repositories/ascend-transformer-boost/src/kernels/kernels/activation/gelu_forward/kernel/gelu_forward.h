/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPP_FASTER_GELU_FORWARD_H
#define OPP_FASTER_GELU_FORWARD_H
#include "kernel_operator.h"
#include "kernels/activation/gelu_forward/tiling/tiling_data.h"
#include "kernels/utils/kernel/kernel_utils.h"
using namespace AscendC;

template <typename inType, typename outType, bool highPerformance = true> class GeluForward {
public:
    __aicore__ inline GeluForward() {}
    __aicore__ inline ~GeluForward() {}

    __aicore__ inline void Init(GM_ADDR inputAddr, GM_ADDR outAddr, AsdOps::GeluForwardTilingData &tilingData)
    {
        this->blockLength = tilingData.blockLength;
        this->tileNum = tilingData.tileNum;
        this->tileLength = tilingData.tileLength;
        this->tailLength = tilingData.tailLength;
        this->bufferNum = tilingData.bufferNum;

        inputGM.SetGlobalBuffer((__gm__ inType *)inputAddr + this->blockLength * GetBlockIdx(), this->blockLength);
        outputGM.SetGlobalBuffer((__gm__ outType *)outAddr + this->blockLength * GetBlockIdx(), this->blockLength);

        // 不能整除的情况下，可能存在浪费
        pipe.InitBuffer(inQueueX, this->bufferNum, this->tileLength * sizeof(inType));
        pipe.InitBuffer(outQueueZ, this->bufferNum, this->tileLength * sizeof(outType));
    }

    __aicore__ inline void Process()
    {
        uint64_t offset = 0;
        pipe.InitBuffer(inFp32Buffer, this->tileLength * sizeof(float));
        pipe.InitBuffer(outFp32Buffer, this->tileLength * sizeof(float));
        LocalTensor<float> inLocal = inFp32Buffer.Get<float>();
        LocalTensor<float> outLocal = outFp32Buffer.Get<float>();
        for (int32_t i = 0; i < this->tileNum; i++) {
            CopyInAndCastF32(inLocal, inputGM, inQueueX, offset, this->tileLength);
            Compute(outLocal, inLocal, this->tileLength);
            Cast16AndCopyOut(outLocal, outputGM, outQueueZ, offset, this->tileLength);
            offset = offset + this->tileLength;
        }
        // 处理每个核上的尾块
        if (this->tailLength != 0) {
            CopyInAndCastF32(inLocal, inputGM, inQueueX, offset, this->tailLength);
            Compute(outLocal, inLocal, this->tailLength);
            Cast16AndCopyOut(outLocal, outputGM, outQueueZ, offset, this->tailLength);
        }
    }

    __aicore__ inline void ProcessFP32()
    {
        uint64_t offset = 0;
        for (uint32_t i = 0; i < this->tileNum; i++) {
            CopyIn(inputGM, inQueueX, offset, this->tileLength);
            LocalTensor<float> inLocal = inQueueX.DeQue<float>();
            LocalTensor<float> outLocal = outQueueZ.AllocTensor<float>();
            Compute(outLocal, inLocal, this->tileLength);
            inQueueX.FreeTensor(inLocal);
            outQueueZ.EnQue<float>(outLocal);
            CopyOut(outputGM, outQueueZ, offset, this->tileLength);
            offset = offset + this->tileLength;
        }
        // 处理每个核上的尾块
        if (this->tailLength != 0) {
            CopyIn(inputGM, inQueueX, offset, this->tailLength);
            LocalTensor<float> inLocal = inQueueX.DeQue<float>();
            LocalTensor<float> outLocal = outQueueZ.AllocTensor<float>();
            Compute(outLocal, inLocal, this->tailLength);
            inQueueX.FreeTensor(inLocal);
            outQueueZ.EnQue<float>(outLocal);
            CopyOut(outputGM, outQueueZ, offset, this->tailLength);
        }
    }

    __aicore__ inline void Compute(LocalTensor<float> &zLocal, LocalTensor<float> &xLocal, uint64_t tileLength)
    {
        //  z = x ^ 3
        Mul(zLocal, xLocal, xLocal, tileLength);
        PipeBarrier<PIPE_V>();
        Mul(zLocal, zLocal, xLocal, tileLength);
        PipeBarrier<PIPE_V>();

        // z = 0.044715 * z
        Muls(zLocal, zLocal, attr1, tileLength);
        PipeBarrier<PIPE_V>();

        // z = x + z
        Add(zLocal, xLocal, zLocal, tileLength);
        PipeBarrier<PIPE_V>();

        // z = -1.59576912 * z
        Muls(zLocal, zLocal, attr2, tileLength);
        PipeBarrier<PIPE_V>();

        // z = e ^ z
        Exp(zLocal, zLocal, tileLength);
        PipeBarrier<PIPE_V>();

        // z = 1 + z
        Adds(zLocal, zLocal, attr3, tileLength);
        PipeBarrier<PIPE_V>();

        // z = x / z
        Div(zLocal, xLocal, zLocal, tileLength);
        PipeBarrier<PIPE_V>();
    }

private:
    TBuf<AscendC::TPosition::VECCALC> inFp32Buffer;
    TBuf<AscendC::TPosition::VECCALC> outFp32Buffer;
    TQue<QuePosition::VECIN, AsdOps::GELU_FORWARD_BUFF_NUM> inQueueX;
    TQue<QuePosition::VECOUT, AsdOps::GELU_FORWARD_BUFF_NUM> outQueueZ;
    GlobalTensor<inType> inputGM;
    GlobalTensor<outType> outputGM;
    TPipe pipe;
    uint32_t blockLength{0};
    uint32_t tileNum{0};
    uint32_t tileLength{0};
    uint32_t tailLength{0};
    uint32_t bufferNum{2};
    const float attr1 = 0.044715;
    const float attr2 = -1.59576912;
    const float attr3 = 1.0;
};
#endif // OPP_FASTER_GELU_FORWARD_H