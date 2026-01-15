/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef SWI_GLU_QUANT_H
#define SWI_GLU_QUANT_H

#include "swi_glu_quant_base.h"

namespace AtbOps {
using namespace AscendC;

template <typename inType, typename outType>
class SwiGluQuant : public SwiGluQuantBase {
public:
    __aicore__ inline SwiGluQuant(TPipe *pipe)
    {
        pPipe = pipe;
    }

    __aicore__ inline void Init(GM_ADDR input_gm, GM_ADDR y_gm, GM_ADDR scale_gm,
        const SwiGluQuantTilingData *__restrict tilingData)
    {
        ParseTilingData(tilingData);
        InitParams();
        InitBaseBuffer();
        InitAndSetBuffer(input_gm, y_gm, scale_gm);
    }

    __aicore__ inline void Process()
    {
        DuplicateConst();
        ProcessCoreMultiUbMulti();
    }

private:
    __aicore__ inline void InitParams()
    {
        mergedColLen = SPLIT_NUM * tilingData_.colLen;
        colLen = tilingData_.colLen;
        basicColLen = tilingData_.basicColLen;
        coreIdx = static_cast<uint32_t>(GetBlockIdx());
        headCoreNum = tilingData_.headCoreNum;
        if (coreIdx < headCoreNum) {
            rowLenPerCore = tilingData_.rowLenPerHeadCore;
            basicRowLen = tilingData_.basicRowLenHeadCore;
            rowLoop = CeilDiv(rowLenPerCore, basicRowLen);
            baseRow = coreIdx * rowLenPerCore;
        } else if (coreIdx >= headCoreNum && coreIdx < tilingData_.realCoreNum) {
            rowLenPerCore = tilingData_.rowLenPerTailCore;
            basicRowLen = tilingData_.basicRowLenTailCore;
            rowLoop = CeilDiv(rowLenPerCore, basicRowLen);
            baseRow = headCoreNum * tilingData_.rowLenPerHeadCore + (coreIdx - headCoreNum) * rowLenPerCore;
        }
        outAlignLen = AlignUp(basicColLen, SWI_GLU_QUANT_THIRTY_TWO);
        outLen = basicRowLen * (outAlignLen == 0 ? (BLOCK_SIZE / sizeof(outType)) : outAlignLen);
        uint32_t alignedNum = BLOCK_SIZE / sizeof(inType);
        sizeHalfLen = AlignUp(basicColLen, alignedNum);
        tileLength = basicRowLen * (sizeHalfLen == 0 ? (BLOCK_SIZE / sizeof(inType)) : sizeHalfLen);
        rightPadding = sizeHalfLen - basicColLen;
        isPad = (rightPadding > 0);
        blockUnit = (isPad) ? 1 : BLOCK_SIZE;
    }

    __aicore__ inline void InitAndSetBuffer(GM_ADDR input_gm, GM_ADDR y_gm, GM_ADDR scale_gm)
    {
        // gm数据
        xGm.SetGlobalBuffer((__gm__ inType *)input_gm, SPLIT_NUM * tilingData_.rowLen * tilingData_.colLen);
        yGm.SetGlobalBuffer((__gm__ outType *)y_gm, tilingData_.rowLen * tilingData_.colLen);
        scale_Gm.SetGlobalBuffer((__gm__ float *)scale_gm, tilingData_.rowLen);
        // queue
        pPipe->InitBuffer(inQueueA, BUFFER_NUM, tileLength * sizeof(inType));
        pPipe->InitBuffer(inQueueB, BUFFER_NUM, tileLength * sizeof(inType));
        // todo
        pPipe->InitBuffer(outQueueY, BUFFER_NUM, outLen * sizeof(outType));
        pPipe->InitBuffer(scaleQueue, BUFFER_NUM, basicRowLen * sizeof(float));
        // 定义过程变量
        pPipe->InitBuffer(sharedTempBuf, tileLength * sizeof(float));
        pPipe->InitBuffer(tempBufferY, tileLength * sizeof(float));
        pPipe->InitBuffer(tempYUnit, sizeHalfLen * sizeof(float));
    }

    __aicore__ inline void ProcessCoreMultiUbMulti()
    {
        uint32_t offsetRow = 0;

        for (uint32_t ridx = 0; ridx < rowLoop; ridx++) {
            offsetRow = baseRow + ridx * basicRowLen;
            // 处理最后一行
            basicRowLenCal = static_cast<uint16_t>((ridx == rowLoop - 1)
                                                       ? (rowLenPerCore - (rowLoop - 1) * basicRowLen)
                                                       : basicRowLen);  // 每核处理的最后一个行循环单独处理
            ProcessCoreMultiUbMultiAlign(ridx, offsetRow);
        }
    }

    __aicore__ inline void ComputeVecInGmOffset(uint32_t ridx)
    {
        if (coreIdx < headCoreNum) {
            offsetParam.tmpVecGmOffset =
                static_cast<uint64_t>(coreIdx) * rowLenPerCore * mergedColLen + ridx * basicRowLen * mergedColLen;
            splitCopyoutOffset = coreIdx * rowLenPerCore * colLen + ridx * basicRowLen * basicColLen;
        } else {
            offsetParam.tmpVecGmOffset =
                static_cast<uint64_t>(headCoreNum) * tilingData_.rowLenPerHeadCore * mergedColLen +
                static_cast<uint64_t>(coreIdx - headCoreNum) * rowLenPerCore * mergedColLen +
                ridx * basicRowLen * mergedColLen;
            splitCopyoutOffset = headCoreNum * tilingData_.rowLenPerHeadCore * colLen +
                                 (coreIdx - headCoreNum) * rowLenPerCore * colLen + ridx * basicRowLen * basicColLen;
        }
    }

    __aicore__ inline void ProcessCoreMultiUbMultiAlign(uint32_t ridx, uint16_t offsetRow)
    {
        DataCopyParams splitCopyinParams;
        DataCopyParams splitCopyoutParams;
        splitCopyinParams = {basicRowLenCal, (uint16_t)(basicColLen * sizeof(inType) / blockUnit),
            (uint16_t)((mergedColLen - basicColLen) * sizeof(inType) / blockUnit), 0};
        splitCopyoutParams = {basicRowLenCal, (uint16_t)(basicColLen * sizeof(outType)), 0,
            (uint16_t)((colLen - basicColLen) * sizeof(outType))};
        ComputeVecInGmOffset(ridx);
        offsetParam.splitVecGmOffset1 = offsetParam.tmpVecGmOffset;
        offsetParam.splitVecGmOffset2 = offsetParam.splitVecGmOffset1 + tilingData_.colLen;
        CopyIn(offsetParam, splitCopyinParams);
        Compute(offsetRow);
        CopyOut(splitCopyoutOffset, splitCopyoutParams, ridx, basicRowLenCal);
    }

    __aicore__ inline void CopyIn(XxGluSingleTileOffsetParam &offsetParam, DataCopyParams &splitCopyinParams)
    {
        LocalTensor<inType> aLocal = this->inQueueA.template AllocTensor<inType>();
        LocalTensor<inType> bLocal = this->inQueueB.template AllocTensor<inType>();
        if (isPad) {
            DataCopyPadParams padParams{false, 0, rightPadding, 0};
            DataCopyPad(aLocal, this->xGm[offsetParam.splitVecGmOffset1], splitCopyinParams, padParams); // Copy A
            DataCopyPad(bLocal, this->xGm[offsetParam.splitVecGmOffset2], splitCopyinParams, padParams); // Copy B
        } else {
            DataCopy(aLocal, this->xGm[offsetParam.splitVecGmOffset1], splitCopyinParams); // Copy A
            DataCopy(bLocal, this->xGm[offsetParam.splitVecGmOffset2], splitCopyinParams); // Copy B
        }
        this->inQueueA.template EnQue(aLocal);
        this->inQueueB.template EnQue(bLocal);
    }

    __aicore__ inline void CopyOut(
        uint64_t splitCopyoutOffset, DataCopyParams &splitCopyoutParams, uint32_t ridx, uint32_t basicRowLenCal)
    {
        LocalTensor<outType> outLocal = outQueueY.DeQue<outType>();
        DataCopyPad(yGm[splitCopyoutOffset], outLocal, splitCopyoutParams);
        outQueueY.FreeTensor(outLocal);

        LocalTensor<float> scaleLocal = scaleQueue.DeQue<float>();
        DataCopyParams copyParams1{1, (uint16_t)(basicRowLenCal * sizeof(float)), 0, 0};
        uint32_t oooo = baseRow + basicRowLen * ridx;
        DataCopyPad(scale_Gm[oooo], scaleLocal, copyParams1);
        scaleQueue.FreeTensor(scaleLocal);
    }

    __aicore__ inline void Compute(uint32_t offsetRow)
    {
        LocalTensor<float> tmpALocal = sharedTempBuf.Get<float>();
        LocalTensor<float> tmpYLocal = tempBufferY.Get<float>();
        LocalTensor<inType> aLocal = inQueueA.template DeQue<inType>();
        if constexpr (sizeof(inType) == sizeof(float)) {
            DataCopy(tmpALocal, aLocal, tileLength);
        } else {
            Cast(tmpALocal, aLocal, RoundMode::CAST_NONE, tileLength);
        }
        PipeBarrier<PIPE_V>();
        inQueueA.template FreeTensor(aLocal);
        Muls(tmpYLocal, tmpALocal, static_cast<float>(-1.0), tileLength);
        PipeBarrier<PIPE_V>();
        Exp(tmpYLocal, tmpYLocal, tileLength);
        PipeBarrier<PIPE_V>();
        Adds(tmpYLocal, tmpYLocal, static_cast<float>(1.0), tileLength);
        PipeBarrier<PIPE_V>();
        Div(tmpYLocal, tmpALocal, tmpYLocal, tileLength);
        SetFlag<AscendC::HardEvent::V_S>(EVENT_ID1);
        WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID1);

        LocalTensor<inType> bLocal = inQueueB.template DeQue<inType>();
        LocalTensor<float> tmpBLocal = sharedTempBuf.Get<float>();
        if constexpr (sizeof(inType) == sizeof(float)) {
            DataCopy(tmpBLocal, bLocal, tileLength);
        } else {
            Cast(tmpBLocal, bLocal, RoundMode::CAST_NONE, tileLength);
        }
        PipeBarrier<PIPE_V>();
        inQueueB.template FreeTensor(bLocal);
        Mul(tmpYLocal, tmpYLocal, tmpBLocal, tileLength);
        PipeBarrier<PIPE_V>();
        ComputeQuant(tmpYLocal);
    }

    __aicore__ inline void ComputeQuant(LocalTensor<float> &tmpYLocal)
    {
        uint32_t index = 0; /**  quant相关的计算  */
        uint32_t realRowNum = 0;
        
        LocalTensor<float> scaleLocal = scaleQueue.AllocTensor<float>();
        LocalTensor<float> tempFp32 = tempYUnit.Get<float>();
        LocalTensor<float> tmpLocal = sharedTempBuf.Get<float>(sizeHalfLen);
        LocalTensor<int32_t> tempInt32 = sharedTempBuf.Get<int32_t>(sizeHalfLen);
        auto tempHalf = tempFp32.ReinterpretCast<half>();
        LocalTensor<outType> outLocal = outQueueY.AllocTensor<outType>();
        for (int32_t i = 0; i < basicRowLenCal; i++) {
            index = i * sizeHalfLen;
            DataCopy(tempFp32, tmpYLocal[index], sizeHalfLen);
            PipeBarrier<PIPE_V>();
            Abs(tmpLocal, tempFp32, basicColLen);
            PipeBarrier<PIPE_V>();
            ReduceMax(tmpLocal, tmpLocal, tmpLocal, basicColLen, false);
            PipeBarrier<PIPE_V>();
            Div(tmpLocal, constScale, tmpLocal, MAX_VALUE_NUM);
            SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            float scale = tmpLocal.GetValue(0);
            scaleLocal.SetValue(i, 1 / scale);
            Muls(tempFp32, tempFp32, scale, basicColLen);
            PipeBarrier<PIPE_V>();
            Cast(tempInt32, tempFp32, RoundMode::CAST_RINT, basicColLen);
            PipeBarrier<PIPE_V>();

            SetDeqScale(static_cast<half>(1.0));
            PipeBarrier<PIPE_V>();
            Cast(tempHalf, tempInt32, RoundMode::CAST_ROUND, basicColLen);
            PipeBarrier<PIPE_V>();

            Cast(outLocal[i * outAlignLen], tempHalf, RoundMode::CAST_TRUNC, basicColLen);
            PipeBarrier<PIPE_V>();
        }
        outQueueY.template EnQue<outType>(outLocal);
        scaleQueue.EnQue<float>(scaleLocal);
    }

private:
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueA;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueB;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    // quant
    TQue<QuePosition::VECOUT, BUFFER_NUM> scaleQueue;
    TBuf<TPosition::VECCALC> sharedTempBuf;
    TBuf<TPosition::VECCALC> tempBufferY;
    TBuf<TPosition::VECCALC> fp32_buf_, tempYUnit;
    GlobalTensor<inType> xGm;
    GlobalTensor<outType> yGm;
    GlobalTensor<float> scale_Gm;
    uint64_t splitCopyoutOffset;
};
}  // namespace SwiGluQuantOpt
#endif  // SWI_GLU_QUANT_H