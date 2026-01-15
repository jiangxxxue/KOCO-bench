/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LCCL_DATACOPY_GM2GM_H
#define LCCL_DATACOPY_GM2GM_H
#include <type_traits>
#include "comm_args.h"

using namespace AscendC;
using namespace Lcal;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t TILE_NUM = 2;
constexpr int32_t BLOCK_SIZE = UB_SINGLE_DMA_SIZE_MAX / TILE_NUM / BUFFER_NUM;

template <typename T>
FORCE_INLINE_AICORE void SetAtomicOpType(int op)
{
    switch (op) {
        case ADD:
            AscendC::SetAtomicAdd<T>();
            break;

        case MUL:
            break;
        case MAX:
            AscendC::SetAtomicMax<T>();
            break;
        case MIN:
            AscendC::SetAtomicMin<T>();
            break;
        default:
            AscendC::SetAtomicNone();
            ;
    }
}

template <typename T>
FORCE_INLINE_AICORE void CpUB2GM(__gm__ T *gmAddr, __ubuf__ T *ubAddr, uint32_t size)
{
    LocalTensor<uint8_t> ubTensor;
    GlobalTensor<uint8_t> gmTensor;
    DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    ubTensor.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(ubAddr);
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(gmAddr));
    DataCopyPad(gmTensor, ubTensor, dataCopyParams);
}

template <typename T>
FORCE_INLINE_AICORE void CpGM2UB(__ubuf__ T *ubAddr, __gm__ T *gmAddr, uint32_t size)
{
    LocalTensor<uint8_t> ubTensor;
    GlobalTensor<uint8_t> gmTensor;
    DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    ubTensor.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(ubAddr);
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(gmAddr));
    DataCopyPadExtParams<uint8_t> padParams;
    DataCopyPad(ubTensor, gmTensor, dataCopyParams, padParams);
}


template<typename T>
FORCE_INLINE_AICORE void CopyUB2UB(__ubuf__ T *dst, __ubuf__ T *src, const uint32_t calCount)
{
    LocalTensor<T> srcTensor;
    LocalTensor<T> dstTensor;
    TBuffAddr srcAddr, dstAddr;
    srcAddr.bufferAddr = reinterpret_cast<uint64_t>(src);
    dstAddr.bufferAddr = reinterpret_cast<uint64_t>(dst);
    srcTensor.SetAddr(srcAddr);
    dstTensor.SetAddr(dstAddr);
    DataCopy(dstTensor, srcTensor, calCount);
}
template<typename T>
__aicore__ inline void DataCopyWrap(const GlobalTensor<T> &dstGlobal, const LocalTensor<T> &srcLocal,
                                    const uint32_t size)
{
    if (size % UB_ALIGN_SIZE == 0) {
        DataCopy(dstGlobal, srcLocal, size / sizeof(T));
    } else {
        DataCopyExtParams copyParams{1, size, 0, 0, 0};
        DataCopyPad(dstGlobal, srcLocal, copyParams);
    }
}

template<typename T>
__aicore__ inline void DataCopyWrap(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &srcGlobal,
                                    const uint32_t size)
{
    if (size % UB_ALIGN_SIZE == 0) {
        DataCopy(dstLocal, srcGlobal, size / sizeof(T));
    } else {
        DataCopyExtParams copyParams{1, size, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 1, 0};
        DataCopyPad(dstLocal, srcGlobal, copyParams, padParams);
    }
}

template <typename T, typename U = T>
class DataCopyGM2GM {
    constexpr static int32_t UB_HEAD_OFFSET = 64;
    constexpr static int32_t BLOCK_SIZE_PIECE = BLOCK_SIZE / (sizeof(T) + sizeof(U)) / ALIGN_SIZE * ALIGN_SIZE;
    constexpr static int32_t INPUT_BLOCK_SIZE = std::is_same_v<T, U> ? BLOCK_SIZE : BLOCK_SIZE_PIECE * sizeof(U);
    constexpr static int32_t OUTPUT_BLOCK_SIZE = std::is_same_v<T, U> ? BLOCK_SIZE : BLOCK_SIZE_PIECE * sizeof(T);
public:
    FORCE_INLINE_AICORE DataCopyGM2GM() {}
    FORCE_INLINE_AICORE void Init(const GlobalTensor<T>& outputGt, const GlobalTensor<U>& inputGt,
        const uint32_t calCount, int op)
    {
        inputGm = inputGt.GetPhyAddr();
        outputGm = outputGt.GetPhyAddr();
        inputUB = (__ubuf__ U*)(UB_HEAD_OFFSET);
        if constexpr (std::is_same_v<T, U>) {
            outputUB = (__ubuf__ T*)inputUB;
        } else {
            outputUB = (__ubuf__ T*)(UB_HEAD_OFFSET + INPUT_BLOCK_SIZE);
        }
        this->op = op;
        dataSizeRemain = calCount * sizeof(T);
    }

    FORCE_INLINE_AICORE void Process()
    {
        SetAtomic(op);
        int64_t i = 0;
        while (dataSizeRemain >= OUTPUT_BLOCK_SIZE) {
            CpGM2UB(inputUB, (__gm__ U*)inputGm + i * INPUT_BLOCK_SIZE / sizeof(U), INPUT_BLOCK_SIZE);
            AscendC::SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            if constexpr (!std::is_same_v<T, U>) {
                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                CastImpl(outputUB, inputUB, RoundMode::CAST_NONE, INPUT_BLOCK_SIZE / sizeof(U));
                AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            }
            CpUB2GM((__gm__ T*)outputGm + i * OUTPUT_BLOCK_SIZE / sizeof(T), (__ubuf__ T*)outputUB,
                OUTPUT_BLOCK_SIZE);
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            i += 1;
            dataSizeRemain -= OUTPUT_BLOCK_SIZE;
        }
        if (dataSizeRemain > 0) {
            CpGM2UB(inputUB, (__gm__ U*)inputGm + i * INPUT_BLOCK_SIZE / sizeof(U),
                dataSizeRemain / sizeof(T) * sizeof(U));
            AscendC::SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            if constexpr (!std::is_same_v<T, U>) {
                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                CastImpl(outputUB, inputUB, RoundMode::CAST_NONE, dataSizeRemain / sizeof(T));
                AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            }
            CpUB2GM((__gm__ T*)outputGm + i * OUTPUT_BLOCK_SIZE / sizeof(T), (__ubuf__ T*)outputUB,
                dataSizeRemain);
            PipeBarrier<PIPE_ALL>();
        }
        UnsetAtomic(op);
    }

    FORCE_INLINE_AICORE void Process(T scale, T offset)
    {
        SetAtomic(op);
        int64_t i = 0;
        int64_t batchDataNum = OUTPUT_BLOCK_SIZE / sizeof(T);
        while (dataSizeRemain > 0) {
            int64_t curProcessNum = (dataSizeRemain > OUTPUT_BLOCK_SIZE ? OUTPUT_BLOCK_SIZE : dataSizeRemain) /
                sizeof(T);
            CpGM2UB(inputUB, (__gm__ U*)inputGm + i * batchDataNum, curProcessNum * sizeof(U));
            AscendC::SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            if constexpr (!std::is_same_v<T, U>) {
                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                CastImpl(outputUB, inputUB, RoundMode::CAST_NONE, curProcessNum);
                PipeBarrier<PIPE_V>();
                AddsImpl(outputUB, outputUB, offset, curProcessNum);
                PipeBarrier<PIPE_V>();
                MulsImpl(outputUB, outputUB, scale, curProcessNum);
                AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            }
            CpUB2GM((__gm__ T*)outputGm + i * batchDataNum, (__ubuf__ T*)outputUB, curProcessNum * sizeof(T));
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            i += 1;
            dataSizeRemain -= OUTPUT_BLOCK_SIZE;
        }
        UnsetAtomic(op);
    }

    FORCE_INLINE_AICORE void Process(const GlobalTensor<T>& scaleGT, int64_t scaleCount, T offset)
    {
        if (scaleCount > UB_SINGLE_DMA_SIZE_MAX / (sizeof(T) + sizeof(U) + sizeof(T)) / ALIGN_SIZE * ALIGN_SIZE) {
            ProcessForBigScale(scaleGT, scaleCount, offset);
        } else {
            ProcessForSmallScale(scaleGT, scaleCount, offset);
        }
    }
private:
    FORCE_INLINE_AICORE void UnsetAtomic(int op)
    {
        if (op != -1) {
            AscendC::SetAtomicNone();
        }
        PipeBarrier<PIPE_ALL>();
    }

    FORCE_INLINE_AICORE void SetAtomic(int op)
    {
        PipeBarrier<PIPE_ALL>();
        if (op != -1) {
#ifdef __DAV_C220_VEC__
            SetAtomicOpType<T>(op);
#endif
        }
        PipeBarrier<PIPE_ALL>();
    }

    FORCE_INLINE_AICORE void ProcessForSmallScale(const GlobalTensor<T>& scaleGT, int64_t scaleCount, T offset)
    {
        SetAtomic(op);
        constexpr int32_t blockPieceNum = UB_SINGLE_DMA_SIZE_MAX / (sizeof(T) + sizeof(T) + sizeof(U)) / ALIGN_SIZE *
            ALIGN_SIZE;
        const int32_t batchDataNum = blockPieceNum / scaleCount * scaleCount;
        const int32_t inputBlockSize = batchDataNum * sizeof(U);
        const int32_t outputBlockSize = batchDataNum * sizeof(T);
        scaleUB = (__ubuf__ T*)(UB_HEAD_OFFSET);
        outputUB = (__ubuf__ T*)(scaleUB + blockPieceNum);
        inputUB = (__ubuf__ U*)(outputUB + blockPieceNum);
        __gm__ T *scale = const_cast<__gm__ T*>(scaleGT.GetPhyAddr());

        CpGM2UB((__ubuf__ T*)scaleUB, scale, scaleCount * sizeof(T));
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);

        int64_t repeatTimes = (dataSizeRemain > outputBlockSize ? outputBlockSize : dataSizeRemain) / sizeof(T) /
            scaleCount;
        int64_t mulVal = 2;
        for (int64_t i = 1; i < repeatTimes; i *= mulVal) {
            PipeBarrier<PIPE_V>();
            CopyUB2UB(scaleUB + i * scaleCount, scaleUB, (repeatTimes > i * mulVal ? i : repeatTimes - i) * scaleCount);
        }
        int64_t i = 0;
        while (dataSizeRemain > 0) {
            int64_t curProcessNum = (dataSizeRemain > outputBlockSize ? outputBlockSize : dataSizeRemain) / sizeof(T);
            CpGM2UB(inputUB, (__gm__ U*)inputGm + i * batchDataNum, curProcessNum * sizeof(U));
            AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            CastImpl(outputUB, inputUB, RoundMode::CAST_NONE, curProcessNum);
            PipeBarrier<PIPE_V>();
            AddsImpl(outputUB, outputUB, offset, curProcessNum);
            PipeBarrier<PIPE_V>();
            MulImpl(outputUB, outputUB, scaleUB, curProcessNum);
            AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            CpUB2GM((__gm__ T*)outputGm + i * batchDataNum, (__ubuf__ T*)outputUB, curProcessNum * sizeof(T));
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            i += 1;
            dataSizeRemain -= outputBlockSize;
        }
        UnsetAtomic(op);
    }

    FORCE_INLINE_AICORE void ProcessForBigScale(const GlobalTensor<T>& scaleGT, int64_t scaleCount, T offset)
    {
        SetAtomic(op);
        const int32_t blockPieceNum = UB_SINGLE_DMA_SIZE_MAX / (sizeof(T) + sizeof(U) + sizeof(T)) / ALIGN_SIZE *
            ALIGN_SIZE;
        const int32_t inputBlockSize = blockPieceNum * sizeof(U);
        const int32_t outputBlockSize = blockPieceNum * sizeof(T);
        const int32_t dataNumPerBatch = outputBlockSize / sizeof(T);
        const int32_t scaleBatchNum = (scaleCount + dataNumPerBatch - 1) / dataNumPerBatch;

        scaleUB = (__ubuf__ T*)(UB_HEAD_OFFSET);
        outputUB = (__ubuf__ T*)(scaleUB + outputBlockSize / sizeof(T));
        inputUB = (__ubuf__ U*)(outputUB + outputBlockSize / sizeof(T));
        __gm__ T *scale = const_cast<__gm__ T*>(scaleGT.GetPhyAddr());

        int64_t i = 0;
        int32_t curDataNum = 0;
        int32_t processedNum = 0;
        while (dataSizeRemain > 0) {
            if (i % scaleBatchNum == scaleBatchNum - 1) {
                curDataNum = scaleCount - i % scaleBatchNum * dataNumPerBatch;
            } else {
                curDataNum = dataNumPerBatch;
            }
            CpGM2UB(inputUB, (__gm__ U*)inputGm + processedNum, curDataNum * sizeof(U));
            AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            CpGM2UB(scaleUB, scale + i % scaleBatchNum * dataNumPerBatch, curDataNum * sizeof(T));
            CastImpl(outputUB, inputUB, RoundMode::CAST_NONE, curDataNum);
            AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            AddsImpl(outputUB, outputUB, offset, curDataNum);
            PipeBarrier<PIPE_V>();
            MulImpl(outputUB, outputUB, scaleUB, curDataNum);
            AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            CpUB2GM((__gm__ T*)outputGm + processedNum, (__ubuf__ T*)outputUB, curDataNum * sizeof(T));
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            i += 1;
            dataSizeRemain -= curDataNum * sizeof(T);
            processedNum += curDataNum;
        }
        UnsetAtomic(op);
    }
private:
    int64_t dataSizeRemain = 0;
    __ubuf__ U* inputUB = nullptr;
    __ubuf__ T* outputUB = nullptr;
    __ubuf__ T* scaleUB = nullptr;
    const __gm__ U* inputGm = nullptr;
    const __gm__ T* outputGm = nullptr;
    int op;
};
#endif // LCCL_DATACOPY_GM2GM_H