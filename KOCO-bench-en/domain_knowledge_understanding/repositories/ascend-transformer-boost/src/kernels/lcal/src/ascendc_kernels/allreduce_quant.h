/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LCCL_ALLREDUCE_QUANT_H
#define LCCL_ALLREDUCE_QUANT_H
#include "collectives.h"
using namespace AscendC;

class AllReduceQuant : protected Collectives {
    constexpr static int32_t UB_HEAD_OFFSET = 96;
    constexpr static int32_t UB_MID_OFFSET = UB_HEAD_OFFSET + UB_SINGLE_PING_PONG_ADD_SIZE_MAX + ALIGN_SIZE;
public:
    FORCE_INLINE_AICORE AllReduceQuant(int rank, int rankSize, uint32_t extraFlag)
        : Collectives(rank, rankSize, extraFlag) {}

    template <typename T, typename U>
    FORCE_INLINE_AICORE void CpGM2GM(const GlobalTensor<T>& outputGT, const GlobalTensor<U>& inputGT,
        const uint32_t calCount, int op, T scale, T offset)
    {
        DataCopyGM2GM<T, U> cpKernel;
        cpKernel.Init(outputGT, inputGT, calCount, op);
        cpKernel.Process(scale, offset);
    }

    template <typename T, typename U>
    FORCE_INLINE_AICORE void CpGM2GM(const GlobalTensor<T>& outputGT, const GlobalTensor<U>& inputGT,
        const uint32_t calCount, int op, const GlobalTensor<T>& scaleGT, int64_t scaleCount, T offset)
    {
        DataCopyGM2GM<T, U> cpKernel;
        cpKernel.Init(outputGT, inputGT, calCount, op);
        cpKernel.Process(scaleGT, scaleCount, offset);
    }

    template <typename T, typename U>
    FORCE_INLINE_AICORE void CpGM2GMPingPong(int64_t dataSizeRemain, const GlobalTensor<U>& inputGT,
        const GlobalTensor<T>& outputGT, int op, T scale, T offset)
    {
        constexpr int32_t ubBlockSize = UB_SINGLE_PING_PONG_ADD_SIZE_MAX;
        constexpr int32_t ubAlignNum = ubBlockSize / (sizeof(T) + sizeof(U)) / ALIGN_SIZE * ALIGN_SIZE;
        constexpr int32_t inputUbBlockSize = std::is_same_v<T, U> ? ubBlockSize : ubAlignNum * sizeof(U);
        constexpr int32_t outputUbBlockSize = std::is_same_v<T, U> ? ubBlockSize : ubAlignNum * sizeof(T);
        __gm__ U *input = const_cast<__gm__ U *>(inputGT.GetPhyAddr());
        __gm__ T *output = const_cast<__gm__ T *>(outputGT.GetPhyAddr());
        __ubuf__ U* inputUB[2] = {(__ubuf__ U*)(UB_HEAD_OFFSET), (__ubuf__ U*)(UB_MID_OFFSET)};
        __ubuf__ T* outputUB[2] = {(__ubuf__ T*)(inputUB[0] + inputUbBlockSize / sizeof(U)),
            (__ubuf__ T*)(inputUB[1] + inputUbBlockSize / sizeof(U))};
        __ubuf__ T* targetOutputUB = nullptr;
        int inputOffsetNum = 0;
        int outputOffsetNum = 0;

        SetAtomic<T>(op);

        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (int64_t i = 0; dataSizeRemain > 0; i++) {
            uint32_t size = dataSizeRemain > outputUbBlockSize ? outputUbBlockSize : dataSizeRemain;
            event_t eventId = (i & 1) ? EVENT_ID0 : EVENT_ID1;
            targetOutputUB = (i & 1) ? outputUB[0] : outputUB[1];
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            CpGM2UB((i & 1) ? inputUB[0] : inputUB[1], input + inputOffsetNum, size / sizeof(T) * sizeof(U));
            SetWaitEvent<HardEvent::MTE2_V>(eventId);
            CastImpl(targetOutputUB, (i & 1) ? inputUB[0] : inputUB[1], RoundMode::CAST_NONE, size / sizeof(T));
            PipeBarrier<PIPE_V>();
            AddsImpl(targetOutputUB, targetOutputUB, offset, size / sizeof(T));
            PipeBarrier<PIPE_V>();
            MulsImpl(targetOutputUB, targetOutputUB, scale, size / sizeof(T));
            SetWaitEvent<HardEvent::V_MTE3>(eventId);
            SetWaitEvent<HardEvent::MTE2_MTE3>(eventId);
            CpUB2GM(output + outputOffsetNum, targetOutputUB, size);
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);

            dataSizeRemain -= size;
            inputOffsetNum += (size / sizeof(T));
            outputOffsetNum += (size / sizeof(T));
        }
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

        SetWaitEvent<HardEvent::MTE3_S>(EVENT_ID3);
        UnsetAtomic(op);
        return;
    }

    template <typename T, typename U>
    FORCE_INLINE_AICORE void CpGM2GMPingPong(int64_t dataSizeRemain, const GlobalTensor<U>& inputGT,
        const GlobalTensor<T>& outputGT, int op, const GlobalTensor<T>& scaleGT, int64_t scaleCount, T offset)
    {
        constexpr int32_t ubSplitSize = sizeof(T) + sizeof(U) + sizeof(T) + sizeof(U) + sizeof(T);
        constexpr int64_t ubAlignNum = UB_SINGLE_DMA_SIZE_MAX / ubSplitSize / ALIGN_SIZE * ALIGN_SIZE;
        __gm__ T *scale = const_cast<__gm__ T *>(scaleGT.GetPhyAddr());
        __gm__ U *input = const_cast<__gm__ U *>(inputGT.GetPhyAddr());
        __gm__ T *output = const_cast<__gm__ T *>(outputGT.GetPhyAddr());
        if (scaleCount > ubAlignNum) {
            CpGM2GMPingPongForBigScale(dataSizeRemain, input, output, op, scale, scaleCount, offset);
        } else {
            CpGM2GMPingPongForSmallScale(dataSizeRemain, input, output, op, scale, scaleCount, offset);
        }
        return;
    }

protected:
    template <typename T, typename U>
    FORCE_INLINE_AICORE void CpGM2GMPingPongForBigScale(int64_t dataSizeRemain, __gm__ U *input,
        __gm__ T *output, int op, __gm__ T *scale, int64_t scaleCount, T offset)
    {
        constexpr int64_t mulVal = 2;
        constexpr int64_t ubSplitSize = (sizeof(T) + sizeof(U) + sizeof(T)) * mulVal;
        constexpr int64_t ubAlignNum = UB_SINGLE_DMA_SIZE_MAX / ubSplitSize / ALIGN_SIZE * ALIGN_SIZE;
        const int64_t batchDataNum = (scaleCount + ubAlignNum - 1) / ubAlignNum;

        __ubuf__ T* scaleUB[2] = {(__ubuf__ T*)(UB_HEAD_OFFSET), (__ubuf__ T*)(UB_MID_OFFSET)};
        __ubuf__ U* inputUB[2] = {(__ubuf__ U*)(UB_HEAD_OFFSET + ubAlignNum * sizeof(T)),
            (__ubuf__ U*)(UB_MID_OFFSET + ubAlignNum * sizeof(T))};
        __ubuf__ T* outputUB[2] = {(__ubuf__ T*)(UB_HEAD_OFFSET + ubAlignNum * (sizeof(T) + sizeof(U))),
            (__ubuf__ T*)(UB_MID_OFFSET + ubAlignNum * (sizeof(T) + sizeof(U)))};
        __ubuf__ T* targetOutputUB = nullptr;
        int64_t i = 0;
        int32_t curDataNum = 0;
        int32_t processedNum = 0;

        SetAtomic<T>(op);

        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        while (dataSizeRemain > 0) {
            if (i % batchDataNum == batchDataNum - 1) {
                curDataNum = scaleCount - i % batchDataNum * ubAlignNum;
            } else {
                curDataNum = ubAlignNum;
            }
            event_t eventId = (i & 1) ? EVENT_ID0 : EVENT_ID1;
            targetOutputUB = (i & 1) ? outputUB[0] : outputUB[1];

            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            CpGM2UB((i & 1) ? inputUB[0] : inputUB[1], input + processedNum, curDataNum * sizeof(U));
            SetWaitEvent<HardEvent::MTE2_V>(eventId);
            CpGM2UB((i & 1) ? scaleUB[0] : scaleUB[1], scale + i % batchDataNum * ubAlignNum, curDataNum * sizeof(T));
            CastImpl(targetOutputUB, (i & 1) ? inputUB[0] : inputUB[1], RoundMode::CAST_NONE, curDataNum);
            SetWaitEvent<HardEvent::MTE2_V>(eventId);
            AddsImpl(targetOutputUB, targetOutputUB, offset, curDataNum);
            PipeBarrier<PIPE_V>();
            MulImpl(targetOutputUB, targetOutputUB, (i & 1) ? scaleUB[0] : scaleUB[1], curDataNum);
            SetWaitEvent<HardEvent::V_MTE3>(eventId);
            CpUB2GM(output + processedNum, targetOutputUB, curDataNum * sizeof(T));
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);

            dataSizeRemain -= curDataNum * sizeof(T);
            processedNum += curDataNum;
            ++i;
        }

        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        SetWaitEvent<HardEvent::MTE3_S>(EVENT_ID3);
        UnsetAtomic(op);
        return;
    }

    template <typename T, typename U>
    FORCE_INLINE_AICORE void CpGM2GMPingPongForSmallScale(int64_t dataSizeRemain, __gm__ U *input,
        __gm__ T *output, int op, __gm__ T *scale, int64_t scaleCount, T offset)
    {
        constexpr int32_t ubSplitSize = sizeof(T) + sizeof(U) + sizeof(T) + sizeof(U) + sizeof(T);
        constexpr int64_t ubAlignNum = UB_SINGLE_DMA_SIZE_MAX / ubSplitSize / ALIGN_SIZE * ALIGN_SIZE;
        const int64_t batchDataNum = ubAlignNum / scaleCount * scaleCount;
        const int64_t ubMidOffset = ubAlignNum * (sizeof(T) + sizeof(U) + sizeof(T)) + UB_HEAD_OFFSET + ALIGN_SIZE;

        __ubuf__ T* scaleUB = (__ubuf__ T*)(UB_HEAD_OFFSET);
        __ubuf__ U* inputUB[2] = {(__ubuf__ U*)(UB_HEAD_OFFSET + ubAlignNum * sizeof(T)), (__ubuf__ U*)(ubMidOffset)};
        __ubuf__ T* outputUB[2] = {(__ubuf__ T*)(UB_HEAD_OFFSET + ubAlignNum * (sizeof(T) + sizeof(U))),
            (__ubuf__ T*)(ubMidOffset + ubAlignNum * sizeof(U))};
        __ubuf__ T* targetOutputUB = nullptr;
        int64_t processedNum = 0;
        SetAtomic<T>(op);
        CpGM2UB(scaleUB, scale, scaleCount * sizeof(T));
        SetWaitEvent<HardEvent::MTE2_V>(EVENT_ID1);
        int64_t repeatTimes = batchDataNum / scaleCount;
        int64_t mulVal = 2;
        for (int64_t i = 1; i < repeatTimes; i *= mulVal) {
            PipeBarrier<PIPE_V>();
            CopyUB2UB(scaleUB + i * scaleCount, scaleUB, (repeatTimes > i * mulVal ? i : repeatTimes - i) * scaleCount);
        }
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (int64_t i = 0; dataSizeRemain > 0; i++) {
            uint32_t size = dataSizeRemain > batchDataNum * sizeof(T) ? batchDataNum * sizeof(T) : dataSizeRemain;
            event_t eventId = (i & 1) ? EVENT_ID0 : EVENT_ID1;
            targetOutputUB = (i & 1) ? outputUB[0] : outputUB[1];
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            CpGM2UB((i & 1) ? inputUB[0] : inputUB[1], input + processedNum, size / sizeof(T) * sizeof(U));
            SetWaitEvent<HardEvent::MTE2_V>(eventId);
            CastImpl(targetOutputUB, (i & 1) ? inputUB[0] : inputUB[1], RoundMode::CAST_NONE, size / sizeof(T));
            PipeBarrier<PIPE_V>();
            AddsImpl(targetOutputUB, targetOutputUB, offset, size / sizeof(T));
            PipeBarrier<PIPE_V>();
            MulImpl(targetOutputUB, targetOutputUB, scaleUB, size / sizeof(T));
            SetWaitEvent<HardEvent::V_MTE3>(eventId);
            CpUB2GM(output + processedNum, targetOutputUB, size);
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);
            dataSizeRemain -= size;
            processedNum += (size / sizeof(T));
        }
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        SetWaitEvent<HardEvent::MTE3_S>(EVENT_ID3);
        UnsetAtomic(op);
        return;
    }
};

#endif // LCCL_ALLREDUCE_QUANT_H