/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LCCL_DATACOPY_GM2GM_DELAY_H
#define LCCL_DATACOPY_GM2GM_DELAY_H
#include "datacopy_gm2gm.h"

using namespace AscendC;
using namespace Lcal;

template <typename V, typename T, typename U = T>
class DataCopyGM2GMDelay {
    constexpr static int64_t THREE_NUM = 3;
    constexpr static int64_t FOUR_NUM = 4;
    constexpr static int32_t WORK_OFFSET = 8192;
    constexpr static int32_t WORK_BLOCK_NUM = WORK_OFFSET / sizeof(T);
    constexpr static int32_t UB_HEAD_OFFSET = WORK_OFFSET * 2;
    constexpr static int32_t SCALE_SIZE = 32;
    constexpr static int32_t SCALE_NUM = SCALE_SIZE / sizeof(T);
    constexpr static int32_t SINGLE_SCALE_SIZE = 2;
    constexpr static int32_t BLOCK_NUM = (UB_SINGLE_DMA_SIZE_MAX - WORK_OFFSET * 2 - SCALE_SIZE * 4) / 2 /
                                         (sizeof(U) + sizeof(T)) / ALIGN_SIZE * ALIGN_SIZE;
    constexpr static int32_t IN_BLOCKSIZE = BLOCK_NUM * sizeof(U);

public:
    FORCE_INLINE_AICORE DataCopyGM2GMDelay() {}

    FORCE_INLINE_AICORE void Init(GlobalTensor<V>& outputGt, GlobalTensor<U> (&inputGt)[8],
        GlobalTensor<U> (&inputScaleGt)[8], const uint32_t calNum, int rankCount, GlobalTensor<U>& outScaleGt,
        TBuf<QuePosition::VECCALC> tbuf)
    {
        for (int index = 0; index < rankCount; index++) {
            this->inputGt[index] = inputGt[index];
            this->inputScaleGt[index] = inputScaleGt[index];
        }
        this->outputGt = outputGt;
        this->outScaleGt = outScaleGt;
        inTensor[0] = tbuf.GetWithOffset<U>(BLOCK_NUM, 0);
        inTensor[1] = tbuf.GetWithOffset<U>(BLOCK_NUM, WORK_OFFSET + SCALE_SIZE * HALF_NUM + IN_BLOCKSIZE * THREE_NUM);
        singleScaleUBTensor[0] = tbuf.GetWithOffset<T>(SCALE_NUM, IN_BLOCKSIZE);
        singleScaleUBTensor[1] = tbuf.GetWithOffset<T>(SCALE_NUM, WORK_OFFSET + SCALE_SIZE * HALF_NUM +
                                                        IN_BLOCKSIZE * FOUR_NUM);
        singleScaleUUBTensor[0] = tbuf.GetWithOffset<U>(SCALE_NUM, IN_BLOCKSIZE);
        singleScaleUUBTensor[1] = tbuf.GetWithOffset<U>(SCALE_NUM, WORK_OFFSET + SCALE_SIZE * HALF_NUM +
                                                        IN_BLOCKSIZE * FOUR_NUM);
        scaleUBTensor[0] = tbuf.GetWithOffset<T>(SCALE_NUM, IN_BLOCKSIZE + SCALE_SIZE);
        scaleUBTensor[1] = tbuf.GetWithOffset<T>(SCALE_NUM, WORK_OFFSET + SCALE_SIZE * THREE_NUM +
                                                        IN_BLOCKSIZE * FOUR_NUM);
        scaleUUBTensor[0] = tbuf.GetWithOffset<U>(SCALE_NUM, IN_BLOCKSIZE + SCALE_SIZE);
        scaleUUBTensor[1] = tbuf.GetWithOffset<U>(SCALE_NUM, WORK_OFFSET + SCALE_SIZE * THREE_NUM +
                                                        IN_BLOCKSIZE * FOUR_NUM);
        workUBTensor[0] = tbuf.GetWithOffset<T>(WORK_BLOCK_NUM, IN_BLOCKSIZE + SCALE_SIZE * HALF_NUM);
        workUBTensor[1] = tbuf.GetWithOffset<T>(WORK_BLOCK_NUM, WORK_OFFSET + SCALE_SIZE * FOUR_NUM +
                                                        IN_BLOCKSIZE * FOUR_NUM);
        outputUBTensor[0] = tbuf.GetWithOffset<T>(BLOCK_NUM, IN_BLOCKSIZE + SCALE_SIZE * HALF_NUM + WORK_OFFSET);
        outputUBTensor[1] = tbuf.GetWithOffset<T>(BLOCK_NUM, WORK_OFFSET * HALF_NUM + SCALE_SIZE * FOUR_NUM +
                                                        IN_BLOCKSIZE * FOUR_NUM);
        this->rankCount = rankCount;
        totalDataSize = calNum * sizeof(U);
        this->calNum = calNum;
        this->rankId = rankId;
    }

    FORCE_INLINE_AICORE void PreProcess()
    {
        for (int index = 0; index < rankCount; index++) {
            DataCopyWrap(scaleUUBTensor[0][index * SCALE_SIZE / sizeof(U)], inputScaleGt[index], SCALE_SIZE);
            pipe_barrier(PIPE_ALL);
            DataCopyWrap(scaleUUBTensor[1][index * SCALE_SIZE / sizeof(U)], inputScaleGt[index], SCALE_SIZE);
            pipe_barrier(PIPE_ALL);
        }
        for (int index = 0; index < rankCount; index++) {
            scaleUBTensor[0][index].SetValue(0, scaleUBTensor[0].GetValue(index * SCALE_SIZE / sizeof(T)));
            pipe_barrier(PIPE_ALL);
            scaleUBTensor[1][index].SetValue(0, scaleUBTensor[1].GetValue(index * SCALE_SIZE / sizeof(T)));
            pipe_barrier(PIPE_ALL);
            outputUBTensor[0][index].SetValue(0, 1);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        Div(scaleUBTensor[1], outputUBTensor[0], scaleUBTensor[1], rankCount);
        AscendC::PipeBarrier<PIPE_ALL>();
        ReduceMin<T>(singleScaleUBTensor[0], scaleUBTensor[0],
            workUBTensor[1][WORK_BLOCK_NUM / HALF_NUM], rankCount, false);
        pipe_barrier(PIPE_ALL);
        DataCopyWrap(outScaleGt, singleScaleUUBTensor[0], sizeof(T));
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    FORCE_INLINE_AICORE void LoopUncastAndMul(int idx, int index, event_t eventId)
    {
        PipeBarrier<PIPE_V>();
        T scalarValue = scaleUBTensor[1].GetValue(index);
        PipeBarrier<PIPE_V>();
        int32_t perRankNum;
        PipeBarrier<PIPE_V>();
        for (int j = 0; perRankNumRemain > 0; j++) {
            PipeBarrier<PIPE_V>();
            perRankNum = perRankNumRemain >= WORK_BLOCK_NUM ? WORK_BLOCK_NUM : perRankNumRemain;
            PipeBarrier<PIPE_V>();

            perRankNumRemain -= perRankNum;
            PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::S_V>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventId);
            PipeBarrier<PIPE_V>();
            Cast((idx & 1) ? workUBTensor[0] : workUBTensor[1], (idx & 1) ? inTensor[0][j *
                WORK_BLOCK_NUM] : inTensor[1][j * WORK_BLOCK_NUM], RoundMode::CAST_NONE, perRankNum);
            PipeBarrier<PIPE_V>();
            if (index == 0) {
                Muls<T>((idx & 1) ? outputUBTensor[0][j * WORK_BLOCK_NUM] : outputUBTensor[1][j *
                    WORK_BLOCK_NUM], (idx & 1) ? workUBTensor[0] : workUBTensor[1], scalarValue, perRankNum);
            } else {
                Axpy<T, T>((idx & 1) ? outputUBTensor[0][j * WORK_BLOCK_NUM] : outputUBTensor[1][j *
                    WORK_BLOCK_NUM], (idx & 1) ? workUBTensor[0] : workUBTensor[1], scalarValue, perRankNum);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    FORCE_INLINE_AICORE void Mte3Process(int idx, int index, int calCount, event_t eventId)
    {
        if (index == (rankCount - 1)) {
            if constexpr (std::is_same_v<V, T>) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventId);
                DataCopyWrap(outputGt[idx * BLOCK_NUM], (idx & 1) ?
                    outputUBTensor[0] : outputUBTensor[1], calCount * sizeof(V));
            }
            if constexpr (std::is_same_v<V, U>) {
                PipeBarrier<PIPE_V>();
                T scaleValue = singleScaleUBTensor[0].GetValue(0);
                PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::S_V>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(eventId);
                PipeBarrier<PIPE_V>();
                Muls<T>((idx & 1) ? outputUBTensor[0] : outputUBTensor[1], (idx & 1) ?
                    outputUBTensor[0] : outputUBTensor[1], scaleValue, calCount);
                PipeBarrier<PIPE_V>();
                Cast((idx & 1) ? inTensor[0] : inTensor[1], (idx & 1) ?
                    outputUBTensor[0] : outputUBTensor[1], RoundMode::CAST_NONE, calCount);
                PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eventId);
                DataCopyWrap(outputGt[idx * BLOCK_NUM], (idx & 1) ?
                    inTensor[0] : inTensor[1], calCount * sizeof(V));
            }
        }
    }

    FORCE_INLINE_AICORE int GetSize(int idx, int numOfPiece)
    {
        int size;
        if (idx < (numOfPiece - 1)) {
            size = IN_BLOCKSIZE;
        } else if (idx == (numOfPiece - 1)) {
            size = totalDataSize - (numOfPiece - 1) * IN_BLOCKSIZE;
        } else {
            size = 0;
        }
        return size;
    }

    FORCE_INLINE_AICORE void Process()
    {
        PreProcess();
        int numOfPiece = CeilDiv<int32_t, int32_t>(calNum, BLOCK_NUM);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1);
        for (int64_t i = 0; i < numOfPiece; i += HALF_NUM) {
            for (int index = 0; index < rankCount; index++) {
                for (int k = 0; k < HALF_NUM; k++) {
                    int idx = i + k;
                    int size = GetSize(idx, numOfPiece);
                    int32_t calCount = size / sizeof(U);
                    perRankNumRemain = calCount;
                    event_t eventId = (idx & 1) ? EVENT_ID0 : EVENT_ID1;
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(eventId);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(eventId);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(eventId);
                    DataCopyWrap((idx & 1) ? inTensor[0] : inTensor[1], inputGt[index][BLOCK_NUM * idx], size);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventId);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventId);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventId);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventId);
                    LoopUncastAndMul(idx, index, eventId);
                    Mte3Process(idx, index, calCount, eventId);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventId);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(eventId);
                }
            }
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID1);
    }
private:
    template <typename T1, typename T2>
    FORCE_INLINE_AICORE T1 CeilDiv(T1 a, T2 b)
    {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b;
    }

private:
    int64_t totalDataSize = 0;
    int rankCount;
    int perRankNumRemain;
    int calNum;
    int rankId;
    int numLayer;

    LocalTensor<U> inTensor[2];
    LocalTensor<U> singleScaleUUBTensor[2];
    LocalTensor<T> singleScaleUBTensor[2];
    LocalTensor<U> scaleUUBTensor[2];
    LocalTensor<T> scaleUBTensor[2];
    LocalTensor<T> workUBTensor[2];
    LocalTensor<T> outputUBTensor[2];

    GlobalTensor<V> outputGt;
    GlobalTensor<U> inputGt[8];
    GlobalTensor<U> inputScaleGt[8];
    GlobalTensor<U> outScaleGt;
};

#endif // LCCL_DATACOPY_GM2GM_DELAYH

