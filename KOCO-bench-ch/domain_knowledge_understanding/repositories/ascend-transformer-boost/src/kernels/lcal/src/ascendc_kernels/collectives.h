/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LCCL_COLLECTIVES_H
#define LCCL_COLLECTIVES_H

#include <climits>

#include "datacopy_gm2gm.h"
#include "datacopy_gm2gm_delay.h"
#include "sync_collectives.h"
using namespace AscendC;
using namespace Lcal;

#define KERNELS_ARGS_FUN() \
GM_ADDR input, GM_ADDR output, GM_ADDR commArgs, int64_t len, int64_t magic, int op, int root, int cycleCount, \
GM_ADDR scale, int64_t scaleCount, GM_ADDR offset

#define KERNELS_ARGS_CALL() \
input, output, commArgs, len, magic, op, root, cycleCount, scale, scaleCount, offset

#define KERNELS_GATHER_TABLE_ARGS_FUN() \
GM_ADDR embTable, GM_ADDR lookup, GM_ADDR revData, int64_t lookupLen, int64_t embTableLen, int64_t embTableDim

#define KERNELS_GATHER_TABLE_ARGS_CALL() \
embTable, lookup, revData, lookupLen, embTableLen, embTableDim

enum DfxPos : int {
    MAGIC,
    LEN,
    RUN_STATUS
};

class Collectives {
    constexpr static int32_t UB_HEAD_OFFSET = 96;
    constexpr static int32_t UB_MID_OFFSET = UB_HEAD_OFFSET + UB_SINGLE_PING_PONG_ADD_SIZE_MAX + ALIGN_SIZE;
public:
    FORCE_INLINE_AICORE Collectives(int rank, int rankSize, uint32_t extraFlag) : rank(rank), rankSize(rankSize),
        extraFlag(extraFlag) {}

    FORCE_INLINE_AICORE ~Collectives()
    {
        const int64_t notRunning = 0xdead;
        dfx.SetValue(RUN_STATUS, notRunning);
    }

    FORCE_INLINE_AICORE void Init(KERNELS_ARGS_FUN())
    {
        dumpAddr_ = (reinterpret_cast<__gm__ CommArgs *>(commArgs))->dumpAddr;
        GlobalTensor<GM_ADDR> peerMemsAddrGm;
        peerMemsAddrGm.SetGlobalBuffer(&(reinterpret_cast<__gm__ CommArgs *>(commArgs))->peerMems[0],
                                        LCAL_MAX_RANK_SIZE);
        for (int i = 0; i < rankSize; ++i) {
            shareAddrs[i] = peerMemsAddrGm.GetValue(i) +
                            (magic % PING_PONG_SIZE) * (IPC_BUFF_MAX_SIZE + IPC_DATA_OFFSET);
        }
        dfx.SetGlobalBuffer((reinterpret_cast<__gm__ CommArgs *>(commArgs))->dfx,
            DFX_COUNT);
        this->root = root;
        this->len = len;
        this->magic = magic;
        this->localRank = reinterpret_cast<__gm__ CommArgs *>(commArgs)->localRank;
        this->localRankSize = reinterpret_cast<__gm__ CommArgs *>(commArgs)->localRankSize;
        this->xRankSize = localRankSize;
        this->yRankSize = rankSize / localRankSize;
        this->xRankIdx = rank % localRankSize;
        this->yRankIdx = rank / localRankSize;

        blockIdx = GetBlockIdx();
        blockNum = GetBlockNum() * LCAL_BLOCK_NUM_MULTI;

        sync.Init(rank, rankSize, shareAddrs);
        dfx.SetValue(MAGIC, magic);
        dfx.SetValue(LEN, len);
        const int64_t running = 0xbeef;
        dfx.SetValue(RUN_STATUS, running);
    }

    template <typename T>
    FORCE_INLINE_AICORE void DataCopyWrapPingPong(const GlobalTensor<T>& inputGT, const GlobalTensor<T>& outputGT,
        int64_t dataSizeRemain, int op, TBuf<QuePosition::VECCALC> tbuf)
    {
        if (dataSizeRemain <= 0) {
            return;
        }
        LocalTensor<T> localUB[2];
        localUB[0] = tbuf.GetWithOffset<T>(UB_SINGLE_PING_PONG_ADD_SIZE_MAX, 0);
        localUB[1] = tbuf.GetWithOffset<T>(UB_SINGLE_PING_PONG_ADD_SIZE_MAX, UB_SINGLE_PING_PONG_ADD_SIZE_MAX);

        int inputOffsetNum = 0;
        int outputOffsetNum = 0;

        PipeBarrier<PIPE_ALL>();
        if (op != COPYONLY) {
            SetAscendCAtomic<T>(op);
        }
        PipeBarrier<PIPE_ALL>();

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (int64_t i = 0; dataSizeRemain > 0; i++) {
            uint32_t size = dataSizeRemain > UB_SINGLE_PING_PONG_ADD_SIZE_MAX ?
                UB_SINGLE_PING_PONG_ADD_SIZE_MAX : dataSizeRemain;
            TEventID eventId = (i & 1) ? EVENT_ID0 : EVENT_ID1;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            DataCopyWrap(localUB[(i & 1) ? 0 : 1], inputGT[inputOffsetNum], size);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
            DataCopyWrap(outputGT[outputOffsetNum], localUB[(i & 1) ? 0 : 1], size);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            dataSizeRemain -= size;
            inputOffsetNum += (size / sizeof(T));
            outputOffsetNum += (size / sizeof(T));
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID3);

        if (op != COPYONLY) {
            SetAtomicNone();
        }
        PipeBarrier<PIPE_ALL>();
    }

    template <typename T, typename U = T>
    FORCE_INLINE_AICORE void CpGM2GM(const GlobalTensor<T>& outputGT, const GlobalTensor<U>& inputGT,
        const uint32_t calCount, int op)
    {
        DataCopyGM2GM<T, U> cpKernel;
        cpKernel.Init(outputGT, inputGT, calCount, op);
        cpKernel.Process();
    }

    template <typename V, typename T, typename U = T>
    FORCE_INLINE_AICORE void CpGM2GMDelay(GlobalTensor<V>& outputGT, GlobalTensor<U> (&inputGT)[8],
        GlobalTensor<U> (&inputScaleGT)[8], const uint32_t calCount, int rankCount, GlobalTensor<U>& outScaleGT,
        TBuf<QuePosition::VECCALC> tbuf)
    {
        DataCopyGM2GMDelay<V, T, U> cpKernel;
        cpKernel.Init(outputGT, inputGT, inputScaleGT, calCount, rankCount, outScaleGT, tbuf);
        cpKernel.Process();
    }

    template <typename T1, typename T2>
    FORCE_INLINE_AICORE T1 CeilDiv(T1 a, T2 b)
    {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b;
    }

    template <typename T>
    FORCE_INLINE_AICORE void VecAddCce(int64_t curDealSize, __ubuf__ T *ubuf0, __ubuf__ T *ubuf1)
    {
        if (curDealSize > MAX_VADD_SIZE) {
            vadd(ubuf0, ubuf1, ubuf0, VADD_MAX_REPEAT, 1, 1, 1,
                VADD_UNIT_TO_BLOCK_UNIT_RATIO, VADD_UNIT_TO_BLOCK_UNIT_RATIO, VADD_UNIT_TO_BLOCK_UNIT_RATIO);
            vadd((__ubuf__ T*)((__ubuf__ int8_t*)ubuf0 + VADD_MAX_REPEAT * VADD_UNIT_BYTE),
                (__ubuf__ T*)((__ubuf__ int8_t*)ubuf1 + VADD_MAX_REPEAT * VADD_UNIT_BYTE),
                (__ubuf__ T*)((__ubuf__ int8_t*)ubuf0 + VADD_MAX_REPEAT * VADD_UNIT_BYTE),
                CeilDiv((curDealSize - MAX_VADD_SIZE), VADD_UNIT_BYTE), 1, 1, 1,
                VADD_UNIT_TO_BLOCK_UNIT_RATIO, VADD_UNIT_TO_BLOCK_UNIT_RATIO, VADD_UNIT_TO_BLOCK_UNIT_RATIO);
        } else {
            vadd(ubuf0, ubuf1, ubuf0, CeilDiv(curDealSize, VADD_UNIT_BYTE), 1, 1, 1,
                VADD_UNIT_TO_BLOCK_UNIT_RATIO, VADD_UNIT_TO_BLOCK_UNIT_RATIO, VADD_UNIT_TO_BLOCK_UNIT_RATIO);
        }
    }

    template <typename T>
    FORCE_INLINE_AICORE void LoopVaddCceProcess(__ubuf__ T* localUB[2], const int64_t remainSize,
        int64_t (&targetRankArr)[8], const int64_t targetRankArrValidSize, const int64_t srcIpcOffsetNum,
        __gm__ T *srcGmMem, __gm__ T *dstIpcMem, int64_t alreadyDealNum)
    {
        for (int64_t alreadyDealSize = 0; alreadyDealSize < remainSize;
             alreadyDealSize += UB_SINGLE_PING_PONG_ADD_SIZE_MAX) {
            int64_t curDealSize = UB_SINGLE_PING_PONG_ADD_SIZE_MAX;
            if (remainSize - alreadyDealSize < UB_SINGLE_PING_PONG_ADD_SIZE_MAX) {
                curDealSize = remainSize - alreadyDealSize;
            }
            if (alreadyDealSize != 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            CpGM2UB(localUB[0], srcGmMem + alreadyDealNum, curDealSize);

            for (int64_t i = 0; i < targetRankArrValidSize; i++) {
                int64_t targetRank = targetRankArr[i];
                if (targetRank == rank) {
                    continue;
                }
                if (i > 0 && !((targetRankArr[0] == rank) && i == 1)) {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                }
                CpGM2UB(localUB[1],
                        (__gm__ T*)(shareAddrs[targetRank] + IPC_DATA_OFFSET) + srcIpcOffsetNum + alreadyDealNum,
                    curDealSize);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
                VecAddCce(curDealSize, localUB[0], localUB[1]);
                if (((i + 1) == targetRankArrValidSize)) {
                    continue;
                }
                if ((i + 1 == targetRankArrValidSize - 1) && (targetRankArr[i + 1] == rank)) {
                    continue;
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            CpUB2GM((__gm__ T*)dstIpcMem + alreadyDealNum, localUB[0], curDealSize);
            if (alreadyDealSize + UB_SINGLE_PING_PONG_ADD_SIZE_MAX < remainSize) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            alreadyDealNum += curDealSize / sizeof(T);
        }
    }

    template <typename T>
    FORCE_INLINE_AICORE void LoopVaddCce(__ubuf__ T* localUB[2], const int64_t remainNum, int64_t (&targetRankArr)[8],
            int64_t targetRankArrValidSize, int64_t srcIpcOffsetNum, __gm__ T *srcGmMem, __gm__ T *dstIpcMem)
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        LoopVaddCceProcess(localUB, remainNum * (int64_t)sizeof(T), targetRankArr, targetRankArrValidSize,
                        srcIpcOffsetNum, srcGmMem, dstIpcMem, 0);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <typename T, typename U = T>
    FORCE_INLINE_AICORE void CpGM2GMPingPong(int64_t dataSizeRemain, const GlobalTensor<U>& inputGT,
                                            const GlobalTensor<T>& outputGT, int op)
    {
        constexpr int32_t ubBlockSize = UB_SINGLE_PING_PONG_ADD_SIZE_MAX;
        constexpr int32_t ubAlignNum = ubBlockSize / (sizeof(T) + sizeof(U)) / ALIGN_SIZE * ALIGN_SIZE;
        constexpr int32_t inputUbBlockSize = std::is_same_v<T, U> ? ubBlockSize : ubAlignNum * sizeof(U);
        constexpr int32_t outputUbBlockSize = std::is_same_v<T, U> ? ubBlockSize : ubAlignNum * sizeof(T);

        __gm__ U *input = const_cast<__gm__ U *>(inputGT.GetPhyAddr());
        __gm__ T *output = const_cast<__gm__ T *>(outputGT.GetPhyAddr());
        __ubuf__ U* inputUB[2] = {(__ubuf__ U*)(UB_HEAD_OFFSET), (__ubuf__ U*)(UB_MID_OFFSET)};
        __ubuf__ T* outputUB[2] = {(__ubuf__ T*)inputUB[0], (__ubuf__ T*)inputUB[1]};
        if constexpr (!std::is_same_v<T, U>) {
            outputUB[0] = (__ubuf__ T*)(inputUB[0] + inputUbBlockSize / sizeof(U));
            outputUB[1] = (__ubuf__ T*)(inputUB[1] + inputUbBlockSize / sizeof(U));
        }
        int inputOffsetNum = 0;
        int outputOffsetNum = 0;
        if (dataSizeRemain <= 0) {
            return;
        }

        SetAtomic<T>(op);

        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (int64_t i = 0; dataSizeRemain > 0; i++) {
            uint32_t size = dataSizeRemain > outputUbBlockSize ? outputUbBlockSize : dataSizeRemain;
            event_t eventId = (i & 1) ? EVENT_ID0 : EVENT_ID1;
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            CpGM2UB((i & 1) ? inputUB[0] : inputUB[1], input + inputOffsetNum, size / sizeof(T) * sizeof(U));
            if constexpr (!std::is_same_v<T, U>) {
                SetWaitEvent<HardEvent::MTE2_V>(eventId);
                CastImpl((i & 1) ? outputUB[0] : outputUB[1], (i & 1) ? inputUB[0] : inputUB[1], RoundMode::CAST_NONE,
                    size / sizeof(T));
                SetWaitEvent<HardEvent::V_MTE3>(eventId);
            }
            AscendC::SetFlag<HardEvent::MTE2_MTE3>(eventId);
            AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eventId);
            CpUB2GM(output + outputOffsetNum, (i & 1) ? outputUB[0] : outputUB[1], size);
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);
            dataSizeRemain -= size;
            inputOffsetNum += (size / sizeof(T));
            outputOffsetNum += (size / sizeof(T));
        }
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);

        AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
        AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
        UnsetAtomic(op);
        return;
    }

    template <typename T>
    FORCE_INLINE_AICORE void VecAdd(int64_t curDealSize, LocalTensor<T> &ubuf0, LocalTensor<T> &ubuf1)
    {
        if (curDealSize > MAX_VADD_SIZE) {
            Add<T, false>(ubuf0, ubuf1, ubuf0, MASK_PLACEHOLDER, VADD_MAX_REPEAT,
                {1, 1, 1, VADD_UNIT_TO_BLOCK_UNIT_RATIO, VADD_UNIT_TO_BLOCK_UNIT_RATIO, VADD_UNIT_TO_BLOCK_UNIT_RATIO});

            Add<T, false>(ubuf0[MAX_VADD_SIZE / sizeof(T)], ubuf1[MAX_VADD_SIZE / sizeof(T)],
                ubuf0[MAX_VADD_SIZE / sizeof(T)], MASK_PLACEHOLDER,
                CeilDiv((curDealSize - MAX_VADD_SIZE), VADD_UNIT_BYTE),
                {1, 1, 1, VADD_UNIT_TO_BLOCK_UNIT_RATIO, VADD_UNIT_TO_BLOCK_UNIT_RATIO, VADD_UNIT_TO_BLOCK_UNIT_RATIO});
        } else {
            Add<T, false>(ubuf0, ubuf1, ubuf0, MASK_PLACEHOLDER, CeilDiv(curDealSize, VADD_UNIT_BYTE),
                {1, 1, 1, VADD_UNIT_TO_BLOCK_UNIT_RATIO, VADD_UNIT_TO_BLOCK_UNIT_RATIO, VADD_UNIT_TO_BLOCK_UNIT_RATIO});
        }
    }

    template <typename T>
    FORCE_INLINE_AICORE void LoopVadd(TBuf<QuePosition::VECCALC> tbuf, int64_t &remainNum, int64_t (&targetRankArr)[8],
            int64_t targetRankArrValidSize, int64_t srcIpcOffsetNum, const GlobalTensor<T> &srcGt,
            const GlobalTensor<T> &dstGt)
    {
        if (remainNum <= 0) {
            return;
        }
        LocalTensor<T> localUB[2];
        localUB[0] = tbuf.GetWithOffset<T>(95 * 1024, 0);
        localUB[1] = tbuf.GetWithOffset<T>(95 * 1024, 95 * 1024);

        AscendC::PipeBarrier<PIPE_ALL>();
        LoopVaddProcess(localUB, remainNum * sizeof(T), targetRankArr, targetRankArrValidSize,
                           srcIpcOffsetNum, srcGt, dstGt, 0);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    template <typename T>
    FORCE_INLINE_AICORE void LoopVaddProcess(LocalTensor<T> (&localUB)[2], const int64_t remainSize,
        int64_t (&targetRankArr)[8], const int64_t targetRankArrValidSize, const int64_t srcIpcOffsetNum,
        const GlobalTensor<T> &srcGt, const GlobalTensor<T> &dstGt, int64_t alreadyDealNum)
    {
        for (int64_t alreadyDealSize = 0; alreadyDealSize < remainSize;
                alreadyDealSize += UB_SINGLE_PING_PONG_ADD_SIZE_MAX) {
            int64_t curDealSize = UB_SINGLE_PING_PONG_ADD_SIZE_MAX;
            if (remainSize - alreadyDealSize < UB_SINGLE_PING_PONG_ADD_SIZE_MAX) {
                curDealSize = remainSize - alreadyDealSize;
            }
            if (alreadyDealSize != 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            DataCopyWrap(localUB[0], srcGt[alreadyDealNum], curDealSize);

            for (int64_t i = 0; i < targetRankArrValidSize; i++) {
                int64_t targetRank = targetRankArr[i];
                if (targetRank == rank) {
                    continue;
                }
                if (i > 0 && !((targetRankArr[0] == rank) && i == 1)) {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                }
                GlobalTensor<T> srcGtTmp;
                srcGtTmp.SetGlobalBuffer(
                    (__gm__ T*)(shareAddrs[targetRank] + IPC_DATA_OFFSET) + srcIpcOffsetNum + alreadyDealNum);
                DataCopyWrap(localUB[1], srcGtTmp, curDealSize);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
                VecAdd(curDealSize, localUB[0], localUB[1]);
                if (((i + 1) == targetRankArrValidSize)) {
                    continue;
                }
                if ((i + 1 == targetRankArrValidSize - 1) && (targetRankArr[i + 1] == rank)) {
                    continue;
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyWrap(dstGt[alreadyDealNum], localUB[0], curDealSize);
            if (alreadyDealSize + UB_SINGLE_PING_PONG_ADD_SIZE_MAX < remainSize) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            alreadyDealNum += curDealSize / sizeof(T);
        }
    }

    template <typename T>
    FORCE_INLINE_AICORE void SetSingleValue2Gm(GM_ADDR gm, T value)
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        __ubuf__ T *inputUB = (__ubuf__ T *)(96);
        *inputUB = value;
        AscendC::PipeBarrier<PIPE_ALL>();
        CpUB2GM((__gm__ T *)gm, inputUB, sizeof(T));
        AscendC::PipeBarrier<PIPE_ALL>();
    }

protected:
    int rank;
    int rankSize;
    int localRank = 0;
    int localRankSize = 0;
    int xRankSize = 0;
    int yRankSize = 0;
    int xRankIdx = 0;
    int yRankIdx = 0;
    uint32_t extraFlag;
    int root;
    int64_t len;
    int64_t magic;
    int64_t blockIdx;
    int64_t blockNum;
    GM_ADDR shareAddrs[LCAL_MAX_RANK_SIZE];
    GlobalTensor<int64_t> dfx;
    SyncCollectives sync;
    GM_ADDR dumpAddr_ = nullptr;

    template <typename T>
    FORCE_INLINE_AICORE void SetAscendCAtomic(int op)
    {
        SetAtomicType<T>();
        switch (op) {
            case ADD:
                SetAtomicAdd<T>();
                return;
            case MUL:
                return;
            case MAX:
                SetAtomicMax<T>();
                return;
            case MIN:
                SetAtomicMin<T>();
                return;
            default:
                ;
        }
    }

    template <typename T>
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

    FORCE_INLINE_AICORE void UnsetAtomic(int op)
    {
        if (op != -1) {
            AscendC::SetAtomicNone();
        }
        PipeBarrier<PIPE_ALL>();
    }

    template<HardEvent eventType>
    FORCE_INLINE_AICORE void SetWaitEvent(event_t eventId)
    {
        AscendC::SetFlag<eventType>(eventId);
        AscendC::WaitFlag<eventType>(eventId);
    }

    FORCE_INLINE_AICORE void DumpLcclLogInfo(LogId logId, Op operationType)
    {
#ifdef ENABLE_LCCL_DUMP
        constexpr int32_t UB_HEAD_OFFSET = 96;

        AscendC::PipeBarrier<PIPE_ALL>();
        GM_ADDR blockGm = (GM_ADDR)(dumpAddr_ + LCCL_DUMP_UINT_SIZE * GetBlockIdx());
        __ubuf__ LcclDumpBlockInfo *blockUb = (__ubuf__ LcclDumpBlockInfo*)(UB_HEAD_OFFSET);
        __ubuf__ LcclDumpLogInfo *logUb = (__ubuf__ LcclDumpLogInfo*)(UB_HEAD_OFFSET + sizeof(LcclDumpBlockInfo));

        CpGM2UB((__ubuf__ uint8_t*)blockUb, blockGm, sizeof(LcclDumpBlockInfo));
        AscendC::PipeBarrier<PIPE_ALL>();

        if (blockUb->dumpOffset < sizeof(LcclDumpLogInfo)) {
            return;
        }

        logUb->logId = logId;
        logUb->blockId = GetBlockIdx();
        logUb->syscyc = static_cast<uint64_t>(GetSystemCycle());
        logUb->curPc = static_cast<uint64_t>(get_pc());
        logUb->operationType = operationType;
        logUb->rsv = 0;
        CpUB2GM((GM_ADDR)blockUb->dumpAddr, (__ubuf__ uint8_t*)logUb, sizeof(LcclDumpLogInfo));

        blockUb->dumpAddr += sizeof(LcclDumpBlockInfo);
        blockUb->dumpOffset -= sizeof(LcclDumpLogInfo);
        CpUB2GM(blockGm, (__ubuf__ uint8_t*)blockUb, sizeof(LcclDumpBlockInfo));
        AscendC::PipeBarrier<PIPE_ALL>();
#endif
    }
};

FORCE_INLINE_AICORE int64_t GetDataCount(const int64_t dataLen, const int64_t useBlockNum)
{
    return dataLen / useBlockNum;
}
#endif // LCCL_COLLECTIVES_H
