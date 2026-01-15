/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LCCL_ALLREDUCE_BIG_DATA_H
#define LCCL_ALLREDUCE_BIG_DATA_H

#include "allreduce_quant.h"
#include "sync_collectives.h"
#include "ipc_queue.h"
using namespace AscendC;

template <typename T, typename U = T>
class AllReduceBigData : protected AllReduceQuant {
    constexpr static int QUEUE_DEPTH = 4;
    constexpr static T oneCast = (T) 1;

public:
    FORCE_INLINE_AICORE AllReduceBigData(int rank, int rankSize, uint32_t extraFlag)
        : AllReduceQuant(rank, rankSize, extraFlag) {}
    FORCE_INLINE_AICORE void Init(KERNELS_ARGS_FUN())
    {
        Collectives::Init(KERNELS_ARGS_CALL());
        DumpLcclLogInfo(LogId::INIT, static_cast<Op>(op));
        if constexpr(!std::is_same_v<T, U>) {
            BuildScaleOffset(scale, scaleCount, offset);
            this->input = input;
            this->output = output;
        }

        if (blockIdx >= PING_PONG_SIZE * rankSize) {
            DumpLcclLogInfo(LogId::INIT, static_cast<Op>(op));
            return;
        }

        perStepBlockNum = rankSize;

        __gm__ CommArgs *localArgs = reinterpret_cast<__gm__ CommArgs *>(commArgs);
        int globalRankSize = localArgs->rankSize <= 0 ? rankSize : localArgs->rankSize;
        int localRankSize = localArgs->localRankSize <= 0 ? rankSize : localArgs->localRankSize;
        int serverNum = globalRankSize / localRankSize;
        int64_t ipcBuffMaxSizeAligned = IPC_BUFF_MAX_SIZE / (globalRankSize + serverNum - 1) /
            QUEUE_DEPTH / sizeof(T) /scaleNum * scaleNum * QUEUE_DEPTH * sizeof(T) * globalRankSize;
        curBlockSize = ipcBuffMaxSizeAligned / localRankSize / QUEUE_DEPTH;
        curBlockNum = curBlockSize / sizeof(T);
        atomOp = op;
        int64_t perQueSize = ipcBuffMaxSizeAligned / localRankSize;
        int64_t perQueNum = perQueSize / sizeof(T);

        for (int i = 0; i < rankSize; ++i) {
            rankList[i] = i;
            coreIdxList[i] = rankSize + blockIdx % perStepBlockNum;
        }

        peerRank = blockIdx % perStepBlockNum;
        perRankDataNum = GetDataCount(len, rankSize) / scaleNum * scaleNum;

        curRankDataNum = perRankDataNum;
        if (blockIdx % perStepBlockNum == rankSize - 1) {
            curRankDataNum = len - (rankSize - 1) * perRankDataNum;
        }

        pullRankDataNum = (rank == rankSize - 1) ? (len - rank * perRankDataNum) : perRankDataNum;

        inputBuffOffsetNum = blockIdx % rankSize * perRankDataNum;

        inputGt.SetGlobalBuffer((__gm__ U*)input + inputBuffOffsetNum, curRankDataNum);

        outputBuffOffsetNum = peerRank * perRankDataNum;

        outputGt.SetGlobalBuffer((__gm__ T*)output + outputBuffOffsetNum, curRankDataNum);

        inputIpcGtOffsetNum = perQueSize * (blockIdx % perStepBlockNum);

        if (blockIdx / perStepBlockNum == 0) {
            inputQue.Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET + inputIpcGtOffsetNum,
                perQueNum, curBlockNum);
        } else {
            srcQue.Init(&sync, magic, shareAddrs[peerRank] + IPC_DATA_OFFSET + rank * perQueSize,
                perQueNum, curBlockNum);
            dstQue.Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET + rank * perQueSize,
                perQueNum, curBlockNum);
            pullQue.Init(&sync, magic, shareAddrs[peerRank] + IPC_DATA_OFFSET + peerRank * perQueSize,
                perQueNum, curBlockNum);
        }
        DumpLcclLogInfo(LogId::INIT, static_cast<Op>(op));
    }

    FORCE_INLINE_AICORE void Process()
    {
        DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
        if (blockIdx >= PING_PONG_SIZE * rankSize) {
            DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
            return;
        }

        if constexpr (!std::is_same_v<T, U>) {
            if (rankSize == 1 && blockIdx == 0) {
                int64_t remain = curRankDataNum;
                int64_t loopCount = CeilDiv(curRankDataNum, curBlockNum);
                int64_t count = 0;
                while (count < loopCount) {
                    int64_t copyNum = (remain < curBlockNum) ? remain : curBlockNum;
                    Collectives::CpGM2GMPingPong(copyNum * sizeof(T), inputGt[count * curBlockNum],
                        outputGt[count * curBlockNum], COPYONLY);
                    remain -= curBlockNum;
                    ++count;
                }
            }
            if (rankSize == 1) {
                DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
                return;
            }
        }

        if (blockIdx / perStepBlockNum == 0) {
            Producer();
        } else {
            Consumer();
        }
        DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
    }

    FORCE_INLINE_AICORE void SupportBigScale()
    {
        if constexpr(!std::is_same_v<T, U>) {
            constexpr int32_t bigScaleFlagOffset = 2;
            if (blockIdx == 0) {
                inputGt.SetGlobalBuffer((__gm__ U*)input);
                outputGt.SetGlobalBuffer((__gm__ T*)output);
                CpGM2GMWithScale(len, inputGt, outputGt, COPYONLY);
                sync.SetSyncFlag(magic, 0, blockNum * bigScaleFlagOffset, rank);
            } else {
                sync.WaitSyncFlag(magic, 0, blockNum * bigScaleFlagOffset, rank);
            }
        }
        return;
    }
private:
    FORCE_INLINE_AICORE void Producer()
    {
        int64_t loopCount = CeilDiv(curRankDataNum, curBlockNum);
        int64_t remain = curRankDataNum;
        int count = 0;
        while (count < loopCount) {
            inputQue.DeQue(rankList, coreIdxList, rankSize);
            GlobalTensor<T> outputGm = inputQue.EnQue();
            int64_t copyNum = (remain < curBlockNum) ? remain : curBlockNum;
            if constexpr (std::is_same_v<T, U>) {
                Collectives::CpGM2GMPingPong(copyNum * sizeof(T), inputGt[count * curBlockNum], outputGm, COPYONLY);
            } else {
                if (blockIdx != rank) {
                    GlobalTensor<U> outputGmTmp;
                    outputGmTmp.SetGlobalBuffer((__gm__ U*)outputGm.GetPhyAddr());
                    Collectives::CpGM2GMPingPong(copyNum * sizeof(U), inputGt[count * curBlockNum], outputGmTmp,
                        COPYONLY);
                } else {
                    CpGM2GMWithScale(copyNum, inputGt[count * curBlockNum], outputGm, COPYONLY);
                }
            }
            sync.SetInnerFlag(magic, count);

            remain = remain - curBlockNum;
            count = count + 1;
        }
    }

    FORCE_INLINE_AICORE void Consumer()
    {
        int64_t atomLoopCount = CeilDiv(pullRankDataNum, curBlockNum);
        int64_t atomRemain = pullRankDataNum;
        int64_t loopCount = CeilDiv(curRankDataNum, curBlockNum);
        int64_t remain = curRankDataNum;
        int count = 0;
        while (count < loopCount || count < atomLoopCount) {
            if (peerRank != rank && count != atomLoopCount) {
                sync.WaitInnerFlag(magic, count, rank, rank);
                sync.WaitInnerFlag(magic, count, peerRank, rank);

                GlobalTensor<T> inputGm = srcQue.ReadFront();
                GlobalTensor<T> outputGm = dstQue.EnQue();

                int64_t atomCopyNum = (atomRemain < curBlockNum) ? atomRemain : curBlockNum;
                if constexpr (std::is_same_v<T, U>) {
                    Collectives::CpGM2GMPingPong(atomCopyNum * sizeof(T), inputGm, outputGm, atomOp);
                } else {
                    GlobalTensor<U> inputGmTmp;
                    inputGmTmp.SetGlobalBuffer((__gm__ U*)inputGm.GetPhyAddr());
                    CpGM2GMWithScale(atomCopyNum, inputGmTmp, outputGm, atomOp);
                    }
                    atomRemain = atomRemain - curBlockNum;
                }
                sync.SetOuterFlag(magic, count);
                if (count == loopCount) {
                    break;
                }
                sync.WaitOneRankPartOuterFlag(magic, count, peerRank, rankSize, rankSize);
                if (!(extraFlag & ExtraFlag::RDMA)) {
                    GlobalTensor<T> pullGm = pullQue.ReadFront();
                    int64_t copyNum = (remain < curBlockNum) ? remain : curBlockNum;
                    Collectives::CpGM2GMPingPong(copyNum * sizeof(T), pullGm, outputGt[count * curBlockNum], COPYONLY);
                }

                sync.SetInnerFlag(magic, count);
                remain = remain - curBlockNum;
                count = count + 1;
        }
    }

    FORCE_INLINE_AICORE void BuildScaleOffset(GM_ADDR scale, int64_t scaleCount, GM_ADDR offset)
    {
        if (scale != nullptr && offset != nullptr) {
            scaleGt.SetGlobalBuffer((__gm__ T*)scale);
            this->firstScale = scaleGt.GetValue(0);
            this->offset =* reinterpret_cast<__gm__ T*>(offset);
            this->scaleNum = scaleCount < 1 ? 1 : scaleCount;
            isVectorScale = scaleCount > 1;
            isEnableScale = scaleCount > 0 && !(*(uint16_t *)(&(this->offset)) == 0 &&
                scaleCount == 1 && *(uint16_t *)(&firstScale) == *(uint16_t *)(&oneCast));
        }
    }

   FORCE_INLINE_AICORE void CpGM2GMWithScale(int64_t atomCopyNum, GlobalTensor<U> inputGm, GlobalTensor<T> outputGm,
        int64_t atomOp)
    {
        if (!isEnableScale) {
                Collectives::CpGM2GMPingPong(atomCopyNum * sizeof(T), inputGm, outputGm, atomOp);
        } else if (!isVectorScale) {
            CpGM2GMPingPong(atomCopyNum * sizeof(T), inputGm, outputGm, atomOp, firstScale, offset);
        } else {
            CpGM2GMPingPong(atomCopyNum * sizeof(T), inputGm, outputGm, atomOp, scaleGt, scaleNum,
                offset);
        }
    }
private:
    GlobalTensor<U> inputGt;
    GlobalTensor<T> outputGt;

    int atomOp;

    int64_t perRankDataNum;
    int64_t curRankDataNum;
    int64_t peerRank;
    int64_t pullRankDataNum;
    int64_t inputBuffOffsetNum;
    int64_t outputBuffOffsetNum;
    int64_t inputIpcGtOffsetNum;
    int64_t curBlockSize;
    int64_t perStepBlockNum;
    int64_t curBlockNum;

    IpcQueue<T> inputQue;
    IpcQueue<T> srcQue;
    IpcQueue<T> dstQue;
    IpcQueue<T> pullQue;

    int rankList[LCAL_MAX_RANK_SIZE];
    int coreIdxList[LCAL_MAX_RANK_SIZE];

    GlobalTensor<T> scaleGt;
    int64_t scaleNum = 1;
    T firstScale = 1;
    T offset = 0;
    bool isEnableScale = false;
    bool isVectorScale = false;
    GM_ADDR input = nullptr;
    GM_ADDR output = nullptr;
};

#endif // LCCL_ALLREDUCE_BIG_DATA_H