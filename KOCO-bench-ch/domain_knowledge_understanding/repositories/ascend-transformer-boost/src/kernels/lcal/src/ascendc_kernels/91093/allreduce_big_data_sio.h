/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LCCL_ALLREDUCE_BIG_DATA_SIO_H
#define LCCL_ALLREDUCE_BIG_DATA_SIO_H

#include "collectives.h"
#include "sync_collectives.h"
#include "ipc_queue.h"
using namespace AscendC;

template <typename T>
class AllReduceBigDataSio : protected Collectives {
    constexpr static int QUEUE_DEPTH = 4;

public:
    FORCE_INLINE_AICORE AllReduceBigDataSio(int rank, int rankSize, uint32_t extraFlag)
        : Collectives(rank, rankSize, extraFlag) {}
    FORCE_INLINE_AICORE void Init(KERNELS_ARGS_FUN())
    {
        Collectives::Init(KERNELS_ARGS_CALL());
        DumpLcclLogInfo(LogId::INIT, static_cast<Op>(op));

        perStepBlockNum = rankSize;

        ipcBuffMaxSizeAligned = IPC_BUFF_MAX_SIZE / rankSize / QUEUE_DEPTH * rankSize * QUEUE_DEPTH;
        perQueSize = ipcBuffMaxSizeAligned / rankSize;
        perQueNum = perQueSize / sizeof(T);
        curBlockSize = perQueSize / QUEUE_DEPTH;
        curBlockNum = curBlockSize / sizeof(T);
        atomOp = op;
        for (int i = 0; i < rankSize; ++i) {
            rankList[i] = i;
            coreIdxList[i] = PING_PONG_SIZE * rankSize + blockIdx % perStepBlockNum;
        }

        peerRank = blockIdx % perStepBlockNum;
        perRankDataNum = len / rankSize;

        if (rank % RANK_SIZE_TWO == 0) {
            adjRank = rank + 1;
        } else {
            adjRank = rank - 1;
        }

        curRankDataNum = perRankDataNum;
        if (blockIdx % perStepBlockNum == rankSize - 1) {
            curRankDataNum = len - (rankSize - 1) * perRankDataNum;
        }
        pullRankDataNum = perRankDataNum;
        if (rank == rankSize - 1) {
            pullRankDataNum = len - rank * perRankDataNum;
        }
        inputBuffOffsetNum = blockIdx % rankSize * perRankDataNum;

        inputGt.SetGlobalBuffer((__gm__ T*)input + inputBuffOffsetNum, curRankDataNum);

        outputBuffOffsetNum = peerRank * perRankDataNum;

        outputGt.SetGlobalBuffer((__gm__ T*)output + outputBuffOffsetNum, curRankDataNum);
        inputIpcGtOffsetNum = perQueSize * (blockIdx % perStepBlockNum);

        if (blockIdx / perStepBlockNum == 0) {
            ProducerInit();
        } else if (blockIdx / perStepBlockNum == 1) {
            ConsumerInit();
        } else {
            PullerInit();
        }
        DumpLcclLogInfo(LogId::INIT, static_cast<Op>(op));
    }

    FORCE_INLINE_AICORE void Process()
    {
        DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
        if (blockIdx / perStepBlockNum == 0) {
            Producer();
        } else if (blockIdx / perStepBlockNum == 1) {
            Consumer();
        } else {
            Puller();
        }
        DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
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
            CpGM2GMPingPong(copyNum * sizeof(T), inputGt[count * curBlockNum], outputGm, COPYONLY);
            sync.SetOuterFlag(magic, count);

            if (blockIdx % RANK_SIZE_TWO == rank % RANK_SIZE_TWO) {
                sync.WaitOuterFlag(magic, count, rank, blockIdx);
                sync.WaitOuterFlag(magic, count, adjRank, blockIdx);
                GlobalTensor <T> inputGm = sioAtomSrcQue.ReadFront();
                GlobalTensor<T> outputGm = sioAtomDstQue.EnQue();
                CpGM2GMPingPong(copyNum * sizeof(T), inputGm, outputGm, atomOp);
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
        int64_t maxLoopCount = (loopCount < atomLoopCount) ? loopCount : atomLoopCount;
        while (count < maxLoopCount) {
            if (peerRank != rank && rank % RANK_SIZE_TWO == peerRank % RANK_SIZE_TWO && count != atomLoopCount) {
                sync.WaitInnerFlag(magic, count, rank, rank);
                sync.WaitInnerFlag(magic, count, peerRank, rank);

                GlobalTensor<T> inputGm = srcQue.ReadFront();
                GlobalTensor<T> outputGm = dstQue.EnQue();

                int64_t atomCopyNum = (atomRemain < curBlockNum) ? atomRemain : curBlockNum;
                CpGM2GMPingPong(atomCopyNum * sizeof(T), inputGm, outputGm, atomOp);
                atomRemain = atomRemain - curBlockNum;
            }
                sync.SetOuterFlag(magic, count);
                if (count == loopCount) {
                    break;
                }
                if (rank % RANK_SIZE_TWO == peerRank % RANK_SIZE_TWO) {
                    sync.WaitOneRankPartOuterFlag(magic, count, peerRank, perStepBlockNum, perStepBlockNum);
                    if (peerRank != rank) {
                        GlobalTensor <T> inputGm = pullSrcQue.ReadFront();
                        GlobalTensor <T> outputGm = pullDstQue.EnQue();
                        int64_t copyNum = (remain < curBlockNum) ? remain : curBlockNum;
                        CpGM2GMPingPong(copyNum * sizeof(T), inputGm, outputGm, COPYONLY);
                    }
                    sync.SetInnerFlag(magic, count);
                }
                remain = remain - curBlockNum;
                count = count + 1;
        }
    }
    FORCE_INLINE_AICORE void Puller()
    {
        int64_t loopCount = CeilDiv(curRankDataNum, curBlockNum);
        int64_t remain = curRankDataNum;
        int count = 0;
        while (count < loopCount) {
            if (rank % RANK_SIZE_TWO == peerRank % RANK_SIZE_TWO) {
                sync.WaitInnerFlag(magic, count, rank, blockIdx - perStepBlockNum);
            } else {
                sync.WaitInnerFlag(magic, count, adjRank, blockIdx - perStepBlockNum);
            }
            GlobalTensor<T> inputGm = pullQue.ReadFront();
            int64_t copyNum = (remain < curBlockNum) ? remain : curBlockNum;
            CpGM2GMPingPong(copyNum * sizeof(T), inputGm, outputGt[count * curBlockNum], COPYONLY);
            sync.SetInnerFlag(magic, count);
            remain = remain - curBlockNum;
            count = count + 1;
        }
    }

    FORCE_INLINE_AICORE void ProducerInit()
    {
        inputQue.Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET + inputIpcGtOffsetNum,
            perQueNum, curBlockNum);
        if (blockIdx % RANK_SIZE_TWO == rank % RANK_SIZE_TWO) {
            sioAtomSrcQue.Init(&sync, magic, shareAddrs[adjRank] + IPC_DATA_OFFSET + inputIpcGtOffsetNum,
                perQueNum, curBlockNum);
            sioAtomDstQue.Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET + inputIpcGtOffsetNum,
                perQueNum, curBlockNum);
        }
    }
    FORCE_INLINE_AICORE void ConsumerInit()
    {
        srcQue.Init(&sync, magic, shareAddrs[peerRank] + IPC_DATA_OFFSET + rank * perQueSize,
                    perQueNum, curBlockNum);
        dstQue.Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET + rank * perQueSize,
                    perQueNum, curBlockNum);
        if (peerRank != rank && rank % RANK_SIZE_TWO == peerRank % RANK_SIZE_TWO) {
            pullSrcQue.Init(&sync, magic, shareAddrs[peerRank] + IPC_DATA_OFFSET +
                            peerRank * perQueSize, perQueNum, curBlockNum);
            pullDstQue.Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET +
                            peerRank * perQueSize, perQueNum, curBlockNum);
        }
    }

    FORCE_INLINE_AICORE void PullerInit()
    {
        if (rank % RANK_SIZE_TWO == peerRank % RANK_SIZE_TWO) {
            pullQue.Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET + inputIpcGtOffsetNum,
                perQueNum, curBlockNum);
        } else {
            pullQue.Init(&sync, magic, shareAddrs[adjRank] + IPC_DATA_OFFSET + inputIpcGtOffsetNum,
                perQueNum, curBlockNum);
        }
    }
private:
    GlobalTensor<T> inputGt;
    GlobalTensor<T> outputGt;

    int atomOp;
    int64_t ipcBuffMaxSizeAligned;

    int64_t perRankDataNum;
    int64_t curRankDataNum;
    int64_t peerRank;
    int64_t adjRank;
    int64_t pullRankDataNum;
    int64_t inputBuffOffsetNum;
    int64_t outputBuffOffsetNum;
    int64_t inputIpcGtOffsetNum;
    int64_t curBlockSize;
    int64_t perStepBlockNum;
    int64_t curBlockNum;
    int64_t perQueSize;
    int64_t perQueNum;

    IpcQueue<T> inputQue;
    IpcQueue<T> srcQue;
    IpcQueue<T> dstQue;
    IpcQueue<T> pullQue;
    IpcQueue<T> sioAtomSrcQue;
    IpcQueue<T> sioAtomDstQue;
    IpcQueue<T> pullSrcQue;
    IpcQueue<T> pullDstQue;

    int rankList[LCAL_MAX_RANK_SIZE];
    int coreIdxList[LCAL_MAX_RANK_SIZE];
};

#endif // LCCL_ALLREDUCE_BIG_DATA_H