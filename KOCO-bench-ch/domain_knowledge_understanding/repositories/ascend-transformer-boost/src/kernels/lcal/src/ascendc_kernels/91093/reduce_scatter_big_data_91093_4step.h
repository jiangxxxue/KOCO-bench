/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LCCL_REDUCE_SCATTER_BIG_DATA_91093_H
#define LCCL_REDUCE_SCATTER_BIG_DATA_91093_H

#include "sync_collectives.h"
#include "collectives.h"
#include "ipc_queue.h"

constexpr int PER_STEP_BLOCKNUM = 8;
constexpr int ARRAY_MAX_SIZE = 10;
constexpr int NUM_OF_TWO = 2;
constexpr int NUM_OF_THREE = 3;
constexpr int NUM_OF_FOUR = 4;

template<typename T>
class ReduceScatterBigData91093 : protected Collectives {
public:
    __aicore__ inline ReduceScatterBigData91093(int rank, int rankSize, uint32_t extraFlag)
        : Collectives(rank, rankSize, extraFlag) {}
    __aicore__ inline void Init(KERNELS_ARGS_FUN())
    {
        Collectives::Init(KERNELS_ARGS_CALL());
        DumpLcclLogInfo(LogId::INIT, static_cast<Op>(op));
        constexpr int IPC_QUEUE_DEPTH_91093 = NUM_OF_FOUR;
        atomOp = op;
        relaBlockIdx = blockIdx % PER_STEP_BLOCKNUM;
        ipcSizeOfBlock = IPC_BUFF_MAX_SIZE / rankSize;
        ipcNumOfBlock = ipcSizeOfBlock / sizeof(T);
        ipcBlockNum = ipcNumOfBlock / IPC_QUEUE_DEPTH_91093;
        totalBlockDataNum = len;
        loopCount = CeilDiv(totalBlockDataNum, ipcBlockNum);
        dstOutputGlobal.SetGlobalBuffer((__gm__ T*)output, totalBlockDataNum);
        if ((rank % NUM_OF_TWO) == 0) {
            adjPeerRank = rank + 1;
        } else {
            adjPeerRank = rank - 1;
        }
        StepRankPerCoreInit();
        IpcQueueInit();
        if ((blockIdx / PER_STEP_BLOCKNUM) == 0) {
            for (int i = 0; i < stepOneRankPerCore; i++) {
                srcInputGlobal[i].SetGlobalBuffer((__gm__ T*)input + (blockIdx * stepOneOriginRankPerCore + i) *
                    totalBlockDataNum, totalBlockDataNum);
            }
        }
        DumpLcclLogInfo(LogId::INIT, static_cast<Op>(op));
    }

    __aicore__ inline void StepRankPerCoreInit()
    {
        int halfRankSize = rankSize / NUM_OF_TWO;
        stepOneOriginRankPerCore = CeilDiv(rankSize, PER_STEP_BLOCKNUM);
        stepTwoOriginRankPerCore = CeilDiv(halfRankSize, PER_STEP_BLOCKNUM);
        stepThreeOriginRankPerCore = CeilDiv(halfRankSize, PER_STEP_BLOCKNUM);
        stepOneInUseBlockNum = CeilDiv(rankSize, stepOneOriginRankPerCore);
        stepTwoInUseBlockNum = CeilDiv(halfRankSize, stepTwoOriginRankPerCore);
        stepThreeInUseBlockNum = CeilDiv(halfRankSize, stepThreeOriginRankPerCore);
        if ((blockIdx / PER_STEP_BLOCKNUM) == 0) {
            if ((blockIdx % PER_STEP_BLOCKNUM) == (stepOneInUseBlockNum - 1)) {
                stepOneRankPerCore = rankSize - (blockIdx % PER_STEP_BLOCKNUM) * stepOneOriginRankPerCore;
            } else {
                stepOneRankPerCore = stepOneOriginRankPerCore;
            }
        } else if ((blockIdx / PER_STEP_BLOCKNUM) == 1) {
            if ((blockIdx % PER_STEP_BLOCKNUM) == (stepTwoInUseBlockNum - 1)) {
                stepTwoRankPerCore = halfRankSize - (blockIdx % PER_STEP_BLOCKNUM) * stepTwoOriginRankPerCore;
            } else {
                stepTwoRankPerCore = stepTwoOriginRankPerCore;
            }
        } else if ((blockIdx / PER_STEP_BLOCKNUM) == NUM_OF_TWO || (blockIdx / PER_STEP_BLOCKNUM) == NUM_OF_THREE) {
            if (((blockIdx - PER_STEP_BLOCKNUM * NUM_OF_TWO) / NUM_OF_TWO) == (stepThreeInUseBlockNum - 1)) {
                stepThreeRankPerCore = halfRankSize - ((blockIdx - PER_STEP_BLOCKNUM * NUM_OF_TWO) /
                        NUM_OF_TWO) * stepThreeOriginRankPerCore;
            } else {
                stepThreeRankPerCore = stepThreeOriginRankPerCore;
            }
        }
    }

    __aicore__ inline void IpcQueueInit()
    {
        int ipcRank;
        if ((blockIdx / PER_STEP_BLOCKNUM) == 0) {
            for (int i = 0; i < stepOneRankPerCore; i++) {
                ipcRank = blockIdx * stepOneOriginRankPerCore + i;
                writeIpcQue[i].Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET + ipcRank *
                    ipcNumOfBlock * sizeof(T), ipcNumOfBlock, ipcBlockNum);
            }
        } else if ((blockIdx / PER_STEP_BLOCKNUM) == 1) {
            for (int i = 0; i < stepTwoRankPerCore; i++) {
                ipcRank = ((blockIdx % PER_STEP_BLOCKNUM) * stepTwoOriginRankPerCore + i) *
                    NUM_OF_TWO + (rank % NUM_OF_TWO);
                readIpcQue[i].Init(&sync, magic, shareAddrs[adjPeerRank] + IPC_DATA_OFFSET + ipcRank *
                    ipcNumOfBlock * sizeof(T), ipcNumOfBlock, ipcBlockNum);
                writeIpcQue[i].Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET + ipcRank *
                    ipcNumOfBlock * sizeof(T), ipcNumOfBlock, ipcBlockNum);
            }
        } else if ((blockIdx / PER_STEP_BLOCKNUM) == NUM_OF_TWO || (blockIdx / PER_STEP_BLOCKNUM) == NUM_OF_THREE) {
            for (int i = 0; i < stepThreeRankPerCore; i++) {
                stepThreeRank = (((blockIdx - PER_STEP_BLOCKNUM * NUM_OF_TWO) / NUM_OF_TWO) *
                    stepThreeOriginRankPerCore + i) * NUM_OF_TWO + (rank % NUM_OF_TWO);
                writeIpcQue[i].Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET + rank *
                    ipcNumOfBlock * sizeof(T), ipcNumOfBlock, ipcBlockNum);
                readIpcQue[i].Init(&sync, magic, shareAddrs[stepThreeRank] + IPC_DATA_OFFSET + rank *
                    ipcNumOfBlock * sizeof(T), ipcNumOfBlock, ipcBlockNum);
            }
        } else if (blockIdx == (NUM_OF_FOUR * PER_STEP_BLOCKNUM)) {
            readIpcQue[0].Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET + rank *
                ipcNumOfBlock * sizeof(T), ipcNumOfBlock, ipcBlockNum);
        }
    }

    __aicore__ inline void Process()
    {
        DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
        int stepIndex = blockIdx / PER_STEP_BLOCKNUM;
        if (stepIndex == 0 && ((relaBlockIdx * stepOneOriginRankPerCore) >= rankSize)) {
            DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
            return;
        }
        if (stepIndex == 1 && ((relaBlockIdx * stepTwoOriginRankPerCore) >= (rankSize / NUM_OF_TWO))) {
            DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
            return;
        }
        if ((stepIndex == NUM_OF_TWO || stepIndex == NUM_OF_THREE) && ((blockIdx - PER_STEP_BLOCKNUM *
            NUM_OF_TWO) / NUM_OF_TWO * stepThreeOriginRankPerCore) >= (rankSize / NUM_OF_TWO)) {
            DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
            return;
        }
        if (stepIndex == 0) {
            StepOneProcess();
        } else if (stepIndex == 1) {
            StepTwoProcess();
        } else if ((stepIndex == NUM_OF_TWO || stepIndex == NUM_OF_THREE) && ((blockIdx % NUM_OF_TWO) == 0)) {
            StepThreeProcess();
        } else if (blockIdx == (NUM_OF_FOUR * PER_STEP_BLOCKNUM)) {
            StepFourProcess();
        }
        DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
    }

    __aicore__ inline void StepOneProcess()
    {
        for (int i = 0; i < stepOneRankPerCore; i++) {
            if ((blockIdx * stepOneOriginRankPerCore + i) % NUM_OF_TWO == rank % NUM_OF_TWO) {
                if ((blockIdx * stepOneOriginRankPerCore + i) == rank) {
                    waitWriteRankArr[i] = rank;
                    waitWriteBlockArr[i] = PER_STEP_BLOCKNUM * NUM_OF_FOUR;
                } else {
                    waitWriteRankArr[i] = blockIdx * stepOneOriginRankPerCore + i;
                    waitWriteBlockArr[i] = PER_STEP_BLOCKNUM * NUM_OF_TWO + ((rank / NUM_OF_TWO) /
                            stepThreeOriginRankPerCore) * NUM_OF_TWO;
                }
            } else {
                waitWriteRankArr[i] = adjPeerRank;
                waitWriteBlockArr[i] = PER_STEP_BLOCKNUM + ((blockIdx * stepOneOriginRankPerCore + i) /
                        NUM_OF_TWO) / stepTwoOriginRankPerCore;
            }
        }
        InputToIpcProcess(waitWriteRankArr, waitWriteBlockArr, stepOneRankPerCore);
    }
    __aicore__ inline void StepTwoProcess()
    {
        int waitReadRank;
        int processRank;
        waitReadRank = adjPeerRank;
        for (int i = 0; i < stepTwoRankPerCore; i++) {
            processRank = (relaBlockIdx * stepTwoOriginRankPerCore + i) * NUM_OF_TWO + (rank % NUM_OF_TWO);
            waitReadRankArr[i] = waitReadRank;
            waitReadBlockArr[i] = processRank / stepOneOriginRankPerCore;
            if (processRank == rank) {
                waitWriteRankArr[i] = rank;
                waitWriteBlockArr[i] = PER_STEP_BLOCKNUM * NUM_OF_FOUR;
            } else {
                waitWriteRankArr[i] = processRank;
                waitWriteBlockArr[i] = PER_STEP_BLOCKNUM * NUM_OF_TWO + ((rank / NUM_OF_TWO) /
                    stepThreeOriginRankPerCore) * NUM_OF_TWO;
            }
        }
        SioAtomicToIpcProcess(waitReadRankArr, waitReadBlockArr, waitWriteRankArr,
            waitWriteBlockArr, stepTwoRankPerCore);
    }

    __aicore__ inline void StepThreeProcess()
    {
        for (int i = 0; i < stepThreeRankPerCore; i++) {
            waitReadRankArr[i] = (((blockIdx - PER_STEP_BLOCKNUM * NUM_OF_TWO) / NUM_OF_TWO) *
                    stepThreeOriginRankPerCore + i) * NUM_OF_TWO + (rank % NUM_OF_TWO);
            waitReadBlockArr[i] = PER_STEP_BLOCKNUM + (rank / NUM_OF_TWO) / stepTwoOriginRankPerCore;
            waitWriteRankArr[i] = rank;
            waitWriteBlockArr[i] = PER_STEP_BLOCKNUM * NUM_OF_FOUR;
        }
        HccsAtomicToIpcProcess(waitReadRankArr, waitReadBlockArr, waitWriteRankArr,
            waitWriteBlockArr, stepThreeRankPerCore);
    }

    __aicore__ inline void StepFourProcess()
    {
        for (int i = 0; i < stepThreeInUseBlockNum; i++) {
            waitReadRankArr[i] = rank;
            waitReadBlockArr[i] = PER_STEP_BLOCKNUM * NUM_OF_TWO + i * NUM_OF_TWO;
        }
        IpcToOutputProcess(waitReadRankArr, waitReadBlockArr, stepThreeInUseBlockNum);
    }

    __aicore__ inline void InputToIpcProcess(int *waitWriteRank, int *waitWriteBlock, int waitCount)
    {
        int processBlockNum = ipcBlockNum;
        for (int count = 0; count < loopCount; count++) {
            if (count == (loopCount - 1)) {
                processBlockNum = totalBlockDataNum - ipcBlockNum * count;
            }
            for (int i = 0; i < waitCount; i++) {
                writeIpcQue[i].DeQue(waitWriteRank[i], waitWriteBlock[i]);
                dstIpcGlobal = writeIpcQue[i].EnQue();
                CpInputToIpc(count, processBlockNum, srcInputGlobal[i]);
            }
            sync.SetInnerFlag(magic, count);
        }
    }

    __aicore__ inline void SioAtomicToIpcProcess(int *waitReadRank, int *waitReadBlock,  int *waitWriteRank,
                                                int *waitWriteBlock, int waitCount)
    {
        int processBlockNum = ipcBlockNum;
        for (int count = 0; count < loopCount; count++) {
            if (count == (loopCount - 1)) {
                processBlockNum = totalBlockDataNum - ipcBlockNum * count;
            }
            for (int i = 0; i < waitCount; i++) {
                srcIpcGlobal = readIpcQue[i].ReadFront();
                sync.WaitInnerFlag(magic, count, waitReadRank[i], waitReadBlock[i]);
                sync.WaitInnerFlag(magic, count, rank, waitReadBlock[i]);
                writeIpcQue[i].DeQue(waitWriteRank[i], waitWriteBlock[i]);
                dstIpcGlobal = writeIpcQue[i].EnQue();
                SioAtomicAddToIpc(count, processBlockNum, waitWriteRankArr[i], i);
            }
            sync.SetInnerFlag(magic, count);
        }
    }

    __aicore__ inline void HccsAtomicToIpcProcess(int *waitReadRank, int *waitReadBlock, int *waitWriteRank,
                                                int *waitWriteBlock, int waitCount)
    {
        int processBlockNum = ipcBlockNum;
        for (int count = 0; count < loopCount; count++) {
            if (count == (loopCount - 1)) {
                processBlockNum = totalBlockDataNum - ipcBlockNum * count;
            }
            for (int i = 0; i < waitCount; i++) {
                sync.WaitInnerFlag(magic, count, waitReadRank[i], waitReadBlock[i]);
                sync.WaitInnerFlag(magic, count, rank, waitReadBlock[i]);
                srcIpcGlobal = readIpcQue[i].ReadFront();
                writeIpcQue[i].DeQue(waitWriteRank[i], waitWriteBlock[i]);
                dstIpcGlobal = writeIpcQue[i].EnQue();
                HccsAtomicAddToIpc(count, processBlockNum, waitReadRank[i], i);
            }
            sync.SetInnerFlag(magic, count);
        }
    }

    __aicore__ inline void IpcToOutputProcess(int *waitReadRank, int *waitReadBlock, int waitCount)
    {
        int processBlockNum = ipcBlockNum;
        for (int count = 0; count < loopCount; count++) {
            if (count == (loopCount - 1)) {
                processBlockNum = totalBlockDataNum - ipcBlockNum * count;
            }
            for (int i = 0; i < waitCount; i++) {
                sync.WaitInnerFlag(magic, count, waitReadRank[i], waitReadBlock[i]);
            }
            srcIpcGlobal = readIpcQue[0].ReadFront();
            CpIpcToOutput(count, processBlockNum);
            sync.SetInnerFlag(magic, count);
        }
    }

protected:
    GlobalTensor<T> srcInputGlobal[ARRAY_MAX_SIZE];
    GlobalTensor<T> srcIpcGlobal;
    GlobalTensor<T> dstIpcGlobal;
    GlobalTensor<T> dstOutputGlobal;

    int totalBlockDataNum;
    int atomOp;
    int relaBlockIdx;
    int ipcBlockNum;
    int loopCount;
    int ipcNumOfBlock;
    int ipcSizeOfBlock;
    IpcQueue<T> writeIpcQue[ARRAY_MAX_SIZE];
    IpcQueue<T> readIpcQue[ARRAY_MAX_SIZE];
    int adjPeerRank;
    int stepThreeRank;
    int stepOneRankPerCore;
    int stepTwoRankPerCore;
    int stepThreeRankPerCore;
    int stepOneOriginRankPerCore;
    int stepTwoOriginRankPerCore;
    int stepThreeOriginRankPerCore;
    int stepOneInUseBlockNum;
    int stepTwoInUseBlockNum;
    int stepThreeInUseBlockNum;
    int waitWriteRankArr[ARRAY_MAX_SIZE];
    int waitWriteBlockArr[ARRAY_MAX_SIZE];
    int waitReadRankArr[ARRAY_MAX_SIZE];
    int waitReadBlockArr[ARRAY_MAX_SIZE];

private:
    __aicore__ inline void HccsAtomicAddToIpc(int num, int processBlockNum, int waitRank, int i)
    {
        if (waitRank != rank) {
            CpGM2GMPingPong<T>(processBlockNum * sizeof(T), srcIpcGlobal, dstIpcGlobal, atomOp);
        }
    }

    __aicore__ inline void CpInputToIpc(int num, int processBlockNum, GlobalTensor<T> inputTensor)
    {
        CpGM2GMPingPong<T>(processBlockNum * sizeof(T), inputTensor[num * ipcBlockNum], dstIpcGlobal, -1);
    }

    __aicore__ inline void SioAtomicAddToIpc(int num, int processBlockNum, int processRank, int i)
    {
        CpGM2GMPingPong<T>(processBlockNum * sizeof(T), srcIpcGlobal, dstIpcGlobal, atomOp);
    }

    __aicore__ inline void CpIpcToOutput(int num, int processBlockNum)
    {
        CpGM2GMPingPong<T>(processBlockNum * sizeof(T), srcIpcGlobal, dstOutputGlobal[num * ipcBlockNum], -1);
    }
};
#endif // LCCL_REDUCE_SCATTER_BIG_DATA_91093_H
