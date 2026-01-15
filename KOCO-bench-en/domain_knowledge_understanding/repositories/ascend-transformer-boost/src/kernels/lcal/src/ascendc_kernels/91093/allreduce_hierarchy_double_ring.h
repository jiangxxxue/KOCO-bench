/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LCCL_ALLREDUCE_HIERARCHY_DOUBLE_RING_H
#define LCCL_ALLREDUCE_HIERARCHY_DOUBLE_RING_H

#include "sync_collectives.h"
#include "collectives.h"
#include "ipc_queue.h"
using namespace AscendC;

template<typename T>
class AllReduceHierarchyDoubleRing : protected Collectives {
    constexpr static int32_t RING_LAYER_NUM = 2;
    constexpr static int32_t INPUT_CORE_NUM = 4;
    constexpr static int32_t SIO_CORE_NUM = 12;
    constexpr static int32_t RING_CORE_NUM = 12;
    constexpr static int32_t OUTPUT_CORE_NUM = 6;
    constexpr static int32_t IPC_QUE_DEPTH = 32;
    constexpr static int32_t RING_GATHER_QUE_DEPTH = 3;
    constexpr static int32_t SIO_GATHER_QUE_DEPTH = 2;
    constexpr static int32_t INPUT_FLAG = 0 * RING_CORE_NUM;
    constexpr static int32_t SIO_REDUCE_FLAG = 1 * RING_CORE_NUM;
    constexpr static int32_t RING_REDUCE_FLAG = 2 * RING_CORE_NUM;
    constexpr static int32_t RING_REDUCE_PEER_FLAG = 3 * RING_CORE_NUM;
    constexpr static int32_t RING_GATHER_FLAG = 4 * RING_CORE_NUM;
    constexpr static int32_t RING_GATHER_PEER_FLAG = 5 * RING_CORE_NUM;
    constexpr static int32_t SIO_GATHER_PEER_FLAG = 6 * RING_CORE_NUM;
    constexpr static int32_t SIO_GATHER_FLAG = 7 * RING_CORE_NUM;
    constexpr static int32_t SIO_GATHER_OUTPUT_FLAG = 8 * RING_CORE_NUM;
    constexpr static int32_t OUTPUT_FLAG = 9 * RING_CORE_NUM;
    constexpr static int32_t INPUT_CORE_SCALE = RING_CORE_NUM / INPUT_CORE_NUM;
    constexpr static int32_t SIO_CORE_SCALE = RING_CORE_NUM / SIO_CORE_NUM;
    constexpr static int32_t OUTPUT_CORE_SCALE = RING_CORE_NUM / OUTPUT_CORE_NUM;
    constexpr static int64_t BLOCK_NUM_ALIGN = BLOCK_SIZE / sizeof(T);

public:
    FORCE_INLINE_AICORE AllReduceHierarchyDoubleRing(int rank, int rankSize, uint32_t extraFlag)
        : Collectives(rank, rankSize, extraFlag) {}
    FORCE_INLINE_AICORE void Init(KERNELS_ARGS_FUN())
    {
        Collectives::Init(KERNELS_ARGS_CALL());
        atomOp = op;
        DumpLcclLogInfo(LogId::INIT, static_cast<Op>(atomOp));
        blockNum = INPUT_CORE_NUM + SIO_CORE_NUM + RING_CORE_NUM + OUTPUT_CORE_NUM;
        if (blockIdx >= blockNum) {
            DumpLcclLogInfo(LogId::INIT, static_cast<Op>(atomOp));
            return;
        }
        sioLayerId = rank / RING_LAYER_NUM;
        ringLayerId = rank % RING_LAYER_NUM;
        ringRankSize = rankSize / RING_LAYER_NUM;
        ringNextRankId = (sioLayerId + 1) % ringRankSize * RING_LAYER_NUM + ringLayerId;
        ringPrevRankId = (sioLayerId + (ringRankSize - 1)) % ringRankSize * RING_LAYER_NUM + ringLayerId;
        sioPeerRankId = sioLayerId * RING_LAYER_NUM + (ringLayerId + 1) % RING_LAYER_NUM;
        ipcBlockNum = IPC_BUFF_MAX_SIZE / (IPC_QUE_DEPTH + RING_GATHER_QUE_DEPTH + SIO_GATHER_QUE_DEPTH) / sizeof(T);
        dmaPerLoop = ipcBlockNum - rankSize;
        loopCount = CeilDiv(len, rankSize * dmaPerLoop);
        const int64_t sumDataLastLoop = len - (loopCount - 1) * rankSize * dmaPerLoop;
        dmaLastLoop = sumDataLastLoop / rankSize;
        dmaLastRankLoop = sumDataLastLoop - (rankSize - 1) * dmaLastLoop;
        totalBlockDataNum = (loopCount - 1) * dmaPerLoop + dmaLastLoop;

        InitQue();
        inputTensor.SetGlobalBuffer((__gm__ T*) input);
        outputTensor.SetGlobalBuffer((__gm__ T*) output);
        DumpLcclLogInfo(LogId::INIT, static_cast<Op>(atomOp));
    }

    FORCE_INLINE_AICORE void Process()
    {
        DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
        if (blockIdx >= blockNum) {
            DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
            return;
        }
        for (curLoopCnt = 0; curLoopCnt < loopCount; ++curLoopCnt) {
            for (sioLayerLoop = 0; sioLayerLoop < ringRankSize; ++sioLayerLoop) {
                if (blockIdx < INPUT_CORE_NUM) {
                    Input2Ipc();
                } else if (blockIdx < INPUT_CORE_NUM + SIO_CORE_NUM) {
                    SioReduce();
                } else if (blockIdx < INPUT_CORE_NUM + SIO_CORE_NUM + RING_CORE_NUM) {
                    RingReduce();
                } else {
                    PrepareOutput();
                }
                ++ipcQueIdx;
            }

            for (sioLayerLoop = 0; sioLayerLoop < ringRankSize; ++sioLayerLoop) {
                if (blockIdx < INPUT_CORE_NUM) {
                    ;
                } else if (blockIdx < INPUT_CORE_NUM + SIO_CORE_NUM) {
                    SioGather();
                } else if (blockIdx < INPUT_CORE_NUM + SIO_CORE_NUM + RING_CORE_NUM) {
                    RingGather();
                } else {
                    Ipc2Output();
                }
                ++gatherQueIdx;
            }
        }
        DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
    }
private:
    IpcQueue<T> inputQueList[INPUT_CORE_SCALE];
    IpcQueue<T> sioQueList[SIO_CORE_SCALE];
    IpcQueue<T> sioGatherSrc1QueList[SIO_CORE_SCALE];
    IpcQueue<T> sioGatherSrc2QueList[SIO_CORE_SCALE];
    IpcQueue<T> sioGatherDstQueList[SIO_CORE_SCALE];
    IpcQueue<T> ringSrcQue;
    IpcQueue<T> ringDstQue;
    IpcQueue<T> ringGatherSrcQue;
    IpcQueue<T> ringGatherDstQue;
    IpcQueue<T> outputSrc1QueList[OUTPUT_CORE_SCALE];
    IpcQueue<T> outputSrc2QueList[OUTPUT_CORE_SCALE];
    IpcQueue<T> outputSrc3QueList[OUTPUT_CORE_SCALE];

    IpcQueue<T> *inputQue = nullptr;
    IpcQueue<T> *sioQue = nullptr;
    IpcQueue<T> *sioGatherSrc1Que = nullptr;
    IpcQueue<T> *sioGatherSrc2Que = nullptr;
    IpcQueue<T> *sioGatherDstQue = nullptr;
    IpcQueue<T> *outputSrc1Que = nullptr;
    IpcQueue<T> *outputSrc2Que = nullptr;
    IpcQueue<T> *outputSrc3Que = nullptr;
    GlobalTensor<T> srcIpcTensor;
    GlobalTensor<T> dstIpcTensor;
    GlobalTensor<T> inputTensor;
    GlobalTensor<T> outputTensor;
    int atomOp = COPYONLY;
    int32_t sioLayerId = 0;
    int32_t ringLayerId = 0;
    int32_t ringRankSize = 0;
    int32_t ringNextRankId = 0;
    int32_t ringPrevRankId = 0;
    int32_t sioPeerRankId = 0;
    int32_t localBlockIdx = 0;
    int64_t ipcBlockNum = 0;
    int64_t totalBlockDataNum = 0;
    int64_t dmaPerLoop = 0;
    int64_t dmaLastLoop = 0;
    int64_t dmaLastRankLoop = 0;
    int32_t ipcQueIdx = 0;
    int32_t gatherQueIdx = 0;
    int32_t loopCount = 0;
    int32_t curLoopCnt = 0;
    int32_t sioLayerLoop = 0;
    int64_t coreDataNum = 0;
    int64_t lastCoreDataNum = 0;
    int64_t curCoreDataNum = 0;

    FORCE_INLINE_AICORE void InitQue()
    {
        const int64_t dmaSizePerCore = ipcBlockNum / RING_CORE_NUM * sizeof(T);
        const int64_t ipcBlockSize = ipcBlockNum * sizeof(T);
        if (blockIdx < INPUT_CORE_NUM) {
            for (int32_t blockLoop = 0; blockLoop < INPUT_CORE_SCALE; ++blockLoop) {
                localBlockIdx = blockIdx * INPUT_CORE_SCALE + blockLoop;
                inputQueList[blockLoop].Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET +
                    dmaSizePerCore * localBlockIdx, ipcBlockNum * IPC_QUE_DEPTH, ipcBlockNum);
            }
        } else if (blockIdx < INPUT_CORE_NUM + SIO_CORE_NUM) {
            for (int32_t blockLoop = 0; blockLoop < SIO_CORE_SCALE; ++blockLoop) {
                localBlockIdx = (blockIdx - INPUT_CORE_NUM) * SIO_CORE_SCALE + blockLoop;
                sioQueList[blockLoop].Init(&sync, magic, shareAddrs[sioPeerRankId] + IPC_DATA_OFFSET +
                    dmaSizePerCore * localBlockIdx, ipcBlockNum * IPC_QUE_DEPTH, ipcBlockNum);
                sioGatherSrc1QueList[blockLoop].Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET +
                    dmaSizePerCore * localBlockIdx, ipcBlockNum * IPC_QUE_DEPTH, ipcBlockNum);
                sioGatherSrc2QueList[blockLoop].Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET +
                    IPC_QUE_DEPTH * ipcBlockSize + dmaSizePerCore * localBlockIdx,
                    ipcBlockNum * RING_GATHER_QUE_DEPTH, ipcBlockNum);
                sioGatherDstQueList[blockLoop].Init(&sync, magic, shareAddrs[sioPeerRankId] + IPC_DATA_OFFSET +
                    (IPC_QUE_DEPTH + RING_GATHER_QUE_DEPTH) * ipcBlockSize + dmaSizePerCore * localBlockIdx,
                    ipcBlockNum * SIO_GATHER_QUE_DEPTH, ipcBlockNum);
            }
        } else if (blockIdx < INPUT_CORE_NUM + SIO_CORE_NUM + RING_CORE_NUM) {
            localBlockIdx = (blockIdx - (INPUT_CORE_NUM + SIO_CORE_NUM));
            ringSrcQue.Init(&sync, magic, shareAddrs[ringPrevRankId] + IPC_DATA_OFFSET +
                    dmaSizePerCore * localBlockIdx, ipcBlockNum * IPC_QUE_DEPTH, ipcBlockNum);
            ringDstQue.Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET +
                    dmaSizePerCore * localBlockIdx, ipcBlockNum * IPC_QUE_DEPTH, ipcBlockNum);
            ringGatherSrcQue.Init(&sync, magic, shareAddrs[ringPrevRankId] + IPC_DATA_OFFSET +
                    IPC_QUE_DEPTH * ipcBlockSize + dmaSizePerCore * localBlockIdx,
                ipcBlockNum * RING_GATHER_QUE_DEPTH, ipcBlockNum);
            ringGatherDstQue.Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET +
                IPC_QUE_DEPTH * ipcBlockSize + dmaSizePerCore * localBlockIdx,
                ipcBlockNum * RING_GATHER_QUE_DEPTH, ipcBlockNum);
        } else {
            for (int32_t blockLoop = 0; blockLoop < OUTPUT_CORE_SCALE; ++blockLoop) {
                localBlockIdx = (blockIdx - (INPUT_CORE_NUM + SIO_CORE_NUM + RING_CORE_NUM)) * OUTPUT_CORE_SCALE +
                    blockLoop;
                outputSrc1QueList[blockLoop].Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET +
                    (IPC_QUE_DEPTH + RING_GATHER_QUE_DEPTH) * ipcBlockSize + dmaSizePerCore * localBlockIdx,
                    ipcBlockNum * SIO_GATHER_QUE_DEPTH, ipcBlockNum);
                outputSrc2QueList[blockLoop].Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET +
                    IPC_QUE_DEPTH * ipcBlockSize + dmaSizePerCore * localBlockIdx,
                    ipcBlockNum * RING_GATHER_QUE_DEPTH, ipcBlockNum);
                outputSrc3QueList[blockLoop].Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET +
                    dmaSizePerCore * localBlockIdx, ipcBlockNum * IPC_QUE_DEPTH, ipcBlockNum);
            }
        }
    }

    FORCE_INLINE_AICORE void Input2Ipc()
    {
        for (int32_t blockLoop = 0; blockLoop < INPUT_CORE_SCALE; ++blockLoop) {
            localBlockIdx = blockIdx * INPUT_CORE_SCALE + blockLoop;
            inputQue = &(inputQueList[blockLoop]);
            Input2IpcByCore();
        }
    }

    FORCE_INLINE_AICORE void Input2IpcByCore()
    {
        const int32_t targetSioLayerId = (sioLayerId + (ringRankSize - 1 - sioLayerLoop)) % ringRankSize;
        const int32_t targetRankOffset = targetSioLayerId * RING_LAYER_NUM + ringLayerId;

        (*inputQue).DeQue(rank, RING_REDUCE_PEER_FLAG + localBlockIdx);
        const int32_t consumedQueIdx = ipcQueIdx - (IPC_QUE_DEPTH + ringRankSize - 1);
        if (consumedQueIdx >= 0 && consumedQueIdx % ringRankSize == 0) {
            sync.WaitSyncFlag(magic, consumedQueIdx, OUTPUT_FLAG + localBlockIdx, rank);
            sync.WaitSyncFlag(magic, consumedQueIdx, RING_GATHER_PEER_FLAG + localBlockIdx, rank);
        }

        BuildCoreDataNum(curLoopCnt, targetRankOffset);
        srcIpcTensor = inputTensor[targetRankOffset * totalBlockDataNum + curLoopCnt * dmaPerLoop +
            localBlockIdx * coreDataNum];
        dstIpcTensor = (*inputQue).EnQue();
        CpGM2GMPingPong(curCoreDataNum * sizeof(T), srcIpcTensor, dstIpcTensor, COPYONLY);
        sync.SetSyncFlag(magic, ipcQueIdx, INPUT_FLAG + localBlockIdx, sioPeerRankId);
    }

    FORCE_INLINE_AICORE void SioReduce()
    {
        for (int32_t blockLoop = 0; blockLoop < SIO_CORE_SCALE; ++blockLoop) {
            if (sioLayerLoop < ringRankSize - 1) {
                sioGatherSrc1QueList[blockLoop].ReadFront();
            }
        }
        if (curLoopCnt > 0 && sioLayerLoop == 0) {
            return;
        }
        const int32_t endIdx = (curLoopCnt < loopCount - 1) && (sioLayerLoop == ringRankSize - 1) ? 1 : 0;
        for (int32_t i = 0; i <= endIdx; ++i) {
            for (int32_t blockLoop = 0; blockLoop < SIO_CORE_SCALE; ++blockLoop) {
                localBlockIdx = (blockIdx - INPUT_CORE_NUM) * SIO_CORE_SCALE + blockLoop;
                sioQue = &(sioQueList[blockLoop]);
                SioReduceByCore(curLoopCnt + i, (sioLayerLoop + i) % ringRankSize, ipcQueIdx + i);
            }
        }
    }

    FORCE_INLINE_AICORE void SioReduceByCore(int32_t newLoopCnt, int32_t newLayerLoop, int32_t newQueIdx)
    {
        const int32_t targetSioLayerId = (sioLayerId + (ringRankSize - 1 - newLayerLoop)) % ringRankSize;
        const int32_t targetRankOffset = targetSioLayerId * RING_LAYER_NUM + (ringLayerId + 1) % RING_LAYER_NUM;

        sync.WaitSyncFlag(magic, newQueIdx, INPUT_FLAG + localBlockIdx, rank);
        BuildCoreDataNum(newLoopCnt, targetRankOffset);
        srcIpcTensor = inputTensor[targetRankOffset * totalBlockDataNum + newLoopCnt * dmaPerLoop +
            localBlockIdx * coreDataNum];
        dstIpcTensor = (*sioQue).EnQue();
        CpGM2GMPingPong(curCoreDataNum * sizeof(T), srcIpcTensor, dstIpcTensor, atomOp);
        sync.SetSyncFlag(magic, newQueIdx, SIO_REDUCE_FLAG + localBlockIdx, sioPeerRankId);
    }

    FORCE_INLINE_AICORE void BuildCoreDataNum(int32_t processLoopIdx, int32_t targetRankOffset)
    {
        const int64_t damCurLoop = (processLoopIdx == loopCount - 1) ?
            (targetRankOffset == rankSize - 1 ? dmaLastRankLoop : dmaLastLoop) : dmaPerLoop;
        coreDataNum = ipcBlockNum / RING_CORE_NUM;
        const int32_t maxIdx = damCurLoop / coreDataNum;
        const int32_t lastIdx = maxIdx >= RING_CORE_NUM ? (RING_CORE_NUM - 1) : maxIdx;

        lastCoreDataNum = damCurLoop - lastIdx * coreDataNum;
        curCoreDataNum = localBlockIdx < lastIdx ? coreDataNum : (localBlockIdx == lastIdx ? lastCoreDataNum : 0);
    }

    FORCE_INLINE_AICORE void SioGather()
    {
        for (int32_t blockLoop = 0; blockLoop < SIO_CORE_SCALE; ++blockLoop) {
            localBlockIdx = (blockIdx - INPUT_CORE_NUM) * SIO_CORE_SCALE + blockLoop;
            sioGatherSrc1Que = &(sioGatherSrc1QueList[blockLoop]);
            sioGatherSrc2Que = &(sioGatherSrc2QueList[blockLoop]);
            sioGatherDstQue = &(sioGatherDstQueList[blockLoop]);
            SioGatherByCore();
        }
    }

    FORCE_INLINE_AICORE void SioGatherByCore()
    {
        const int32_t targetSioLayerId = (sioLayerId + (ringRankSize - sioLayerLoop)) % ringRankSize;
        const int32_t targetRankOffset = targetSioLayerId * RING_LAYER_NUM + ringLayerId;

        sync.WaitSyncFlag(magic, gatherQueIdx, RING_GATHER_FLAG + localBlockIdx, rank);
        if (gatherQueIdx >= SIO_GATHER_QUE_DEPTH) {
            sync.WaitSyncFlag(magic, gatherQueIdx - SIO_GATHER_QUE_DEPTH, SIO_GATHER_OUTPUT_FLAG + localBlockIdx, rank);
        }
        BuildCoreDataNum(curLoopCnt, targetRankOffset);
        srcIpcTensor = (sioLayerLoop == 0 ? (*sioGatherSrc1Que).ReadFront() : (*sioGatherSrc2Que).ReadFront());
        dstIpcTensor = (*sioGatherDstQue).ReadFront();
        CpGM2GMPingPong(curCoreDataNum * sizeof(T), srcIpcTensor, dstIpcTensor, COPYONLY);
        sync.SetSyncFlag(magic, gatherQueIdx, SIO_GATHER_PEER_FLAG + localBlockIdx, sioPeerRankId);
        sync.SetSyncFlag(magic, gatherQueIdx, SIO_GATHER_FLAG + localBlockIdx, rank);
    }

    FORCE_INLINE_AICORE void RingReduce()
    {
        if (sioLayerLoop == 0) {
            ringDstQue.ReadFront();
            return;
        }

        const int32_t consumedQueIdx = ipcQueIdx - 1;
        sync.WaitSyncFlag(magic, consumedQueIdx + 1, SIO_REDUCE_FLAG + localBlockIdx, rank);
        if (sioLayerLoop == 1) {
            sync.WaitSyncFlag(magic, consumedQueIdx, SIO_REDUCE_FLAG + localBlockIdx, ringPrevRankId);
        } else {
            sync.WaitSyncFlag(magic, consumedQueIdx - 1, RING_REDUCE_FLAG + localBlockIdx,
                ringPrevRankId);
        }
        const int32_t targetSioLayerId = (sioLayerId + (ringRankSize - 1 -sioLayerLoop)) % ringRankSize;
        const int32_t targetRankOffset = targetSioLayerId * RING_LAYER_NUM + ringLayerId;
        BuildCoreDataNum(curLoopCnt, targetRankOffset);
        srcIpcTensor = ringSrcQue.ReadFront();
        dstIpcTensor = ringDstQue.ReadFront();
        CpGM2GMPingPong(curCoreDataNum * sizeof(T), srcIpcTensor, dstIpcTensor, atomOp);
        sync.SetSyncFlag(magic, consumedQueIdx, RING_REDUCE_FLAG + localBlockIdx, rank);
        sync.SetSyncFlag(magic, consumedQueIdx, RING_REDUCE_PEER_FLAG + localBlockIdx, ringPrevRankId);
    }

    FORCE_INLINE_AICORE void RingGather()
    {
        if (sioLayerLoop == 0) {
            sync.SetSyncFlag(magic, gatherQueIdx, RING_GATHER_FLAG + localBlockIdx, rank);
            sync.SetSyncFlag(magic, gatherQueIdx, RING_GATHER_PEER_FLAG + localBlockIdx, ringPrevRankId);
            return;
        }

        const int32_t targetSioLayerId = (sioLayerId + (ringRankSize - sioLayerLoop)) % ringRankSize;
        const int32_t targetRankOffset = targetSioLayerId * RING_LAYER_NUM + ringLayerId;
        sync.WaitSyncFlag(magic, gatherQueIdx - 1, RING_GATHER_FLAG + localBlockIdx, ringPrevRankId);
        if (gatherQueIdx > RING_GATHER_QUE_DEPTH) {
            sync.WaitSyncFlag(magic, gatherQueIdx - RING_GATHER_QUE_DEPTH, OUTPUT_FLAG + localBlockIdx, rank);
            if (targetRankOffset != ringPrevRankId) {
                sync.WaitSyncFlag(magic, gatherQueIdx - RING_GATHER_QUE_DEPTH, RING_GATHER_PEER_FLAG + localBlockIdx,
                    rank);
            }
        }

        BuildCoreDataNum(curLoopCnt, targetRankOffset);
        if (sioLayerLoop == 1) {
            srcIpcTensor = ringSrcQue.ReadFront();
        } else {
            srcIpcTensor = ringGatherSrcQue.ReadFront();
        }
        dstIpcTensor = ringGatherDstQue.ReadFront();
        CpGM2GMPingPong(curCoreDataNum * sizeof(T), srcIpcTensor, dstIpcTensor, COPYONLY);
        sync.SetSyncFlag(magic, gatherQueIdx, RING_GATHER_FLAG + localBlockIdx, rank);
        if (gatherQueIdx > 0) {
            sync.SetSyncFlag(magic, gatherQueIdx - 1, RING_GATHER_PEER_FLAG + localBlockIdx, ringPrevRankId);
        }
        if (sioLayerLoop == ringRankSize - 1) {
            ringGatherSrcQue.ReadFront();
        }
    }

    FORCE_INLINE_AICORE void PrepareOutput()
    {
        for (int32_t blockLoop = 0; blockLoop < OUTPUT_CORE_SCALE; ++blockLoop) {
            localBlockIdx = (blockIdx - (INPUT_CORE_NUM + SIO_CORE_NUM + RING_CORE_NUM)) * OUTPUT_CORE_SCALE +
                    blockLoop;
            if (sioLayerLoop < ringRankSize - 1) {
                outputSrc3QueList[blockLoop].ReadFront();
            }
        }
    }

    FORCE_INLINE_AICORE void Ipc2Output()
    {
        for (int32_t blockLoop = 0; blockLoop < OUTPUT_CORE_SCALE; ++blockLoop) {
            localBlockIdx = (blockIdx - (INPUT_CORE_NUM + SIO_CORE_NUM + RING_CORE_NUM)) * OUTPUT_CORE_SCALE +
                    blockLoop;
            outputSrc1Que = &(outputSrc1QueList[blockLoop]);
            outputSrc2Que = &(outputSrc2QueList[blockLoop]);
            outputSrc3Que = &(outputSrc3QueList[blockLoop]);
            Ipc2OutputByCore();
        }
    }

    FORCE_INLINE_AICORE void Ipc2OutputByCore()
    {
        const int32_t targetSioLayerId = (sioLayerId + (ringRankSize - sioLayerLoop)) % ringRankSize;
        const int32_t targetSioRankOffset = targetSioLayerId * RING_LAYER_NUM + (ringLayerId + 1) % RING_LAYER_NUM;
        const int32_t targetRingRankOffset = targetSioLayerId * RING_LAYER_NUM + ringLayerId;
        BuildCoreDataNum(curLoopCnt, targetSioRankOffset);
        sync.WaitSyncFlag(magic, gatherQueIdx, SIO_GATHER_PEER_FLAG + localBlockIdx, rank);
        srcIpcTensor = (*outputSrc1Que).ReadFront();
        dstIpcTensor = outputTensor[targetSioRankOffset * totalBlockDataNum + curLoopCnt * dmaPerLoop +
            localBlockIdx * coreDataNum];
        CpGM2GMPingPong(curCoreDataNum * sizeof(T), srcIpcTensor, dstIpcTensor, COPYONLY);
        sync.SetSyncFlag(magic, gatherQueIdx, SIO_GATHER_OUTPUT_FLAG + localBlockIdx, sioPeerRankId);
        BuildCoreDataNum(curLoopCnt, targetRingRankOffset);
        sync.WaitSyncFlag(magic, gatherQueIdx, SIO_GATHER_FLAG + localBlockIdx, rank);
        srcIpcTensor = sioLayerLoop == 0 ? (*outputSrc3Que).ReadFront() : (*outputSrc2Que).ReadFront();
        dstIpcTensor = outputTensor[targetRingRankOffset * totalBlockDataNum +
            curLoopCnt * dmaPerLoop + localBlockIdx * coreDataNum];
        CpGM2GMPingPong(curCoreDataNum * sizeof(T), srcIpcTensor, dstIpcTensor, COPYONLY);
        sync.SetSyncFlag(magic, gatherQueIdx, OUTPUT_FLAG + localBlockIdx, rank);
    }
};
#endif // LCCL_ALLREDUCE_HIERARCHY_DOUBLE_RING_H