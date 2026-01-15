/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LCCL_REDUCE_SCATTER_HIERARCHY_DOUBLE_RING_H
#define LCCL_REDUCE_SCATTER_HIERARCHY_DOUBLE_RING_H

#include "sync_collectives.h"
#include "collectives.h"
#include "ipc_queue.h"
using namespace AscendC;

template<typename T>
class ReduceScatterHierarchyDoubleRing : protected Collectives {
    constexpr static int32_t RING_LAYER_NUM = 2;
    constexpr static int32_t INPUT_CORE_NUM = 12;
    constexpr static int32_t SIO_CORE_NUM = 12;
    constexpr static int32_t RING_CORE_NUM = 12;
    constexpr static int32_t IPC_QUE_DEPTH = 32;
    constexpr static int32_t INPUT_SIO_PEER_FLAG = 0 * RING_CORE_NUM;
    constexpr static int32_t SIO_REDUCE_FLAG = 1 * RING_CORE_NUM;
    constexpr static int32_t RING_REDUCE_FLAG = 2 * RING_CORE_NUM;
    constexpr static int32_t RING_REDUCE_PEER_FLAG = 3 * RING_CORE_NUM;
    constexpr static int32_t OUTPUT_FLAG = 4 * RING_CORE_NUM;
    constexpr static int32_t INPUT_FLAG = 5 * RING_CORE_NUM;

    constexpr static int32_t INPUT_CORE_SCALE = RING_CORE_NUM / INPUT_CORE_NUM;
    constexpr static int32_t SIO_CORE_SCALE = RING_CORE_NUM / SIO_CORE_NUM;
    constexpr static int64_t BLOCK_NUM_ALIGN = BLOCK_SIZE / sizeof(T);
    constexpr static int32_t BREAK_CYCLE = 10;

public:
    FORCE_INLINE_AICORE ReduceScatterHierarchyDoubleRing(int rank, int rankSize, uint32_t extraFlag)
        : Collectives(rank, rankSize, extraFlag) {}
    FORCE_INLINE_AICORE void Init(KERNELS_ARGS_FUN())
    {
        Collectives::Init(KERNELS_ARGS_CALL());
        atomOp = op;
        DumpLcclLogInfo(LogId::INIT, static_cast<Op>(atomOp));
        blockNum = INPUT_CORE_NUM + SIO_CORE_NUM + RING_CORE_NUM;
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
        ipcBlockNum = IPC_BUFF_MAX_SIZE / IPC_QUE_DEPTH / sizeof(T);
        totalBlockDataNum = len;
        loopCount = CeilDiv(totalBlockDataNum, ipcBlockNum);
        dmaPerLoop = ipcBlockNum;
        dmaLastLoop = totalBlockDataNum - (loopCount - 1) * ipcBlockNum;
        const int64_t dmaSizePerCore = ipcBlockNum / RING_CORE_NUM * sizeof(T);
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
            }
        } else {
            localBlockIdx = (blockIdx - (INPUT_CORE_NUM + SIO_CORE_NUM));
            ringSrcQue.Init(&sync, magic, shareAddrs[ringPrevRankId] + IPC_DATA_OFFSET +
                    dmaSizePerCore * localBlockIdx, ipcBlockNum * IPC_QUE_DEPTH, ipcBlockNum);
            ringDstQue.Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET +
                    dmaSizePerCore * localBlockIdx, ipcBlockNum * IPC_QUE_DEPTH, ipcBlockNum);
        }
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
            const int64_t damCurLoop = (curLoopCnt == loopCount - 1) ? dmaLastLoop : dmaPerLoop;
            coreDataNum = damCurLoop / RING_CORE_NUM;
            lastCoreDataNum = damCurLoop - (RING_CORE_NUM - 1) * coreDataNum;
            for (sioLayerLoop = 0; sioLayerLoop < ringRankSize; ++sioLayerLoop) {
                if (blockIdx < INPUT_CORE_NUM) {
                    Input2Ipc();
                } else if (blockIdx < INPUT_CORE_NUM + SIO_CORE_NUM) {
                    SioReduce();
                } else {
                    RingReduceOutput();
                }
                ++ipcQueIdx;
            }
        }
        DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
    }

private:
    IpcQueue<T> inputQueList[INPUT_CORE_SCALE];
    IpcQueue<T> sioQueList[SIO_CORE_SCALE];
    IpcQueue<T> ringSrcQue;
    IpcQueue<T> ringDstQue;
    IpcQueue<T> *inputQue = nullptr;
    IpcQueue<T> *sioQue = nullptr;
    GlobalTensor<T> inputTensor;
    GlobalTensor<T> outputTensor;
    GlobalTensor<T> srcTensor;
    GlobalTensor<T> dstTensor;
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
    int32_t ipcQueIdx = 0;
    int32_t loopCount = 0;
    int32_t curLoopCnt = 0;
    int32_t sioLayerLoop = 0;
    int64_t coreDataNum = 0;
    int64_t lastCoreDataNum = 0;
    int64_t curCoreDataNum = 0;

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
        curCoreDataNum = (localBlockIdx == RING_CORE_NUM - 1) ? lastCoreDataNum : coreDataNum;
        srcTensor = inputTensor[targetRankOffset * totalBlockDataNum + curLoopCnt * ipcBlockNum +
            localBlockIdx * coreDataNum];
        dstTensor = (*inputQue).EnQue();
        CpGM2GMPingPong(curCoreDataNum * sizeof(T), srcTensor, dstTensor, COPYONLY);
        sync.SetSyncFlag(magic, ipcQueIdx, INPUT_FLAG + localBlockIdx, rank);
    }

    FORCE_INLINE_AICORE void SioReduce()
    {
        for (int32_t blockLoop = 0; blockLoop < SIO_CORE_SCALE; ++blockLoop) {
            localBlockIdx = (blockIdx - INPUT_CORE_NUM) * SIO_CORE_SCALE + blockLoop;
            sioQue = &(sioQueList[blockLoop]);
            SioReduceByCore();
        }
    }

    FORCE_INLINE_AICORE void SioReduceByCore()
    {
        const int32_t targetSioLayerId = (sioLayerId + (ringRankSize - 1 - sioLayerLoop)) % ringRankSize;
        const int32_t targetRankOffset = targetSioLayerId * RING_LAYER_NUM + (ringLayerId + 1) % RING_LAYER_NUM;

        curCoreDataNum = (localBlockIdx == RING_CORE_NUM - 1) ? lastCoreDataNum : coreDataNum;
        srcTensor = inputTensor[targetRankOffset * totalBlockDataNum + curLoopCnt * ipcBlockNum +
            localBlockIdx * coreDataNum];
        dstTensor = (*sioQue).EnQue();
        if (ipcQueIdx == 0) {
            sync.WaitSyncFlag(magic, ipcQueIdx, INPUT_FLAG + localBlockIdx, sioPeerRankId, BREAK_CYCLE);
        } else {
            sync.WaitSyncFlag(magic, ipcQueIdx, INPUT_FLAG + localBlockIdx, sioPeerRankId);
        }
        CpGM2GMPingPong(curCoreDataNum * sizeof(T), srcTensor, dstTensor, atomOp);
        sync.SetSyncFlag(magic, ipcQueIdx, SIO_REDUCE_FLAG + localBlockIdx, sioPeerRankId);
    }

    FORCE_INLINE_AICORE void RingReduceOutput()
    {
        if (sioLayerLoop == 0) {
            ringDstQue.ReadFront();
            return;
        }
        curCoreDataNum = (localBlockIdx == RING_CORE_NUM - 1) ? lastCoreDataNum : coreDataNum;
        srcTensor = ringSrcQue.ReadFront();
        dstTensor = ringDstQue.ReadFront();
        GlobalTensor<T> srcOutTensor;
        GlobalTensor<T> dstOutTensor;
        if (sioLayerLoop == ringRankSize - 1) {
            ringSrcQue.ReadFront();
            srcOutTensor = dstTensor;
            dstOutTensor = outputTensor[curLoopCnt * ipcBlockNum + localBlockIdx * coreDataNum];
        }
        const int32_t consumedQueIdx = ipcQueIdx - 1;
        if (consumedQueIdx == 0) {
            sync.WaitSyncFlag(magic, consumedQueIdx, SIO_REDUCE_FLAG + localBlockIdx, ringPrevRankId, BREAK_CYCLE);
        } else {
            sync.WaitSyncFlag(magic, consumedQueIdx, SIO_REDUCE_FLAG + localBlockIdx, ringPrevRankId);
        }
        if (sioLayerLoop > 1) {
            sync.WaitSyncFlag(magic, consumedQueIdx - 1, RING_REDUCE_FLAG + localBlockIdx, ringPrevRankId);
        }
        sync.WaitSyncFlag(magic, ipcQueIdx, INPUT_FLAG + localBlockIdx, rank);
        CpGM2GMPingPong(curCoreDataNum * sizeof(T), srcTensor, dstTensor, atomOp);
        if (sioLayerLoop != ringRankSize - 1) {
            sync.SetSyncFlag(magic, consumedQueIdx, RING_REDUCE_FLAG + localBlockIdx, rank);
        } else {
            sync.WaitSyncFlag(magic, ipcQueIdx, SIO_REDUCE_FLAG + localBlockIdx, rank);
            CpGM2GMPingPong(curCoreDataNum * sizeof(T), srcOutTensor, dstOutTensor, COPYONLY);
        }
    }
};

#endif // LCCL_REDUCE_SCATTER_HIERARCHY_DOUBLE_RING_H