/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LCCL_ALLGATHER_HIERARCHY_DOUBLE_RING_H
#define LCCL_ALLGATHER_HIERARCHY_DOUBLE_RING_H

#include "collectives.h"
#include "ipc_queue.h"
using namespace AscendC;

constexpr int STAGE_NUM = 4;
constexpr int QUE_DEPTH = 8;
constexpr int QUE_NUM_LOCAL = 2;
constexpr int RING_NUM = 2;
constexpr int STAGE_EVENT = 0;
constexpr int RING_EVENT = 1;

enum STAGE {
    HCCS_RING = 0,
    HCCS_TO_OUT,
    HCCS_TO_SIO,
    SIO_TO_OUT
};

template <typename T>
class AllGatherHierarchyDoubleRing : public Collectives {
public:
    FORCE_INLINE_AICORE AllGatherHierarchyDoubleRing(int rank, int rankSize, uint32_t extraFlag)
        : Collectives(rank, rankSize, extraFlag) {}

    FORCE_INLINE_AICORE void Init(KERNELS_ARGS_FUN())
    {
        Collectives::Init(KERNELS_ARGS_CALL());
        DumpLcclLogInfo(LogId::INIT, Op::COPYONLY);
        int64_t dataTotalSize = len * sizeof(T);
        const int coreNumPerStep = blockNum / STAGE_NUM;
        stage = blockIdx / coreNumPerStep;
        int stageCoreIdx = blockIdx % coreNumPerStep;
        dataSizePerCore = dataTotalSize / coreNumPerStep;
        const int64_t inputOffset = stageCoreIdx * dataSizePerCore;
        if (stageCoreIdx == coreNumPerStep - 1) {
            dataSizePerCore = dataTotalSize - (coreNumPerStep - 1) * dataSizePerCore;
        }

        inputGm.SetGlobalBuffer(input + inputOffset, dataSizePerCore);
        if (stage == STAGE::HCCS_TO_OUT) {
            for (int i = rank % RING_NUM; i < rankSize; i += RING_NUM) {
                outputGm[i / RING_NUM].SetGlobalBuffer(output + dataTotalSize * i + inputOffset, dataSizePerCore);
            }
        } else if (stage == STAGE::SIO_TO_OUT) {
            for (int i = (rank + 1) % RING_NUM; i < rankSize; i += RING_NUM) {
                outputGm[i / RING_NUM].SetGlobalBuffer(output + dataTotalSize * i + inputOffset, dataSizePerCore);
            }
        }

        int64_t queTotalSize = IPC_BUFF_MAX_SIZE / coreNumPerStep;
        int64_t queSize = queTotalSize / QUE_NUM_LOCAL;
        int64_t queHccsOffset = stageCoreIdx * queTotalSize;
        blockSize = queSize / QUE_DEPTH;

        queHccsLocal.Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET + queHccsOffset, queSize, blockSize);
        queSioLocal.Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET + queHccsOffset + queSize,
            queSize, blockSize);
        rankRingForward = (rank + RING_NUM) % rankSize;
        queHccsForward.Init(&sync, magic, shareAddrs[rankRingForward] + IPC_DATA_OFFSET + queHccsOffset,
            queSize, blockSize);
        rankSioAdjoint = rank ^ 1;
        queSioAdjoint.Init(&sync, magic,
            shareAddrs[rankSioAdjoint] + IPC_DATA_OFFSET + queHccsOffset + queSize, queSize, blockSize);

        for (int i = 0; i < STAGE_NUM; ++i) {
            stageEvents[i] = sync.CalEventIdByMulBlockNum(STAGE_EVENT, stageCoreIdx + coreNumPerStep * i);
        }

        DumpLcclLogInfo(LogId::INIT, Op::COPYONLY);
    }

    FORCE_INLINE_AICORE void Process()
    {
        DumpLcclLogInfo(LogId::PROCESS, Op::COPYONLY);
        int count = rankSize / RING_NUM * CeilDiv<int64_t, int64_t>(dataSizePerCore, blockSize);
        if (stage == STAGE::HCCS_RING) {
            ProcessHccsRing(count);
        } else if (stage == STAGE::HCCS_TO_OUT) {
            ProcessHccsToOut(count);
        } else if (stage == STAGE::HCCS_TO_SIO) {
            ProcessHccsToSio(count);
        } else if (stage == STAGE::SIO_TO_OUT) {
            ProcessSioToOut(count);
        }
        DumpLcclLogInfo(LogId::PROCESS, Op::COPYONLY);
    }
private:
    FORCE_INLINE_AICORE void ProcessHccsRing(const int count)
    {
        constexpr int dependencyNum = 3;
        int deQueWaitRanks[dependencyNum] = {(rank + rankSize - RING_NUM) % rankSize, rank, rank};
        int deQueWaitEvents[dependencyNum] = {
            sync.CalEventIdByMulBlockNum(RING_EVENT, blockIdx),
            stageEvents[static_cast<int>(STAGE::HCCS_TO_OUT)],
            stageEvents[static_cast<int>(STAGE::HCCS_TO_SIO)]};
        int64_t remainSize = dataSizePerCore;
        int64_t dataSize = 0;
        GlobalTensor<uint8_t> input;
        GlobalTensor<uint8_t> output;
        int64_t waitFlag = 0;
        int i = 0;
        while (i < count) {
            int countRankId = (rank + i * RING_NUM) % rankSize;
            if (countRankId == rank) {
                dataSize = (remainSize >= blockSize) ? blockSize : remainSize;
                input = inputGm[dataSizePerCore - remainSize];
                remainSize -= blockSize;
            } else {
                if (i == 1) {
                    sync.WaitSyncFlag(magic, 0, stageEvents[static_cast<int>(STAGE::HCCS_RING)], rankRingForward);
                    waitFlag = sync.GetInnerFlag(rankRingForward,
                        stageEvents[static_cast<int>(STAGE::HCCS_RING)]) & EVENT_ID_MASK;
                }
                if (waitFlag < i - 1) {
                    waitFlag = sync.GetInnerFlag(rankRingForward,
                        stageEvents[static_cast<int>(STAGE::HCCS_RING)]) & EVENT_ID_MASK;
                    continue;
                }
                input = queHccsForward.ReadFront();
            }
            queHccsLocal.DeQue(deQueWaitRanks, deQueWaitEvents, dependencyNum);
            output = queHccsLocal.EnQue();
            CpGM2GMPingPong(dataSize, input, output, -1);

            sync.SetSyncFlag(magic, i, stageEvents[static_cast<int>(STAGE::HCCS_RING)], rank);
            if (countRankId != rank) {
                if ((rank + (i + 1) * RING_NUM) % rankSize == rank) {
                    queHccsForward.ReadFront();
                    sync.SetSyncFlag(magic, i, sync.CalEventIdByMulBlockNum(RING_EVENT, blockIdx), rank);
                } else {
                    sync.SetSyncFlag(magic, i - 1, sync.CalEventIdByMulBlockNum(RING_EVENT, blockIdx), rank);
                }
            }
            ++i;
        }
    }

    FORCE_INLINE_AICORE void ProcessHccsToOut(const int count)
    {
        GlobalTensor<uint8_t> input;
        GlobalTensor<uint8_t> output;
        int64_t remainSize = dataSizePerCore;
        int64_t dataSize = 0;
        sync.WaitSyncFlag(magic, 0, stageEvents[static_cast<int>(STAGE::HCCS_RING)], rank);
        int64_t waitFlag = sync.GetInnerFlag(rank, stageEvents[static_cast<int>(STAGE::HCCS_RING)]) & EVENT_ID_MASK;
        int i = 0;
        while (i < count) {
            if (waitFlag < i) {
                waitFlag = sync.GetInnerFlag(rank, stageEvents[static_cast<int>(STAGE::HCCS_RING)]) & EVENT_ID_MASK;
                continue;
            }
            int countRankId = (rank + i * RING_NUM) % rankSize;
            if (countRankId == rank) {
                dataSize = (remainSize >= blockSize) ? blockSize : remainSize;
            }
            input = queHccsLocal.ReadFront();
            output = outputGm[countRankId / RING_NUM][dataSizePerCore - remainSize];
            CpGM2GMPingPong(dataSize, input, output, -1);
            constexpr int32_t halfQueDepth = 2;
            if (i % (QUE_DEPTH / halfQueDepth) == 0) {
                sync.SetSyncFlag(magic, i, stageEvents[static_cast<int>(STAGE::HCCS_TO_OUT)], rank);
            }
            if ((countRankId + RING_NUM) % rankSize == rank) {
                remainSize -= blockSize;
            }
            ++i;
        }
    }
    FORCE_INLINE_AICORE void ProcessHccsToSio(const int count)
    {
        GlobalTensor<uint8_t> input;
        GlobalTensor<uint8_t> output;
        int64_t remainSize = dataSizePerCore;
        int64_t dataSize = 0;
        sync.WaitSyncFlag(magic, 0, stageEvents[static_cast<int>(STAGE::HCCS_RING)], rank);
        int64_t waitFlag = sync.GetInnerFlag(rank, stageEvents[static_cast<int>(STAGE::HCCS_RING)]) & EVENT_ID_MASK;
        int i = 0;
        while (i < count) {
            if (waitFlag < i) {
                waitFlag = sync.GetInnerFlag(rank, stageEvents[static_cast<int>(STAGE::HCCS_RING)]) & EVENT_ID_MASK;
                continue;
            }
            int countRankId = (rank + i * RING_NUM) % rankSize;
            if (countRankId == rank) {
                dataSize = (remainSize >= blockSize) ? blockSize : remainSize;
                remainSize -= blockSize;
            }
            input = queHccsLocal.ReadFront();
            queSioAdjoint.DeQue(rankSioAdjoint, stageEvents[static_cast<int>(STAGE::SIO_TO_OUT)]);
            output = queSioAdjoint.EnQue();
            CpGM2GMPingPong(dataSize, input, output, -1);
            sync.SetSyncFlag(magic, i, stageEvents[static_cast<int>(STAGE::HCCS_TO_SIO)], rank);
            ++i;
        }
    }
    FORCE_INLINE_AICORE void ProcessSioToOut(const int count)
    {
        GlobalTensor<uint8_t> input;
        GlobalTensor<uint8_t> output;
        int64_t remainSize = dataSizePerCore;
        int64_t dataSize = 0;
        sync.WaitSyncFlag(magic, 0, stageEvents[static_cast<int>(STAGE::HCCS_TO_SIO)], rankSioAdjoint);
        int64_t waitFlag = sync.GetInnerFlag(rankSioAdjoint,
            stageEvents[static_cast<int>(STAGE::HCCS_TO_SIO)]) & EVENT_ID_MASK;
        int i = 0;
        while (i < count) {
            if (waitFlag < i) {
                waitFlag = sync.GetInnerFlag(rankSioAdjoint,
                    stageEvents[static_cast<int>(STAGE::HCCS_TO_SIO)]) & EVENT_ID_MASK;
                continue;
            }
            int countRankId = (rankSioAdjoint + i * RING_NUM) % rankSize;
            if (countRankId == rankSioAdjoint) {
                dataSize = (remainSize >= blockSize) ? blockSize : remainSize;
            }
            input = queSioLocal.ReadFront();
            output = outputGm[countRankId / RING_NUM][dataSizePerCore - remainSize];
            CpGM2GMPingPong(dataSize, input, output, -1);
            constexpr int32_t halfQueDepth = 2;
            if (i % (QUE_DEPTH / halfQueDepth) == 0) {
                sync.SetSyncFlag(magic, i, stageEvents[static_cast<int>(STAGE::SIO_TO_OUT)], rank);
            }
            if ((countRankId + RING_NUM) % rankSize == rankSioAdjoint) {
                remainSize -= blockSize;
            }
            ++i;
        }
    }
private:
    int stageEvents[STAGE_NUM];
    GlobalTensor<uint8_t> inputGm;
    GlobalTensor<uint8_t> outputGm[LCAL_MAX_RANK_SIZE / RING_NUM];
    IpcQueue<uint8_t> queHccsLocal;
    IpcQueue<uint8_t> queHccsForward;
    IpcQueue<uint8_t> queSioLocal;
    IpcQueue<uint8_t> queSioAdjoint;
    int64_t dataSizePerCore;
    int stage;
    int rankRingForward;
    int rankSioAdjoint;
    int64_t blockSize;
};

#endif // LCCL_ALLGATHER_HIERARCHY_DOUBLE_RING_H