/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LCCL_ALL2ALL_HIERARCHY_H
#define LCCL_ALL2ALL_HIERARCHY_H

#include "collectives.h"
#include "sync_collectives.h"
#include "ipc_queue.h"

using namespace AscendC;

template <typename T>
class All2AllHierarchy : protected Collectives {
    constexpr static int QUEUE_DEPTH = 2;
    constexpr static int32_t STEP_TIMES = 2;
    constexpr static int INVALID_RANK_NUM = 0xFFFFFFFF;
    constexpr static int INVALID_RANK = 0xFFFFFFFF;
    constexpr static const int64_t SIO = 2;
    constexpr static int64_t CORE_NUM_PER_STAGE = 16;
    constexpr static int64_t MULTI_RANK_SIZE = CORE_NUM_PER_STAGE;
    constexpr static int64_t PRODUCER_CORE = 1;
    constexpr static int64_t CONSUMER_CORE = 2;
    static const int64_t DIE_CHANGE = 1;

public:
    FORCE_INLINE_AICORE All2AllHierarchy(int rank, int rankSize, uint32_t extraFlag)
        : Collectives(rank, rankSize, extraFlag) {}
    FORCE_INLINE_AICORE void Init(KERNELS_ARGS_FUN())
    {
        Collectives::Init(KERNELS_ARGS_CALL());
        this->input = (__gm__ T *) input;
        this->output = (__gm__ T *) output;

        perRankDataNum = GetDataCount(len, rankSize);
        curRankDataNum = perRankDataNum;
        InitShare();
        InitCoreGroup();
        InitDataSlice();
    }
    FORCE_INLINE_AICORE void Process()
    {
        if (coreGroup == PRODUCER_CORE) {
            Producer();
        } else {
            Consumer();
        }
    }
private:
    FORCE_INLINE_AICORE void InitShare()
    {
        int64_t queNum = blockNum / STEP_TIMES;
        if (rankSize <= CORE_NUM_PER_STAGE) {
            queNum = rankSize;
        }
        if (len < perQueElemLen) {
            coreNumPerRank = 1;
        }
        perQueElemLen = IPC_BUFF_MAX_SIZE / queNum / QUEUE_DEPTH / sizeof(T);
        queLen = perQueElemLen * QUEUE_DEPTH;
        queSize = queLen * sizeof(T);
    }

    FORCE_INLINE_AICORE void InitCoreGroup()
    {
        coreNumPerRank = 1;
        if (len < perQueElemLen) {
            coreNumPerRank = 1;
        }
        coreNumPerStage = coreNumPerRank * rankSize < CORE_NUM_PER_STAGE ?
        coreNumPerRank * rankSize : CORE_NUM_PER_STAGE;
        rankNumPerCore = CeilDiv(rankSize, coreNumPerStage);
        flagNumPerStage = rankSize;
        groupCore = (rank / coreNumPerStage) * coreNumPerStage;
        if (blockIdx < coreNumPerStage) {
            coreGroup = PRODUCER_CORE;
            for (auto i = 0; i < rankNumPerCore; ++i) {
                groupCoreIdx[i] = (groupCore + i * coreNumPerStage) % rankSize + blockIdx;
            }
        } else if (blockIdx < coreNumPerStage + coreNumPerStage) {
            coreGroup = CONSUMER_CORE;
            for (auto i = 0; i < rankNumPerCore; ++i) {
                int64_t prefix = (groupCore - i * coreNumPerStage) >= 0 ?
                    (groupCore - i * coreNumPerStage) : groupCore + ((rankNumPerCore - i) * coreNumPerStage);
                groupCoreIdx[i] = prefix + blockIdx - coreNumPerStage;
            }
        }
    }

    FORCE_INLINE_AICORE void InitDataSlice()
    {
        ipcDataNumPreBlock = curRankDataNum;
        if (coreGroup == PRODUCER_CORE) {
            for (auto i = 0; i < rankNumPerCore; ++i) {
                if (groupCoreIdx[i] % SIO == rank % SIO) {
                    srcInnerQue[i].Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET +
                        (groupCoreIdx[i] % coreNumPerStage) * queSize, queLen, perQueElemLen);
                } else {
                    SrcSioQue[i].Init(&sync, magic, shareAddrs[sioRank] + IPC_DATA_OFFSET +
                        ((groupCoreIdx[i] + (rank - sioRank)) % coreNumPerStage) * queSize,
                        queLen, perQueElemLen);
                }
                sliceNum = CeilDiv(ipcDataNumPreBlock, perQueElemLen);
            }
        } else if (coreGroup == CONSUMER_CORE) {
            for (auto i = 0; i < rankNumPerCore; ++i) {
                computePullRank(groupCoreIdx[i], rank);
                if (rank % SIO == 0) {
                    pullOffset = DIE_CHANGE * groupCoreIdx[i] % SIO;
                } else {
                    pullOffset = groupCoreIdx[i] % SIO - DIE_CHANGE;
                }

                pullQue[i].Init(&sync, magic, shareAddrs[pullRank] + IPC_DATA_OFFSET +
                    (rank % coreNumPerStage) * queSize + pullOffset * queSize, queLen, perQueElemLen);
                sliceNum = CeilDiv(ipcDataNumPreBlock, perQueElemLen);
            }
        }
    }

    FORCE_INLINE_AICORE void Producer()
    {
        for (auto i = 0; i < rankNumPerCore; ++i) {
            for (auto sliceIdx = 0; sliceIdx < sliceNum; ++sliceIdx) {
                Input2IpcSlice(i, sliceIdx);
            }
        }
    }

    FORCE_INLINE_AICORE void Input2IpcSlice(int64_t idx, int64_t sliceIdx)
    {
        inputGt.SetGlobalBuffer((__gm__ T*)input + groupCoreIdx[idx] * ipcDataNumPreBlock, ipcDataNumPreBlock);
        copyLen = ipcDataNumPreBlock - perQueElemLen * sliceIdx;
        if (copyLen > perQueElemLen) {
            copyLen = perQueElemLen;
        } else if (copyLen < 0) {
            copyLen = 0;
        }
        if (groupCoreIdx[idx] % SIO == rank % SIO) {
            if (idx > 0) {
                sync.WaitSyncFlag(magic, sliceIdx + sliceNum * (idx - 1),
                    groupCoreIdx[idx - 1] + flagNumPerStage, rank);
            }
            srcInnerQue[idx].DeQue(rank, groupCoreIdx[idx] + flagNumPerStage);
            writeGt = srcInnerQue[idx].EnQue();
            if (copyLen > 0) {
                CpGM2GMPingPong<T>(copyLen * sizeof(T), inputGt[sliceIdx * perQueElemLen], writeGt, Op::COPYONLY);
                sync.SetSyncFlag(magic, sliceIdx + sliceNum * idx, groupCoreIdx[idx], rank);
            }
        } else {
            if (idx > 0) {
                sync.WaitSyncFlag(magic, sliceIdx + sliceNum * (idx - 1),
                    groupCoreIdx[idx - 1] + flagNumPerStage + (rank - sioRank), sioRank);
            }
            SrcSioQue[idx].DeQue(sioRank, groupCoreIdx[idx] + (rank - sioRank) + flagNumPerStage);
            writeGt = SrcSioQue[idx].EnQue();
            if (copyLen > 0) {
                CpGM2GMPingPong<T>(copyLen * sizeof(T), inputGt[sliceIdx * perQueElemLen], writeGt, Op::COPYONLY);
                sync.SetSyncFlag(magic, sliceIdx + sliceNum * idx, groupCoreIdx[idx] + (rank - sioRank), sioRank);
            }
        }
    }
    FORCE_INLINE_AICORE void Consumer()
    {
        for (auto i = 0; i < rankNumPerCore; ++i) {
            computePullRank(groupCoreIdx[i], rank);
            for (auto sliceIdx = 0; sliceIdx < sliceNum; ++sliceIdx) {
                Ipc2Output(i, sliceIdx);
            }
        }
    }

    FORCE_INLINE_AICORE void computePullRank(int64_t& target, int64_t rank)
    {
        if (rank % SIO == 0) {
            pullRank = (target / SIO) * SIO;
        } else {
            pullRank = (target / SIO) * SIO + DIE_CHANGE;
        }
    }

    FORCE_INLINE_AICORE void Ipc2Output(int64_t idx, int64_t sliceIdx)
    {
        outputGt.SetGlobalBuffer((__gm__ T*)output + groupCoreIdx[idx] * ipcDataNumPreBlock,
                                ipcDataNumPreBlock);
        cpOffset = rank % SIO == 0 ? rank + groupCoreIdx[idx] % SIO :
                    (rank - DIE_CHANGE) + groupCoreIdx[idx] % SIO;
        copyLen = ipcDataNumPreBlock - perQueElemLen * sliceIdx;
        if (copyLen > perQueElemLen) {
            copyLen = perQueElemLen;
        } else if (copyLen < 0) {
            copyLen = 0;
        }
        readGt = pullQue[idx].ReadFront();
        sync.WaitSyncFlag(magic, sliceIdx + sliceNum * idx, cpOffset, pullRank);
        if (copyLen > 0) {
            CpGM2GMPingPong<T>(copyLen * sizeof(T), readGt, outputGt[sliceIdx * perQueElemLen], Op::COPYONLY);
        }
        sync.SetSyncFlag(magic, sliceIdx + sliceNum * idx, cpOffset + flagNumPerStage, pullRank);
    }
    GlobalTensor<T> inputGt;
    GlobalTensor<T> outputGt;
    GlobalTensor<T> readGt;
    GlobalTensor<T> writeGt;
    __gm__ T *input;
    __gm__ T *output;

    int atomOp;
    IpcQueue<T> srcInnerQue[MULTI_RANK_SIZE];
    IpcQueue<T> SrcSioQue[MULTI_RANK_SIZE];
    IpcQueue<T> pullQue[MULTI_RANK_SIZE];
    int64_t perRankDataNum;
    int64_t curRankDataNum;
    int64_t ipcDataNumPreBlock;
    int64_t pullRank;
    int64_t pullOffset;
    int64_t sioRank = (rank % 2 == 0) ? rank + 1:rank - 1;
    int64_t cpOffset;
    int64_t perQueElemLen;
    int64_t queLen;
    int64_t queSize;
    int64_t coreNumPerStage;
    int64_t flagNumPerStage;
    int64_t coreNumPerRank;
    int64_t rankNumPerCore;
    int64_t coreGroup;
    int64_t groupCoreIdx[MULTI_RANK_SIZE];
    int64_t sliceNum;
    int64_t copyLen;
    int64_t groupCore;
};

#endif // LCCL_ALL2ALL_HIERARCHY_H
