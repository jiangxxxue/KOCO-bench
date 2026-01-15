/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LCCL_ALL2ALL_HIERARCHY_SMALL_H
#define LCCL_ALL2ALL_HIERARCHY_SMALL_H

#include "collectives.h"
#include "sync_collectives.h"
#include "ipc_queue.h"

using namespace AscendC;

template <typename T>
class All2AllHierarchySmall : protected Collectives {
    constexpr static int QUEUE_DEPTH = 2;
    constexpr static int32_t STEP_TIMES = 2;
    constexpr static int INVALID_RANK_NUM = 0xFFFFFFFF;
    constexpr static int INVALID_RANK = 0xFFFFFFFF;
    constexpr static int64_t CORE_NUM_PER_STAGE = 16;
    constexpr static int64_t PRODUCER_CORE = 1;
    constexpr static int64_t CONSUMER_CORE = 2;
    constexpr static int64_t SIO = 2;

public:
    FORCE_INLINE_AICORE All2AllHierarchySmall(int rank, int rankSize, uint32_t extraFlag)
        : Collectives(rank, rankSize, extraFlag) {}
    FORCE_INLINE_AICORE void Init(KERNELS_ARGS_FUN())
    {
        Collectives::Init(KERNELS_ARGS_CALL());
        this->input = (__gm__ T *) input;
        this->output = (__gm__ T *) output;

        curRankDataNum = GetDataCount(len, rankSize);
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
        coreNumPerStage = CORE_NUM_PER_STAGE;
        singleStage = coreNumPerStage / SIO;
        perQueElemLen = IPC_BUFF_MAX_SIZE / SIO / singleStage / QUEUE_DEPTH / sizeof(T);
        queLen = perQueElemLen * QUEUE_DEPTH;
        queSize = queLen * sizeof(T);
        queBlockSize = IPC_BUFF_MAX_SIZE / SIO;
    }

    FORCE_INLINE_AICORE void InitCoreGroup()
    {
        if (len < perQueElemLen) {
            coreNumPerRank = 1;
        }
        loopCount = rankSize / SIO;
        flagNumPerStage = coreNumPerStage;
        if (blockIdx < coreNumPerStage) {
            coreGroup = PRODUCER_CORE;
            groupCoreIdx = blockIdx;
        } else if (blockIdx < coreNumPerStage + coreNumPerStage) {
            coreGroup = CONSUMER_CORE;
            groupCoreIdx = blockIdx - coreNumPerStage;
        }
    }

    FORCE_INLINE_AICORE void InitDataSlice()
    {
        ipcDataNumPreBlock = GetDataCount(curRankDataNum, singleStage);
        int64_t ifOffSet = queBlockSize * (rank % SIO);
        if (coreGroup == PRODUCER_CORE) {
            for (auto i = 0; i < loopCount; ++i) {
                if (groupCoreIdx < singleStage) {
                    srcLocalQue1.Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET + ifOffSet +
                                     groupCoreIdx * queSize, queLen, perQueElemLen);
                } else {
                    srcSioQue1.Init(&sync, magic, shareAddrs[sioRank] + IPC_DATA_OFFSET + ifOffSet +
                                     (groupCoreIdx - singleStage) * queSize, queLen, perQueElemLen);
                }
                sliceNum = CeilDiv(ipcDataNumPreBlock, perQueElemLen);
            }
        }
        sliceNum = CeilDiv(ipcDataNumPreBlock, perQueElemLen);
    }

    FORCE_INLINE_AICORE void Producer()
    {
        for (auto i = 0; i < loopCount; ++i) {
            srcRank = (rank + i * SIO) % rankSize;
            sioSrcRank = (srcRank % SIO == 0) ? srcRank + 1 : srcRank - 1;
            srcLocalQue = srcLocalQue1;
            srcSioQue = srcSioQue1;
            for (auto sliceIdx = 0; sliceIdx < sliceNum; ++sliceIdx) {
                Input2IpcSlice(i, sliceIdx);
            }
        }
    }

    FORCE_INLINE_AICORE void Input2IpcSlice(int64_t idx, int64_t sliceIdx)
    {
        copyLen = ipcDataNumPreBlock - perQueElemLen * sliceIdx;
        if (copyLen > perQueElemLen) {
            copyLen = perQueElemLen;
        } else if (copyLen < 0) {
            copyLen = 0;
        }
        int64_t flagIdx = groupCoreIdx + (rank % SIO) * singleStage;
        if (groupCoreIdx < singleStage) {
            if (idx > 0) {
                int64_t waitRank = (srcRank - SIO) >= 0 ? (srcRank - SIO) : srcRank + ((loopCount - 1) * SIO);
                sync.WaitSyncFlag(magic, sliceIdx + sliceNum * (idx - 1), flagIdx + (waitRank / SIO) * coreNumPerStage
                                + flagNumPerStage, rank);
            }
            inputGt.SetGlobalBuffer((__gm__ T*)input + srcRank * curRankDataNum + groupCoreIdx * ipcDataNumPreBlock,
                ipcDataNumPreBlock);
            srcLocalQue.DeQue(rank, flagIdx + (srcRank / SIO) * coreNumPerStage + flagNumPerStage);
            writeGt = srcLocalQue.EnQue();
            if (copyLen > 0) {
                CpGM2GMPingPong<T>(copyLen * sizeof(T), inputGt[sliceIdx * perQueElemLen], writeGt, Op::COPYONLY);
                sync.SetSyncFlag(magic, sliceIdx + sliceNum * idx, flagIdx, rank);
            }
        } else {
            flagIdx = flagIdx - singleStage;
            if (idx > 0) {
                int64_t waitRank = (sioSrcRank - SIO) >= 0 ? (sioSrcRank - SIO) : sioSrcRank + ((loopCount - 1) * SIO);
                sync.WaitSyncFlag(magic, sliceIdx + sliceNum * (idx - 1), flagIdx + (waitRank / SIO) * coreNumPerStage
                                + flagNumPerStage, sioRank);
            }
            inputGt.SetGlobalBuffer((__gm__ T*)input + sioSrcRank * curRankDataNum +
                                (groupCoreIdx - singleStage) * ipcDataNumPreBlock, ipcDataNumPreBlock);
            srcSioQue.DeQue(sioRank, flagIdx + (sioSrcRank / SIO) * coreNumPerStage + flagNumPerStage);
            writeGt = srcSioQue.EnQue();
            if (copyLen > 0) {
                CpGM2GMPingPong<T>(copyLen * sizeof(T), inputGt[sliceIdx * perQueElemLen], writeGt, Op::COPYONLY);
                sync.SetSyncFlag(magic, sliceIdx + sliceNum * idx, flagIdx, sioRank);
            }
        }
    }
    FORCE_INLINE_AICORE void Consumer()
    {
        for (auto i = 0; i < loopCount; ++i) {
            destRank = (rank - i * SIO) >= 0 ? (rank - i * SIO) : rank + ((loopCount - i) * SIO);
                if (groupCoreIdx < singleStage) {
                    detHccsQue.Init(&sync, magic, shareAddrs[destRank] + IPC_DATA_OFFSET +
                        groupCoreIdx * queSize, queLen, perQueElemLen);
                } else {
                    detHccsSioQue.Init(&sync, magic, shareAddrs[destRank] + IPC_DATA_OFFSET + queBlockSize +
                        (groupCoreIdx - singleStage) * queSize, queLen, perQueElemLen);
                }
            for (auto sliceIdx = 0; sliceIdx < sliceNum; ++sliceIdx) {
                Ipc2Output(i, sliceIdx);
            }
        }
    }

    FORCE_INLINE_AICORE void Ipc2Output(int64_t idx, int64_t sliceIdx)
    {
        outputGt.SetGlobalBuffer((__gm__ T*)output + (destRank / SIO) * SIO * curRankDataNum
                                + groupCoreIdx * ipcDataNumPreBlock, ipcDataNumPreBlock);
        copyLen = ipcDataNumPreBlock - perQueElemLen * sliceIdx;
        if (copyLen > perQueElemLen) {
            copyLen = perQueElemLen;
        } else if (copyLen < 0) {
            copyLen = 0;
        }
        sync.WaitSyncFlag(magic, sliceIdx + sliceNum * idx, groupCoreIdx, destRank);
        if (groupCoreIdx < singleStage) {
            readGt = detHccsQue.ReadFront();
        } else {
            readGt = detHccsSioQue.ReadFront();
        }
        CpGM2GMPingPong<T>(copyLen * sizeof(T), readGt, outputGt[sliceIdx * perQueElemLen], Op::COPYONLY);
        sync.SetSyncFlag(magic, sliceIdx + sliceNum * idx, groupCoreIdx + flagNumPerStage +
            (rank / SIO) * coreNumPerStage, destRank);
    }
    GlobalTensor<T> inputGt;
    GlobalTensor<T> readGt;
    GlobalTensor<T> writeGt;
    GlobalTensor<T> outputGt;
    __gm__ T *input;
    __gm__ T *output;

    int atomOp;
    IpcQueue<T> srcLocalQue;
    IpcQueue<T> srcSioQue;
    IpcQueue<T> detHccsQue;
    IpcQueue<T> detHccsSioQue;
    IpcQueue<T> srcLocalQue1;
    IpcQueue<T> srcSioQue1;

    int64_t loopCount;
    int64_t queBlockSize;
    int64_t srcRank;
    int64_t sioSrcRank;
    int64_t destRank;
    int64_t singleStage;
    int64_t curRankDataNum;
    int64_t ipcDataNumPreBlock;
    int64_t sioRank = (rank % 2 == 0) ? rank + 1:rank - 1;
    int64_t perQueElemLen;
    int64_t queLen;
    int64_t queSize;
    int64_t coreNumPerStage;
    int64_t flagNumPerStage;
    int64_t coreNumPerRank;
    int64_t coreGroup;
    int64_t groupCoreIdx;
    int64_t sliceNum;
    int64_t copyLen;
};

#endif // LCCL_ALL2ALL_HIERARCHY_SMALL_H
