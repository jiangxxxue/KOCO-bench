/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LCCL_ALLGATHER_H
#define LCCL_ALLGATHER_H

#include "collectives.h"

using namespace AscendC;

constexpr int64_t MEM_DMA_UNIT_SIZE = MEM_DMA_UNIT_INT_NUM * sizeof(int64_t);

constexpr int64_t STEP1 = 1;

template <typename T>

class AllGather : public Collectives {
public:
    FORCE_INLINE_AICORE AllGather(int rank, int rankSize, uint32_t extraFlag)
        : Collectives(rank, rankSize, extraFlag) {}

    FORCE_INLINE_AICORE void Init(KERNELS_ARGS_FUN())
    {
        Collectives::Init(KERNELS_ARGS_CALL());
        globalRank = (reinterpret_cast<__gm__ CommArgs *>(commArgs))->rank;
        globalRankSize = (reinterpret_cast<__gm__ CommArgs *>(commArgs))->rankSize;
        localRankSize = (reinterpret_cast<__gm__ CommArgs *>(commArgs))->localRankSize;
        baseOffsetSize = IPC_DATA_OFFSET;
        GetBlockDataCount(len, blockNum, offsetFromInput, countToShare);
        offsetToShare = offsetFromInput;

        inputGm.SetGlobalBuffer((__gm__ T*)input + offsetFromInput, countToShare);
        if (extraFlag & ExtraFlag::RDMA) {
            blockNumPerRank = blockNum / localRankSize;
            useCoreNumToOutput = blockNumPerRank * localRankSize;
        } else {
            blockNumPerRank = blockNum / rankSize;
            useCoreNumToOutput = blockNumPerRank * rankSize;
        }
        if (blockIdx >= useCoreNumToOutput) {
            return;
        }
        GetBlockDataCount(len, blockNumPerRank, offsetFromShare, countToOutput);
        blockRank = blockIdx / blockNumPerRank;
        offsetToOutput = blockRank * len + offsetFromShare;

        if ((extraFlag & ExtraFlag::RDMA) == 0) {
            outputGm.SetGlobalBuffer((__gm__ T*)output + offsetToOutput, countToOutput);
        }
    }
    FORCE_INLINE_AICORE void Process()
    {
        if (extraFlag & ExtraFlag::RDMA) {
            shareGm.SetGlobalBuffer((__gm__ T*)(shareAddrs[rank % localRankSize] + baseOffsetSize) +
                len * globalRank + offsetToShare, countToShare);
            if (countToShare > 0) {
                CpGM2GMPingPong<T>(countToShare * sizeof(T), inputGm, shareGm, COPYONLY);
            }
            sync.SetInnerFlag(magic, STEP1);
            sync.WaitRankInnerFlag(magic, STEP1, blockRank);
            if (blockIdx >= useCoreNumToOutput) {
                return;
            }
            outputGm.SetGlobalBuffer((__gm__ T*)(shareAddrs[globalRank % localRankSize] + baseOffsetSize) +
                len * (globalRank / localRankSize) * localRankSize + offsetToOutput, countToOutput);
            shareGm.SetGlobalBuffer((__gm__ T*)(shareAddrs[blockRank] + baseOffsetSize) +
                len * (globalRank / localRankSize) * localRankSize + offsetToOutput, countToOutput);
            if (countToOutput > 0 && blockRank != rank) {
                CpGM2GMPingPong<T>(countToOutput * sizeof(T), shareGm, outputGm, COPYONLY);
            }
        } else {
            shareGm.SetGlobalBuffer((__gm__ T*)(shareAddrs[rank] + baseOffsetSize) + offsetToShare, countToShare);
            if (countToShare > 0) {
                CpGM2GM<T>(shareGm, inputGm, countToShare, COPYONLY);
            }
            sync.SetInnerFlag(magic, STEP1);
            sync.WaitRankInnerFlag(magic, STEP1, blockRank);
            if (blockIdx >= useCoreNumToOutput) {
                return;
            }
            shareGm.SetGlobalBuffer((__gm__ T*)(shareAddrs[blockRank] + baseOffsetSize) + offsetFromShare,
                countToOutput);
            if (countToOutput > 0) {
                CpGM2GM<T>(outputGm, shareGm, countToOutput, COPYONLY);
            }
        }
    }

private:

    FORCE_INLINE_AICORE void GetBlockDataCount(
        const int64_t dataLen, const int64_t useBlockNum, int64_t& blockDataOffset, int64_t& blockDataCount)
    {
        blockDataCount = CeilDiv(dataLen, useBlockNum);
        blockDataCount = blockDataCount > MEM_DMA_UNIT_SIZE / sizeof(T) ?
                         blockDataCount : MEM_DMA_UNIT_SIZE / sizeof(T);
        blockDataOffset = blockIdx % useBlockNum * blockDataCount;
        if (blockDataOffset >= dataLen) {
            blockDataOffset = dataLen;
            blockDataCount = 0;
            return;
        }
        if (blockDataOffset + blockDataCount > dataLen) {
            blockDataCount = dataLen - blockDataOffset;
        }
    }
private:
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;
    GlobalTensor<T> shareGm;

    int64_t baseOffsetSize;
    int64_t offsetFromInput;
    int64_t offsetToShare;
    int64_t countToShare;
    int64_t useCoreNumToOutput;
    int64_t blockNumPerRank;
    int64_t blockRank;
    int64_t offsetFromShare;;
    int64_t offsetToOutput;
    int64_t countToOutput;
    int globalRank;
    int globalRankSize;
    int localRankSize;
};

#endif // LCCL_ALLREDUCE_TWO_SHOT_H