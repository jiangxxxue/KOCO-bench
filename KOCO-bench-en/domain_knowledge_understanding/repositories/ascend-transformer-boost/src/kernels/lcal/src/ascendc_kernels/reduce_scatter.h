/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LCCL_REDUCE_SCATTER_H
#define LCCL_REDUCE_SCATTER_H

#include "sync_collectives.h"
#include "collectives.h"
using namespace AscendC;

template<typename T>
class ReduceScatter : protected Collectives {
public:
    FORCE_INLINE_AICORE ReduceScatter(int rank, int rankSize, uint32_t extraFlag)
        : Collectives(rank, rankSize, extraFlag) {}

    FORCE_INLINE_AICORE void Init(KERNELS_ARGS_FUN())
    {
        Collectives::Init(KERNELS_ARGS_CALL());
        DumpLcclLogInfo(LogId::INIT, static_cast<Op>(op));
        atomOp = op;
        DMANumMax = BLOCK_SIZE / sizeof(T);
        corePerRank = blockNum / rankSize;
        rankIDOfBlock = blockIdx / corePerRank;
        dataDMAPerCore = CeilDiv(len, corePerRank);
        blockIdxOfLen = blockIdx % corePerRank;
        if (blockIdxOfLen == corePerRank - 1) {
            blockDataNum = len - blockIdxOfLen * dataDMAPerCore;
        } else {
            blockDataNum = dataDMAPerCore;
        }
        inputOffset = rankIDOfBlock * len + (blockIdx % corePerRank) * dataDMAPerCore;
        outputOffset = dataDMAPerCore * (blockIdx % corePerRank);
        dstIpcDataOffset = IPC_DATA_OFFSET / sizeof(T) + rankIDOfBlock * len + outputOffset;
        srcIpcDataOffset = IPC_DATA_OFFSET / sizeof(T) + rank * len + outputOffset;
        srcInputGlobal.SetGlobalBuffer((__gm__ T*)input + inputOffset, blockDataNum);
        if ((extraFlag & ExtraFlag::RDMA) == ExtraFlag::RDMA) {
            dstOutputGlobal.SetGlobalBuffer((__gm__ T*)shareAddrs[rank] + srcIpcDataOffset, blockDataNum);
        } else {
            dstOutputGlobal.SetGlobalBuffer((__gm__ T*)output + outputOffset, blockDataNum);
        }
        dstIPCGlobal.SetGlobalBuffer((__gm__ T*)shareAddrs[rank] + dstIpcDataOffset, blockDataNum);
        srcIPCGlobal.SetGlobalBuffer((__gm__ T*)shareAddrs[rankIDOfBlock] + srcIpcDataOffset, blockDataNum);
        DumpLcclLogInfo(LogId::INIT, static_cast<Op>(op));
    }
    FORCE_INLINE_AICORE void Process()
    {
        DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
        CpInputToBuffAndOutput();
        sync.SetInnerFlag(magic, 1);
        sync.WaitRankInnerFlag(magic, 1, rank);
        sync.WaitInnerFlag(magic, 1, rankIDOfBlock, rank * corePerRank + blockIdx % corePerRank);
        if (rankIDOfBlock != rank) {
            CpGM2GM<T>(dstOutputGlobal, srcIPCGlobal, blockDataNum, atomOp);
        }
        DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
    }

    FORCE_INLINE_AICORE void CpInputToBuffAndOutput()
    {
        CpGM2GM<T>(dstIPCGlobal, srcInputGlobal, blockDataNum, -1);
        if ((extraFlag & ExtraFlag::RDMA) != ExtraFlag::RDMA) {
            if ((blockIdx >= rank * corePerRank) && (blockIdx < (rank * corePerRank + corePerRank))) {
                CpGM2GM<T>(dstOutputGlobal, srcInputGlobal, blockDataNum, -1);
            }
        }
    }

protected:
    GlobalTensor<T> srcInputGlobal;
    GlobalTensor<T> srcIPCGlobal;
    GlobalTensor<T> dstIPCGlobal;
    GlobalTensor<T> dstOutputGlobal;
    int blockIdxOfLen;
    int DMANumMax;
    int rankIDOfBlock;
    int corePerRank;
    int inputOffset;
    int outputOffset;
    int srcIpcDataOffset;
    int dstIpcDataOffset;
    int dataDMAPerCore;
    int blockDataNum;
    int atomOp;
};
#endif // LCCL_REDUCE_SCATTER_H