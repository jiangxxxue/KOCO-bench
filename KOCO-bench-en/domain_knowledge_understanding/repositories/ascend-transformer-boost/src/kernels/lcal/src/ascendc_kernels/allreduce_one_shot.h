/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LCCL_ALLREDUCE_ONE_SHOT_H
#define LCCL_ALLREDUCE_ONE_SHOT_H

#include "sync_collectives.h"
#include "allreduce_quant.h"

using namespace AscendC;
template <typename T, typename U = T>
class AllReduceOneShot : protected AllReduceQuant {
    constexpr static T oneCast = (T) 1;

public:
    FORCE_INLINE_AICORE AllReduceOneShot(int rank, int rankSize, uint32_t extraFlag)
        : AllReduceQuant(rank, rankSize, extraFlag) {}
    FORCE_INLINE_AICORE void Init(KERNELS_ARGS_FUN())
    {
        Collectives::Init(KERNELS_ARGS_CALL());
        DumpLcclLogInfo(LogId::INIT, static_cast<Op>(op));
        if constexpr(!std::is_same_v<T, U>) {
            BuildScaleOffset(scale, scaleCount, offset);
        }
        atomOp = op;
        blockNum = blockNum / rankSize * rankSize;
        if (blockIdx >= blockNum) {
            DumpLcclLogInfo(LogId::INIT, static_cast<Op>(op));
            return;
        }

        corePerRank = blockNum / rankSize;
        rankIDOfBlock = blockIdx / corePerRank;

        dataDMAPerCore = len / rankSize / corePerRank / scaleNum * scaleNum;
        dataReducePerCore = len / corePerRank / scaleNum * scaleNum;

        blockDataNum = dataDMAPerCore;
        if (blockIdx == rankSize * corePerRank - 1) {
            blockDataNum = len - blockIdx * dataDMAPerCore;
        }

        blockReduceNum = dataReducePerCore;
        if (blockIdx % corePerRank == corePerRank - 1) {
            blockReduceNum = len - blockIdx % corePerRank * dataReducePerCore;
        }

        __gm__ U* curRankGm = (__gm__ U*)shareAddrs[rank] + IPC_DATA_OFFSET / sizeof(U);
        __gm__ U* peerRankGm = (__gm__ U*)shareAddrs[rankIDOfBlock] + IPC_DATA_OFFSET / sizeof(U);
        __gm__ U* intputGm = (__gm__ U*)input;
        __gm__ T* outputGm = (__gm__ T*)output;

        srcInputGlobal.SetGlobalBuffer(intputGm + blockIdx * dataDMAPerCore);
        dstIPCGlobal.SetGlobalBuffer(curRankGm + blockIdx * dataDMAPerCore);
        copyOutputGlobal.SetGlobalBuffer(outputGm + blockIdx * dataDMAPerCore);
        srcIPCGlobal.SetGlobalBuffer(peerRankGm + blockIdx % corePerRank * dataReducePerCore);
        dstOutputGlobal.SetGlobalBuffer(outputGm + blockIdx % corePerRank * dataReducePerCore);
        DumpLcclLogInfo(LogId::INIT, static_cast<Op>(op));
    }

    FORCE_INLINE_AICORE void Process()
    {
        DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
        if (blockIdx >= blockNum) {
            DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
            return;
        }
        CpInputToBuffAndOutput();
        sync.SetInnerFlag(magic, 1);

        sync.WaitRankInnerFlag(magic, 1, rank);
        sync.WaitRankInnerFlag(magic, 1, rankIDOfBlock);
        if (rankIDOfBlock != rank) {
            if constexpr (!std::is_same_v<T, U>) {
                if (!isEnableScale) {
                    Collectives::CpGM2GM(dstOutputGlobal, srcIPCGlobal, blockReduceNum, atomOp);
                } else if (!isVectorScale) {
                    CpGM2GM(dstOutputGlobal, srcIPCGlobal, blockReduceNum, atomOp, firstScale, offset);
                } else {
                    CpGM2GM(dstOutputGlobal, srcIPCGlobal, blockReduceNum, atomOp, scaleGt, scaleNum, offset);
                }
            } else {
                Collectives::CpGM2GM<T, U>(dstOutputGlobal, srcIPCGlobal, blockReduceNum, atomOp);
            }
        }
        DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
    }

    FORCE_INLINE_AICORE void CpInputToBuffAndOutput()
    {
        Collectives::CpGM2GM<U>(dstIPCGlobal, srcInputGlobal, blockDataNum, COPYONLY);
        if constexpr (!std::is_same_v<T, U>) {
            if (!isEnableScale) {
                Collectives::CpGM2GM(copyOutputGlobal, srcInputGlobal, blockDataNum, COPYONLY);
            } else if (!isVectorScale) {
                CpGM2GM(copyOutputGlobal, srcInputGlobal, blockDataNum, COPYONLY, firstScale, offset);
            } else {
                CpGM2GM(copyOutputGlobal, srcInputGlobal, blockDataNum, COPYONLY, scaleGt, scaleNum, offset);
            }
        } else {
            Collectives::CpGM2GM<T, U>(copyOutputGlobal, srcInputGlobal, blockDataNum, -1);
        }
    }

protected:
    GlobalTensor<U> srcInputGlobal;
    GlobalTensor<U> srcIPCGlobal;
    GlobalTensor<U> dstIPCGlobal;
    GlobalTensor<T> dstOutputGlobal;
    GlobalTensor<T> copyOutputGlobal;

    int rankIDOfBlock;
    int corePerRank;
    int dataDMAPerCore;
    int dataReducePerCore;
    int blockDataNum;
    int blockReduceNum;
    int atomOp;
    GlobalTensor<T> scaleGt;
    int64_t scaleNum = 1;
    T firstScale = 1;
    T offset = 0;
    bool isEnableScale = false;
    bool isVectorScale = false;

private:
    FORCE_INLINE_AICORE void BuildScaleOffset(GM_ADDR scale, int64_t scaleCount, GM_ADDR offset)
    {
        if (scale != nullptr && offset != nullptr) {
            this->offset =* reinterpret_cast<__gm__ T*>(offset);
            scaleGt.SetGlobalBuffer((__gm__ T*)scale);
            this->firstScale = scaleGt.GetValue(0);
            this->scaleNum = scaleCount < 1 ? 1 : scaleCount;
            isVectorScale = scaleCount > 1;
            isEnableScale = scaleCount > 0 && !(*(uint16_t *)(&(this->offset)) == 0 &&
                scaleCount == 1 && *(uint16_t *)(&firstScale) == *(uint16_t *)(&oneCast));
        }
    }
};

#endif // LCCL_ALLREDUCE_ONE_SHOT_H