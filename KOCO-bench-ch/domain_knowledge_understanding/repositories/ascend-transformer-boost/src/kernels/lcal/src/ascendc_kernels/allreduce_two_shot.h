/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LCCL_ALLREDUCE_TWO_SHOT_H
#define LCCL_ALLREDUCE_TWO_SHOT_H

#include "allreduce_quant.h"
#include "sync_collectives.h"
using namespace AscendC;
template <typename T, typename U = T>
class AllReduceTwoShot : protected AllReduceQuant {
    constexpr static int QUEUE_DEPTH = 4;
    constexpr static T oneCast = (T) 1;

public:
    FORCE_INLINE_AICORE AllReduceTwoShot(int rank, int rankSize, uint32_t extraFlag)
        : AllReduceQuant(rank, rankSize, extraFlag) {}
    FORCE_INLINE_AICORE void Init(KERNELS_ARGS_FUN())
    {
        Collectives::Init(KERNELS_ARGS_CALL());
        DumpLcclLogInfo(LogId::INIT, static_cast<Op>(op));
        if constexpr(!std::is_same_v<T, U>) {
            BuildScaleOffset(scale, scaleCount, offset);
        }

        if (blockIdx >= rankSize) {
            DumpLcclLogInfo(LogId::INIT, static_cast<Op>(op));
            return;
        }

        blockNum = rankSize;

        __gm__ CommArgs *localArgs = reinterpret_cast<__gm__ CommArgs *>(commArgs);

        int localRankSize = localArgs->localRankSize <= 0 ? rankSize : localArgs->localRankSize;
        int globalRankSize = localArgs->rankSize <= 0 ? rankSize : localArgs->rankSize;
        int serverNum = globalRankSize / localRankSize;
        int64_t ipcBuffMaxSizeAligned = IPC_BUFF_MAX_SIZE / (globalRankSize + serverNum - 1) /
            QUEUE_DEPTH / sizeof(T) /scaleNum * scaleNum * QUEUE_DEPTH * sizeof(T) * globalRankSize;
        ipcDataPerParagraphSize = ipcBuffMaxSizeAligned / localRankSize;
        int64_t ipcDataPerParagraphNum = ipcDataPerParagraphSize / sizeof(T);
        atomOp = op;
        corePerRank = blockNum / rankSize;
        coreSegmentedIdx = blockIdx % corePerRank;
        peerRank = blockIdx / corePerRank;
        perRankDataNum = GetDataCount(len, rankSize) / scaleNum * scaleNum;
        curRankDataNum = (rank == rankSize - 1) ? (len - rank * perRankDataNum) : perRankDataNum;
        pullRankDataNum = perRankDataNum;
        if (peerRank == rankSize - 1) {
            pullRankDataNum = len - peerRank * perRankDataNum;
        }
        pullBlockDataNum = GetDataCount(pullRankDataNum, corePerRank);
        dataNumPreBlock = pullBlockDataNum;
        if (coreSegmentedIdx == corePerRank - 1) {
            dataNumPreBlock = pullRankDataNum - coreSegmentedIdx * pullBlockDataNum;
        }
        buffOffsetNum = peerRank * perRankDataNum + coreSegmentedIdx * pullBlockDataNum +
                        ipcDataPerParagraphNum * peerRank;

        curBlockDataNum = GetDataCount(curRankDataNum, corePerRank);
        ipcDataNumPreBlock = curBlockDataNum;
        ipcbuffOffsetNum = rank * perRankDataNum + coreSegmentedIdx * curBlockDataNum + ipcDataPerParagraphNum * rank;

        inputGt.SetGlobalBuffer((__gm__ U*)input + buffOffsetNum - ipcDataPerParagraphNum * peerRank, dataNumPreBlock);
        inputIpcGt.SetGlobalBuffer((__gm__ T*)(shareAddrs[rank] + IPC_DATA_OFFSET) + buffOffsetNum, dataNumPreBlock);
        srcIpcGt.SetGlobalBuffer((__gm__ T*)(shareAddrs[peerRank] + IPC_DATA_OFFSET) + ipcbuffOffsetNum,
                                            ipcDataNumPreBlock);
        processIpcGt.SetGlobalBuffer((__gm__ T*)(shareAddrs[rank] + IPC_DATA_OFFSET) + ipcbuffOffsetNum,
                                    ipcDataNumPreBlock);

        processedIpcGt.SetGlobalBuffer((__gm__ T*)(shareAddrs[peerRank] + IPC_DATA_OFFSET) + buffOffsetNum,
                dataNumPreBlock);
        outputGt.SetGlobalBuffer((__gm__ T*)output + buffOffsetNum - ipcDataPerParagraphNum * peerRank,
                                 dataNumPreBlock);
        DumpLcclLogInfo(LogId::INIT, static_cast<Op>(op));
    }

    FORCE_INLINE_AICORE void Process()
    {
        DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
        if (blockIdx >= rankSize) {
            DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
            return;
        }
        if constexpr (std::is_same_v<T, U>) {
            Collectives::CpGM2GM(inputIpcGt, inputGt, dataNumPreBlock, COPYONLY);
        } else {
            if (peerRank == rank) {
                if (!isEnableScale) {
                    Collectives::CpGM2GM(inputIpcGt, inputGt, dataNumPreBlock, COPYONLY);
                } else if (!isVectorScale) {
                    CpGM2GM(inputIpcGt, inputGt, dataNumPreBlock, COPYONLY, firstScale, offset);
                } else {
                    CpGM2GM(inputIpcGt, inputGt, dataNumPreBlock, COPYONLY, scaleGt, scaleNum, offset);
                }
            } else {
                GlobalTensor<U> inputIpcGtTmp;
                inputIpcGtTmp.SetGlobalBuffer((__gm__ U*)inputIpcGt.GetPhyAddr());
                Collectives::CpGM2GM(inputIpcGtTmp, inputGt, dataNumPreBlock, COPYONLY);
            }
        }
        sync.SetInnerFlag(magic, 1);

        sync.WaitInnerFlag(magic, 1, rank, coreSegmentedIdx + rank * corePerRank);
        sync.WaitInnerFlag(magic, 1, peerRank, coreSegmentedIdx + rank * corePerRank);
        if (peerRank != rank) {
            if constexpr (std::is_same_v<T, U>) {
                Collectives::CpGM2GM(processIpcGt, srcIpcGt, ipcDataNumPreBlock, atomOp);
            } else {
                GlobalTensor<U> srcIpcGtTmp;
                srcIpcGtTmp.SetGlobalBuffer((__gm__ U*)srcIpcGt.GetPhyAddr());
                if (!isEnableScale) {
                    Collectives::CpGM2GM(processIpcGt, srcIpcGtTmp, ipcDataNumPreBlock, atomOp);
                } else if (!isVectorScale) {
                    CpGM2GM(processIpcGt, srcIpcGtTmp, ipcDataNumPreBlock, atomOp, firstScale, offset);
                } else {
                    CpGM2GM(processIpcGt, srcIpcGtTmp, ipcDataNumPreBlock, atomOp, scaleGt, scaleNum, offset);
                }
            }
        }

        if (!(extraFlag & ExtraFlag::RDMA)) {
            sync.SetOuterFlag(magic, 1);
            sync.WaitOneRankOuterFlag(magic, 1, peerRank);
            Collectives::CpGM2GM(outputGt, processedIpcGt, dataNumPreBlock, COPYONLY);
        }
        DumpLcclLogInfo(LogId::PROCESS, static_cast<Op>(atomOp));
    }

private:
    GlobalTensor<U> inputGt;
    GlobalTensor<T> outputGt;
    GlobalTensor<T> inputIpcGt;
    GlobalTensor<T> srcIpcGt;
    GlobalTensor<T> processedIpcGt;
    GlobalTensor<T> processIpcGt;

    int atomOp;

    int64_t corePerRank;
    int64_t coreSegmentedIdx;
    int64_t ipcDataPerParagraphSize;
    int64_t perRankDataNum;
    int64_t curRankDataNum;
    int64_t pullBlockDataNum;
    int64_t curBlockDataNum;
    int64_t peerRank;
    int64_t pullRankDataNum;
    int64_t dataNumPreBlock;
    int64_t buffOffsetNum;
    int64_t ipcDataNumPreBlock;
    int64_t ipcbuffOffsetNum;

    GlobalTensor<T> scaleGt;
    int64_t scaleNum = 1;
    T firstScale = 1;
    T offset = 0;
    bool isEnableScale = false;
    bool isVectorScale = false;
    FORCE_INLINE_AICORE void BuildScaleOffset(GM_ADDR scale, int64_t scaleCount, GM_ADDR offset)
    {
        if (scale != nullptr && offset != nullptr) {
            scaleGt.SetGlobalBuffer((__gm__ T*)scale);
            this->firstScale = scaleGt.GetValue(0);
            this->scaleNum = scaleCount < 1 ? 1 : scaleCount;
            this->offset =* reinterpret_cast<__gm__ T*>(offset);
            isVectorScale = scaleCount > 1;
            isEnableScale = scaleCount > 0 && !(*(uint16_t *)(&(this->offset)) == 0 &&
                scaleCount == 1 && *(uint16_t *)(&firstScale) == *(uint16_t *)(&oneCast));
        }
    }
};

#endif // LCCL_ALLREDUCE_TWO_SHOT_H