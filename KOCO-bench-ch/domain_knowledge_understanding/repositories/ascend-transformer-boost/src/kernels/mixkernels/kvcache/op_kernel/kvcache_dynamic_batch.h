/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KVCACHE_DYNAMIC_BATCH_H
#define KVCACHE_DYNAMIC_BATCH_H
#include "kvcache_nd_init_tiling.h"

// tiling hiddensize、maxseqlen
constexpr uint32_t MAX_PROCESS_NUM = 12288 * 8;
constexpr uint32_t BLOCK_SIZE = 16;

class KvcacheDynamicBatch {
public:
    __aicore__ inline KvcacheDynamicBatch(uint32_t batch, uint32_t hiddenSize, uint32_t maxSeqLen)
    {
        this->hiddenSize = hiddenSize;
        this->maxSeqLen = maxSeqLen;
        this->batch = batch;
        this->blockIdx = AscendC::GetBlockIdx();
    }

    __aicore__ inline void Init(__gm__ uint8_t *newKV, __gm__ uint8_t *layerId, __gm__ uint8_t *cacheIn,
                                __gm__ uint8_t *tokenOffset, __gm__ uint8_t *seqLen, __gm__ uint8_t *batchRunStatus,
                                __gm__ uint8_t *cacheOut, __gm__ uint8_t *tiling)
    {
        cacheOut = cacheIn;
        batchStatus = (int32_t)(*((__gm__ uint32_t *)batchRunStatus + blockIdx));
        if (batchStatus == 0) { // 当前batch不参与拼接
            return;
        }
        newKVGm = (__gm__ half *)newKV;
        layerIdGm = (__gm__ uint32_t *)layerId;
        cacheInGm = (__gm__ half *)cacheIn;
        tokenOffsetGm = (__gm__ uint32_t *)tokenOffset;
        seqLenGm = (__gm__ uint32_t *)seqLen;
        pipe.InitBuffer(outQueue, 1, (MAX_PROCESS_NUM * sizeof(half)));
        AscendC::LocalTensor<half> cachePerloopUb = outQueue.AllocTensor<half>();
        this->tokenOffset = *(this->tokenOffsetGm + blockIdx);
        this->seqLen = *(this->seqLenGm + blockIdx);
        this->tokenOffset -= this->seqLen;
        this->layerId = *(layerIdGm);
        for (uint32_t i = 0; i < blockIdx; ++i) {
            prefixOfNtokens += *(this->seqLenGm + i);
        }

        cacheBuff = cacheInGm;
        WriteKvCache(newKVGm, cacheBuff, 0, cachePerloopUb);
        outQueue.FreeTensor(cachePerloopUb);
    }

    __aicore__ inline void WriteKvCache(__gm__ half *oldCache, __gm__ half *newCache, uint64_t kvOffset,
                                        AscendC::LocalTensor<half> &cachePerloopUb)
    {
        if (g_coreType == AscendC::AIC) {
            return;
        }
        AscendC::GlobalTensor<half> oldCacheGm;
        AscendC::GlobalTensor<half> newCacheGm;
        oldCacheGm.SetGlobalBuffer((__gm__ half *)oldCache);
        newCacheGm.SetGlobalBuffer((__gm__ half *)newCache);

        uint64_t tokensPerLoop = MAX_PROCESS_NUM / hiddenSize;
        uint64_t loopTimes = seqLen / tokensPerLoop;
        uint64_t tailTokens = seqLen % tokensPerLoop;
        uint64_t tailLoopTimes = tailTokens == 0 ? 0 : 1;

        uint64_t qkvSplitBatchOffset = prefixOfNtokens * hiddenSize;
        uint64_t kvCacheBatchOffset = blockIdx * maxSeqLen * hiddenSize + tokenOffset * hiddenSize;

        AscendC::DataCopyParams copyParams = {static_cast<uint16_t>(tokensPerLoop),
                                     static_cast<uint16_t>(hiddenSize / BLOCK_SIZE), 0, 0};
        for (uint64_t loop = 0; loop < loopTimes; ++loop) {
            DataCopy(cachePerloopUb, oldCacheGm[qkvSplitBatchOffset + kvOffset + loop * tokensPerLoop * hiddenSize],
                     copyParams);
            AscendC::PipeBarrier<PIPE_ALL>();
            DataCopy(newCacheGm[kvCacheBatchOffset + loop * tokensPerLoop * hiddenSize], cachePerloopUb, copyParams);
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        // process tail
        copyParams = {static_cast<uint16_t>(tailTokens), static_cast<uint16_t>(hiddenSize / BLOCK_SIZE), 0, 0};
        for (uint64_t loop = 0; loop < tailLoopTimes; ++loop) {
            DataCopy(cachePerloopUb,
                     oldCacheGm[qkvSplitBatchOffset + kvOffset + loopTimes * tokensPerLoop * hiddenSize], copyParams);
            AscendC::PipeBarrier<PIPE_ALL>();
            DataCopy(newCacheGm[kvCacheBatchOffset + loopTimes * tokensPerLoop * hiddenSize], cachePerloopUb,
                     copyParams);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    /* data */
    __gm__ half *newKVGm{nullptr};
    __gm__ uint32_t *layerIdGm{nullptr};
    __gm__ half *cacheInGm{nullptr};
    __gm__ uint32_t *tokenOffsetGm{nullptr};
    __gm__ uint32_t *seqLenGm{nullptr};
    __gm__ half *cacheBuff{nullptr};
    uint64_t batch{0};
    uint64_t hiddenSize{0};
    uint64_t maxSeqLen{0};
    uint64_t layerId{0};
    uint64_t prefixOfNtokens{0};
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> outQueue;
    uint64_t tokenOffset{0};
    uint64_t seqLen{0};
    int32_t batchStatus{1};
    uint64_t blockIdx{0};
};
#endif