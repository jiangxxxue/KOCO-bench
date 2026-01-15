/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KVCACHE_NZ_H
#define KVCACHE_NZ_H
#include "kernel_operator.h"
#include "../tiling_data.h"
#include "mixkernels/utils/common/kernel/kernel_utils.h"

constexpr uint32_t MAX_PROCESS_NUM = 12288 * 8;
constexpr uint32_t BLOCK_SIZE = 16;
constexpr uint32_t BLOCK_BYTE = 32;
constexpr uint32_t INT_BYTE = 4;
constexpr uint32_t STRIDE_MAX = 65536;

using AscendC::HardEvent;

class KvCacheNz {
public:
    __aicore__ inline KvCacheNz(uint32_t batch, uint32_t hiddenSize, uint32_t maxSeqLen, uint32_t batchPerCore)
    {
        this->hiddenSize = hiddenSize;
        this->maxSeqLen = maxSeqLen;
        this->batch = batch;
        this->batchPerCore = batchPerCore;
        this->blockIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
    }
    __aicore__ inline void Init(__gm__ uint8_t *newKV, __gm__ uint8_t *layerId, __gm__ uint8_t *cacheIn,
                                __gm__ uint8_t *tokenOffset, __gm__ uint8_t *seqLen, __gm__ uint8_t *cacheOut,
                                __gm__ uint8_t *tiling)
    {
        newKVGm = (__gm__ half *)newKV;
        cacheOutGm = (__gm__ half *)cacheOut;
        cacheInGm = (__gm__ half *)cacheIn;
        layerIdGm.SetGlobalBuffer((__gm__ uint32_t *)layerId);
        tokenOffsetGm.SetGlobalBuffer((__gm__ uint32_t *)tokenOffset);
        seqLenGm.SetGlobalBuffer((__gm__ uint32_t *)seqLen);

        /* 申请整块buffer，前部分按照block对齐储存所有batch的tokenOffset与seqlen,后部分划分两块存放数据 */
        /* 7 8 一个block存放8个unit32, tokenOffsetSize与seqSize长度相同 */
        uint32_t tokenOffsetSize = CeilDiv(this->batch * INT_BYTE, BLOCK_BYTE) * BLOCK_BYTE;
        pipe.InitBuffer(outQueue, 1, (MAX_PROCESS_NUM * sizeof(half)));
        AscendC::LocalTensor<half> startUbAddr = outQueue.AllocTensor<half>();
        tokenOffsetGmInUb = startUbAddr.ReinterpretCast<uint32_t>();
        seqLenGmInUb = tokenOffsetGmInUb[tokenOffsetSize / (sizeof(uint32_t))];
        AscendC::LocalTensor<half> cachePerloopUb = startUbAddr[2 * tokenOffsetSize / (sizeof(half))];

        DataCopy(tokenOffsetGmInUb, layerIdGm, {1, 1, 0, 0});
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        this->layerId = tokenOffsetGmInUb.GetValue(0);

        DataCopy(seqLenGmInUb, seqLenGm, {1, static_cast<uint16_t>(CeilDiv(this->batch * INT_BYTE, BLOCK_BYTE)), 0, 0});
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);

        /* 搬移当前处理batch的tokenOffset */
        DataCopy(tokenOffsetGmInUb, tokenOffsetGm[blockIdx * batchPerCore],
                 {1, static_cast<uint16_t>(CeilDiv(this->batchPerCore * INT_BYTE, BLOCK_BYTE)), 0, 0});
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID1);

        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        for (uint32_t i = 0; i < batch; ++i) {  // 总的tokens，需要获取到，进行偏移
            ntokens += CeilDiv(seqLenGmInUb.GetValue(i), BLOCK_SIZE) * BLOCK_SIZE;
        }
        for (uint32_t i = 0; i < blockIdx * batchPerCore; ++i) {
            prefixOfNtokens += seqLenGmInUb.GetValue(i);
        }

        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);

        cacheBuff = cacheInGm;
        cacheOut = cacheIn;
        uint64_t offsetCache = static_cast<uint64_t>(this->layerId) * static_cast<uint64_t>(this->batch)
                               * static_cast<uint64_t>(this->maxSeqLen) * static_cast<uint64_t>(this->hiddenSize);
        cacheBuff = cacheBuff + offsetCache;
        /* 除去前面使用的固定内存，后面内存存放数据 */
        uint32_t maxTokensPerLoop = (MAX_PROCESS_NUM - (2 * tokenOffsetSize) / sizeof(half)) / hiddenSize;
        WriteKvCacheNz(newKVGm, cacheBuff, cachePerloopUb, maxTokensPerLoop);
        outQueue.FreeTensor(startUbAddr);
    }
    __aicore__ inline void WriteKvCacheNz(__gm__ half *oldCache, __gm__ half *newCache,
                                          AscendC::LocalTensor<half> &cachePerloopUb, uint32_t maxTokensPerLoop)
    {
        AscendC::GlobalTensor<half> oldCacheGm;
        AscendC::GlobalTensor<half> newCacheGm;
        oldCacheGm.SetGlobalBuffer((__gm__ half *)oldCache);
        newCacheGm.SetGlobalBuffer((__gm__ half *)newCache);

        AscendC::LocalTensor<uint32_t> batchSeqLen = seqLenGmInUb[blockIdx * batchPerCore];
        uint32_t batchPreCoreReal = batchPerCore;
        if ((blockIdx + 1) * batchPerCore > this->batch) {
            batchPreCoreReal = this->batch - blockIdx * batchPerCore;
        }
        uint32_t tokenNumInLoop = 0;
        for (uint32_t i = 0; i < batchPreCoreReal; i++) {
            tokenNumInLoop += batchSeqLen.GetValue(i);
        }

        uint32_t maxTokensPerBuffer = maxTokensPerLoop / 2; // 2 写死划分double buffer
        AscendC::LocalTensor<half> cachePerloopUbPart2 = cachePerloopUb[maxTokensPerBuffer * hiddenSize];

        uint32_t loopTimes = CeilDiv(tokenNumInLoop, maxTokensPerBuffer);
        uint32_t tailTokens = tokenNumInLoop % maxTokensPerBuffer;
        uint32_t batchIndex = 0;
        uint32_t tokenNow = 0;

        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (uint64_t loop = 0; loop < loopTimes; loop++) {
            uint32_t seqLenSum = 0;
            AscendC::LocalTensor<half> cacheLoopUb = (loop % 2) ? cachePerloopUbPart2 : cachePerloopUb;
            AscendC::LocalTensor<half> cacheUb = cacheLoopUb;
            if (loop == (loopTimes - 1) && (tailTokens != 0)) { // 最后一次且尾块不满
                tokenNow = tailTokens;
                AscendC::WaitFlag<HardEvent::MTE3_MTE2>(loop % 2);
                if (ntokens - tailTokens < STRIDE_MAX) {
                    DataCopy(cacheLoopUb,
                        oldCacheGm[loop * maxTokensPerBuffer * BLOCK_SIZE + prefixOfNtokens * BLOCK_SIZE],
                        {static_cast<uint16_t>(hiddenSize / BLOCK_SIZE), static_cast<uint16_t>(tailTokens),
                        static_cast<uint16_t>(ntokens - tailTokens), 0});
                } else {
                    for (int i = 0; i < hiddenSize / BLOCK_SIZE; ++i) {
                        uint64_t offset = loop * maxTokensPerBuffer * BLOCK_SIZE +
                                          prefixOfNtokens * BLOCK_SIZE + i * ntokens * BLOCK_SIZE;
                        DataCopy(cacheLoopUb[i * tailTokens * BLOCK_SIZE], oldCacheGm[offset],
                            {1, static_cast<uint16_t>(tailTokens), 0, 0});
                    }
                }
            } else {
                tokenNow = maxTokensPerBuffer;
                AscendC::WaitFlag<HardEvent::MTE3_MTE2>(loop % 2);
                if (ntokens - maxTokensPerBuffer < STRIDE_MAX) {
                    DataCopy(cacheLoopUb,
                        oldCacheGm[loop * maxTokensPerBuffer * BLOCK_SIZE + prefixOfNtokens * BLOCK_SIZE],
                        {static_cast<uint16_t>(hiddenSize / BLOCK_SIZE), static_cast<uint16_t>(maxTokensPerBuffer),
                        static_cast<uint16_t>(ntokens - maxTokensPerBuffer), 0});
                } else {
                    for (int i = 0; i < hiddenSize / BLOCK_SIZE; ++i) {
                        uint64_t offset = loop * maxTokensPerBuffer * BLOCK_SIZE +
                                          prefixOfNtokens * BLOCK_SIZE + i * ntokens * BLOCK_SIZE;
                        DataCopy(cacheLoopUb[i * maxTokensPerBuffer * BLOCK_SIZE], oldCacheGm[offset],
                            {1, static_cast<uint16_t>(maxTokensPerBuffer), 0, 0});
                    }
                }
            }
            AscendC::SetFlag<HardEvent::MTE2_MTE3>(loop % 2);
            AscendC::WaitFlag<HardEvent::MTE2_MTE3>(loop % 2);

            /* 对于所有的搬入，只对应batchPreCoreReal个batch的搬出 */
            for (; batchIndex < batchPreCoreReal; batchIndex++) {
                uint32_t seqLenNow = batchSeqLen.GetValue(batchIndex);
                uint32_t tokenOffset = tokenOffsetGmInUb.GetValue(batchIndex);
                tokenOffset -= seqLenNow;
                seqLenSum += seqLenNow;
                uint64_t kvCacheNzBatchOffset =
                    (blockIdx * batchPerCore + batchIndex) * maxSeqLen * hiddenSize + tokenOffset * BLOCK_SIZE;
                if (seqLenSum > maxTokensPerBuffer) {
                    uint32_t tailSeqLen = seqLenSum - maxTokensPerBuffer;
                    uint64_t seqMov = maxTokensPerBuffer - (seqLenSum - seqLenNow);
                    if (seqMov == 0) { // ub内搬完后不做搬移
                        break;
                    }
                    DataCopy(newCacheGm[kvCacheNzBatchOffset], cacheUb,
                             {static_cast<uint16_t>(hiddenSize / BLOCK_SIZE), static_cast<uint16_t>(seqMov),
                              static_cast<uint16_t>(tokenNow - seqMov), static_cast<uint16_t>(maxSeqLen - seqMov)});
                    batchSeqLen.SetValue(batchIndex, tailSeqLen);
                    break;
                } else {
                    DataCopy(newCacheGm[kvCacheNzBatchOffset], cacheUb,
                             {static_cast<uint16_t>(hiddenSize / BLOCK_SIZE), static_cast<uint16_t>(seqLenNow),
                              static_cast<uint16_t>(tokenNow - seqLenNow),
                              static_cast<uint16_t>(maxSeqLen - seqLenNow)});
                    cacheUb = cacheUb[seqLenNow * BLOCK_SIZE];
                }
            }
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(loop % 2);
        }
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

private:
    /* data */
    __gm__ half *newKVGm{nullptr};
    __gm__ half *cacheInGm{nullptr};
    __gm__ half *cacheOutGm{nullptr};
    __gm__ half *cacheBuff{nullptr};
    AscendC::GlobalTensor<uint32_t> layerIdGm;
    AscendC::GlobalTensor<uint32_t> tokenOffsetGm;
    AscendC::GlobalTensor<uint32_t> seqLenGm;
    AscendC::LocalTensor<uint32_t> tokenOffsetGmInUb;
    AscendC::LocalTensor<uint32_t> seqLenGmInUb;
    uint32_t batch{0};
    uint32_t hiddenSize{0};
    uint32_t maxSeqLen{0};
    uint32_t layerId{0};
    uint32_t prefixOfNtokens{0};
    uint32_t ntokens{0};
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> outQueue;
    uint32_t batchPerCore{0}; // 每核处理batch数
    uint64_t blockIdx{0};
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *pTilingdata, AtbOps::KVCacheTilingData *tilingData)
{
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingDataInUb = nullptr;
    CopyGmTilingToUb(tilingDataInUb, pTilingdata, sizeof(AtbOps::KVCacheTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingData->batch = (*(__ubuf__ uint32_t *)(tilingDataInUb + 0));
    tilingData->maxSeqLen = (*(__ubuf__ uint32_t *)(tilingDataInUb + 4));
    tilingData->hiddenSize = (*(__ubuf__ uint32_t *)(tilingDataInUb + 8));
    tilingData->batchPerCore = (*(__ubuf__ uint32_t *)(tilingDataInUb + 12));
}
#endif