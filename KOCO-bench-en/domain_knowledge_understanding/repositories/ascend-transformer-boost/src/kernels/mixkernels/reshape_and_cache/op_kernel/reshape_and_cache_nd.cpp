/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "reshape_and_cache_base.h"
#include "mixkernels/utils/common/kernel/kernel_utils.h"

constexpr int32_t MAX_UB_USED = MAX_UB_SIZE / (BUFFER_NUM * 2);

template <typename T>
class ReshapeAndCacheNd : public ReshapeAndCacheBase {
public:
    __aicore__ inline ReshapeAndCacheNd() {}

    template <bool NCT = false>
    __aicore__ inline void ProcessKVUnequal(GM_ADDR keyIn, GM_ADDR valueIn, GM_ADDR keyCacheIn, GM_ADDR valueCacheIn,
        GM_ADDR slotMapping, GM_ADDR keyCacheOut, GM_ADDR valueCacheOut)
    {
        InitGlobalTensor<T>(keyInputGt_, keyIn);
        InitGlobalTensor<T>(valueInputGt_, valueIn);
        InitGlobalTensor<int32_t>(slotInputGt_, slotMapping);
        InitGlobalTensor<T>(keyOutputGt_, keyCacheOut);
        InitGlobalTensor<T>(valueOutputGt_, valueCacheOut);

        tokenSizeK_ = numHeads_ * headSizeK_;
        tokenSizeV_ = numHeads_ * headSizeV_;
        PrepareCopy<T>(tokenSizeK_, loopK_, tailK_, MAX_UB_USED, queBindK_);
        PrepareCopy<T>(tokenSizeV_, loopV_, tailV_, MAX_UB_USED, queBindV_);
        AllocateTask();
        for (uint32_t i = 0; i < perCoreTaskNum_; i++) {
            AscendC::GlobalTensor<T> keyInputGtOffset_ = keyInputGt_[offsetK_];
            AscendC::GlobalTensor<T> valueInputGtOffset_ = valueInputGt_[offsetV_];
            if constexpr (NCT) {
                CopyToCache<T, true>(i, tokenSizeK_, loopK_, tailK_, keyInputGtOffset_,
                    keyOutputGt_, slotInputGt_, MAX_UB_USED, queBindK_);
                CopyToCache<T, true>(i, tokenSizeV_, loopV_, tailV_, valueInputGtOffset_,
                    valueOutputGt_, slotInputGt_, MAX_UB_USED, queBindV_);
            } else {
                CopyToCache<T>(i, tokenSizeK_, loopK_, tailK_, keyInputGt_,
                    keyOutputGt_, slotInputGt_, MAX_UB_USED, queBindK_);
                CopyToCache<T>(i, tokenSizeV_, loopV_, tailV_, valueInputGt_,
                    valueOutputGt_, slotInputGt_, MAX_UB_USED, queBindV_);
            }
        }
    }

    template <bool NCT = false>
    __aicore__ inline void ProcessKVUnequalIncrement(GM_ADDR keyIn, GM_ADDR valueIn, GM_ADDR keyCacheIn,
        GM_ADDR valueCacheIn, GM_ADDR slotMapping, GM_ADDR keyCacheOut, GM_ADDR valueCacheOut)
    {
        InitGlobalTensor<T>(keyInputGt_, keyIn);
        InitGlobalTensor<T>(valueInputGt_, valueIn);
        InitGlobalTensor<int32_t>(slotInputGt_, slotMapping);
        InitGlobalTensor<T>(keyOutputGt_, keyCacheOut);
        InitGlobalTensor<T>(valueOutputGt_, valueCacheOut);

        tokenSizeK_ = numHeads_ * headSizeK_;
        tokenSizeV_ = numHeads_ * headSizeV_;
        numBlocksK_ = tokenSizeK_ * sizeof(T) / BLOCK_SIZE;
        numBlocksV_ = tokenSizeV_ * sizeof(T) / BLOCK_SIZE;
        numBlocksRowsK_ = headSizeK_ * sizeof(T) / BLOCK_SIZE;
        numBlocksRowsV_ = headSizeV_ * sizeof(T) / BLOCK_SIZE;
        uint32_t tokenSize = tokenSizeK_ > tokenSizeV_ ? tokenSizeK_ : tokenSizeV_;
        InitTBuf<T>(tokenBuf_, tokenSize);
        tokenLocal_ = tokenBuf_.Get<T>();
        AllocateTask();
        AscendC::DataCopyParams copyParamsK = {1, static_cast<uint16_t>(numBlocksK_), 0, 0};
        AscendC::DataCopyParams copyParamsV = {1, static_cast<uint16_t>(numBlocksV_), 0, 0};
        for (uint32_t i = 0; i < perCoreTaskNum_; i++) {
            uint32_t startK = (i + startTaskId_) * tokenSizeK_;
            uint32_t startV = (i + startTaskId_) * tokenSizeV_;
            if constexpr (NCT) {
                startK = (i + startTaskId_) * stride_[0] + offsetK_;
                startV = (i + startTaskId_) * stride_[0] + offsetV_;
            }
            int64_t slotValue = (int64_t)(slotInputGt_.GetValue(i + startTaskId_));
            if (slotValue < 0) continue;
            uint64_t cacheStartK = static_cast<uint64_t>(slotValue) * static_cast<uint64_t>(tokenSizeK_);
            uint64_t cacheStartV = static_cast<uint64_t>(slotValue) * static_cast<uint64_t>(tokenSizeV_);
            CopyKvCache(keyInputGt_, tokenLocal_, keyOutputGt_, startK, cacheStartK, copyParamsK, copyParamsK);
            CopyKvCache(valueInputGt_, tokenLocal_, valueOutputGt_, startV, cacheStartV, copyParamsV, copyParamsV);
        }
    }

    template <bool NCT = false>
    __aicore__ inline void ProcessKVEqual(GM_ADDR keyIn, GM_ADDR valueIn, GM_ADDR keyCacheIn, GM_ADDR valueCacheIn,
        GM_ADDR slotMapping, GM_ADDR keyCacheOut, GM_ADDR valueCacheOut)
    {
        InitGlobalTensor<T>(keyInputGt_, keyIn);
        InitGlobalTensor<T>(valueInputGt_, valueIn);
        InitGlobalTensor<int32_t>(slotInputGt_, slotMapping);
        InitGlobalTensor<T>(keyOutputGt_, keyCacheOut);
        InitGlobalTensor<T>(valueOutputGt_, valueCacheOut);

        tokenSizeK_ = numHeads_ * headSizeK_;
        numBlocksK_ = tokenSizeK_ * sizeof(T) / BLOCK_SIZE;
        numBlocksRowsK_ = headSizeK_ * sizeof(T) / BLOCK_SIZE;
        numBlocksRowsV_ = headSizeV_ * sizeof(T) / BLOCK_SIZE;
        InitTBuf<T>(tokenBuf_, tokenSizeK_);
        tokenLocal_ = tokenBuf_.Get<T>();

        AllocateTask();
        AscendC::DataCopyParams copyParams = {1, static_cast<uint16_t>(numBlocksK_), 0, 0};
        for (uint32_t i = 0; i < perCoreTaskNum_; i++) {
            uint32_t start = (i + startTaskId_) * tokenSizeK_;
            uint32_t startK = (i + startTaskId_) * stride_[0] + offsetK_;
            uint32_t startV = (i + startTaskId_) * stride_[0] + offsetV_;
            int64_t slotValue = (int64_t)(slotInputGt_.GetValue(i + startTaskId_));
            if (slotValue < 0) continue;
            uint64_t cacheStart = static_cast<uint64_t>(slotValue) * static_cast<uint64_t>(tokenSizeK_);
            if constexpr (NCT) {
                CopyKvCache(keyInputGt_, tokenLocal_, keyOutputGt_, startK, cacheStart, copyParams, copyParams);
                CopyKvCache(valueInputGt_, tokenLocal_, valueOutputGt_, startV, cacheStart, copyParams, copyParams);
            } else {
                CopyKvCache(keyInputGt_, tokenLocal_, keyOutputGt_, start, cacheStart, copyParams, copyParams);
                CopyKvCache(valueInputGt_, tokenLocal_, valueOutputGt_, start, cacheStart, copyParams, copyParams);
            }
        }
    }

private:
    uint32_t tokenSizeK_ = 0;
    uint32_t tokenSizeV_ = 0;
    uint32_t numBlocksK_ = 0;
    uint32_t numBlocksV_ = 0;
    uint32_t numBlocksRowsK_ = 0;
    uint32_t numBlocksRowsV_ = 0;
    uint32_t loopK_ = 0, tailK_ = 0;
    uint32_t loopV_ = 0, tailV_ = 0;

    AscendC::LocalTensor<T> tokenLocal_;
    AscendC::GlobalTensor<T> keyInputGt_;
    AscendC::GlobalTensor<T> keyOutputGt_;
    AscendC::GlobalTensor<T> valueInputGt_;
    AscendC::GlobalTensor<T> valueOutputGt_;
    AscendC::GlobalTensor<int32_t> slotInputGt_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tokenBuf_;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, BUFFER_NUM> queBindK_, queBindV_;
};

extern "C" __global__ __aicore__ void reshape_and_cache(
    GM_ADDR keyIn, GM_ADDR valueIn, GM_ADDR keyCacheIn, GM_ADDR valueCacheIn,
    GM_ADDR slotMapping, GM_ADDR keyCacheOut, GM_ADDR valueCacheOut, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    if (TILING_KEY_IS(100000000)) {
        GET_TILING_DATA(tilingData, tiling);
        ReshapeAndCacheNd<int8_t> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKVEqual(keyIn, valueIn, keyCacheIn, valueCacheIn, slotMapping, keyCacheOut, valueCacheOut);
    }
    if (TILING_KEY_IS(101000000)) {
        GET_TILING_DATA(tilingData, tiling);
        ReshapeAndCacheNd<int8_t> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKVUnequalIncrement(keyIn, valueIn, keyCacheIn, valueCacheIn, slotMapping, keyCacheOut, valueCacheOut);
    }
    if (TILING_KEY_IS(102000000)) {
        GET_TILING_DATA(tilingData, tiling);
        ReshapeAndCacheNd<int8_t> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKVUnequal(keyIn, valueIn, keyCacheIn, valueCacheIn, slotMapping, keyCacheOut, valueCacheOut);
    }
    if (TILING_KEY_IS(200000000)) {
        GET_TILING_DATA(tilingData, tiling);
        ReshapeAndCacheNd<half> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKVEqual(keyIn, valueIn, keyCacheIn, valueCacheIn, slotMapping, keyCacheOut, valueCacheOut);
    }
    if (TILING_KEY_IS(201000000)) {
        GET_TILING_DATA(tilingData, tiling);
        ReshapeAndCacheNd<half> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKVUnequalIncrement(keyIn, valueIn, keyCacheIn, valueCacheIn, slotMapping, keyCacheOut, valueCacheOut);
    }
    if (TILING_KEY_IS(202000000)) {
        GET_TILING_DATA(tilingData, tiling);
        ReshapeAndCacheNd<half> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKVUnequal(keyIn, valueIn, keyCacheIn, valueCacheIn, slotMapping, keyCacheOut, valueCacheOut);
    }

    if (TILING_KEY_IS(200200000)) {
        ReshapeAndCacheNctTilingData nctTilingData;
        InitTilingData<ReshapeAndCacheNctTilingData, true>(tiling, &(nctTilingData));
        ReshapeAndCacheNd<half> op;
        op.Init<ReshapeAndCacheNctTilingData, true>(&pipe, &nctTilingData);
        op.ProcessKVEqual<true>(keyIn, valueIn, keyCacheIn, valueCacheIn, slotMapping, keyCacheOut, valueCacheOut);
    }
    if (TILING_KEY_IS(201200000)) {
        ReshapeAndCacheNctTilingData nctTilingData;
        InitTilingData<ReshapeAndCacheNctTilingData, true>(tiling, &(nctTilingData));
        ReshapeAndCacheNd<half> op;
        op.Init<ReshapeAndCacheNctTilingData, true>(&pipe, &nctTilingData);
        op.ProcessKVUnequalIncrement<true>(keyIn, valueIn, keyCacheIn, valueCacheIn,
                                           slotMapping, keyCacheOut, valueCacheOut);
    }
    if (TILING_KEY_IS(202200000)) {
        ReshapeAndCacheNctTilingData nctTilingData;
        InitTilingData<ReshapeAndCacheNctTilingData, true>(tiling, &(nctTilingData));
        ReshapeAndCacheNd<half> op;
        op.Init<ReshapeAndCacheNctTilingData, true>(&pipe, &nctTilingData);
        op.ProcessKVUnequal<true>(keyIn, valueIn, keyCacheIn, valueCacheIn, slotMapping, keyCacheOut, valueCacheOut);
    }
}