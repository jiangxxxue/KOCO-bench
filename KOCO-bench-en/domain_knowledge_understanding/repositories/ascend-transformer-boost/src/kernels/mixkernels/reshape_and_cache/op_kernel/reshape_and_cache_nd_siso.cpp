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

constexpr int32_t MAX_UB_USED = MAX_UB_SIZE / BUFFER_NUM;

template <typename T>
class ReshapeAndCacheNdSiso : public ReshapeAndCacheBase {
public:
    __aicore__ inline ReshapeAndCacheNdSiso() {}

    template <bool NCT = false>
    __aicore__ inline void ProcessKey(GM_ADDR keyIn, GM_ADDR keyCacheIn, GM_ADDR slotMapping, GM_ADDR keyCacheOut)
    {
        InitGlobalTensor<T>(keyInputGt_, keyIn);
        InitGlobalTensor<int32_t>(slotInputGt_, slotMapping);
        InitGlobalTensor<T>(keyOutputGt_, keyCacheOut);
        tokenSizeK_ = numHeads_ * headSizeK_;
        PrepareCopy<T>(tokenSizeK_, loopK_, tailK_, MAX_UB_USED, queBindK_);
        AllocateTask();
        for (uint32_t i = 0; i < perCoreTaskNum_; i++) {
            AscendC::GlobalTensor<T> keyInputGtOffset_ = keyInputGt_[offsetK_];
            if constexpr (NCT) {
                CopyToCache<T, true>(i, tokenSizeK_, loopK_, tailK_, keyInputGtOffset_,
                    keyOutputGt_, slotInputGt_, MAX_UB_USED, queBindK_);
            } else {
                CopyToCache<T, false>(i, tokenSizeK_, loopK_, tailK_, keyInputGt_,
                    keyOutputGt_, slotInputGt_, MAX_UB_USED, queBindK_);
            }
        }
    }

    template <bool NCT = false>
    __aicore__ inline void ProcessKeyIncrement(GM_ADDR keyIn, GM_ADDR keyCacheIn,
        GM_ADDR slotMapping, GM_ADDR keyCacheOut)
    {
        InitGlobalTensor<T>(keyInputGt_, keyIn);
        InitGlobalTensor<int32_t>(slotInputGt_, slotMapping);
        InitGlobalTensor<T>(keyOutputGt_, keyCacheOut);
        tokenSizeK_ = numHeads_ * headSizeK_;
        numBlocksK_ = tokenSizeK_ * sizeof(T) / BLOCK_SIZE;
        InitTBuf<T>(tokenBuf_, tokenSizeK_);
        tokenLocal_ = tokenBuf_.Get<T>();
        AllocateTask();
        AscendC::DataCopyParams copyParams = {1, static_cast<uint16_t>(numBlocksK_), 0, 0};
        for (uint32_t i = 0; i < perCoreTaskNum_; i++) {
            uint32_t start = (i + startTaskId_) * tokenSizeK_;
            if constexpr (NCT) {
                start = (i + startTaskId_) * stride_[0] + offsetK_;
            }
            int64_t slotValue = (int64_t)(slotInputGt_.GetValue(i + startTaskId_));
            if (slotValue < 0) continue;
            uint64_t cacheStart = static_cast<uint64_t>(slotValue) * static_cast<uint64_t>(tokenSizeK_);
            CopyKvCache(keyInputGt_, tokenLocal_, keyOutputGt_, start, cacheStart, copyParams, copyParams);
        }
    }

private:
    uint32_t tokenSizeK_ = 0;
    uint32_t numBlocksK_ = 0;
    uint32_t numBlocksRowsK_ = 0;
    uint32_t loopK_ = 0, tailK_ = 0;

    AscendC::LocalTensor<T> tokenLocal_;
    AscendC::GlobalTensor<T> keyInputGt_;
    AscendC::GlobalTensor<T> keyOutputGt_;
    AscendC::GlobalTensor<int32_t> slotInputGt_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tokenBuf_;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, BUFFER_NUM> queBindK_;
};

extern "C" __global__ __aicore__ void reshape_and_cache_siso(
    GM_ADDR keyIn, GM_ADDR keyCacheIn, GM_ADDR slotMapping, GM_ADDR keyCacheOut, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    if (TILING_KEY_IS(140000000)) {
        GET_TILING_DATA(tilingData, tiling);
        ReshapeAndCacheNdSiso<int8_t> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKeyIncrement(keyIn, keyCacheIn, slotMapping, keyCacheOut);
    }
    if (TILING_KEY_IS(142000000)) {
        GET_TILING_DATA(tilingData, tiling);
        ReshapeAndCacheNdSiso<int8_t> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKey(keyIn, keyCacheIn, slotMapping, keyCacheOut);
    }
    if (TILING_KEY_IS(240000000)) {
        GET_TILING_DATA(tilingData, tiling);
        ReshapeAndCacheNdSiso<half> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKeyIncrement(keyIn, keyCacheIn, slotMapping, keyCacheOut);
    }
    if (TILING_KEY_IS(242000000)) {
        GET_TILING_DATA(tilingData, tiling);
        ReshapeAndCacheNdSiso<half> op;
        op.Init(&pipe, &tilingData);
        op.ProcessKey(keyIn, keyCacheIn, slotMapping, keyCacheOut);
    }

    if (TILING_KEY_IS(240200000)) {
        ReshapeAndCacheNctTilingData nctTilingData;
        InitTilingData<ReshapeAndCacheNctTilingData, true>(tiling, &(nctTilingData));
        ReshapeAndCacheNdSiso<half> op;
        op.Init<ReshapeAndCacheNctTilingData, true>(&pipe, &nctTilingData);
        op.ProcessKeyIncrement<true>(keyIn, keyCacheIn, slotMapping, keyCacheOut);
    }
    if (TILING_KEY_IS(242200000)) {
        ReshapeAndCacheNctTilingData nctTilingData;
        InitTilingData<ReshapeAndCacheNctTilingData, true>(tiling, &(nctTilingData));
        ReshapeAndCacheNdSiso<half> op;
        op.Init<ReshapeAndCacheNctTilingData, true>(&pipe, &nctTilingData);
        op.ProcessKey<true>(keyIn, keyCacheIn, slotMapping, keyCacheOut);
    }
}