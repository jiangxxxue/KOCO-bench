/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kvcache_nz.h"

extern "C" __global__ __aicore__ void kvcache_nz_params(GM_ADDR newKV, GM_ADDR layerId, GM_ADDR cacheIn,
                                                        GM_ADDR tokenOffset, GM_ADDR seqLen, GM_ADDR cacheOut,
                                                        GM_ADDR tiling)
{
    AtbOps::KVCacheTilingData tilingData;
    InitTilingData(tiling, &(tilingData));
    KvCacheNz op(tilingData.batch, tilingData.hiddenSize, tilingData.maxSeqLen, tilingData.batchPerCore);
    op.Init(newKV, layerId, cacheIn, tokenOffset, seqLen, cacheOut, tiling);
}