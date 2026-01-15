/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "kernels/utils/kernel/kernel_utils.h"
#include "kernels/elewise/dynamic_quant/dynamic_quant_tiling/tiling_data.h"

#include "dynamic_quant_align.h"
#include "dynamic_quant_unalign_910b.h"
#include "dynamic_quant_unalign_310p.h"


inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata,
    AsdOps::DynamicQuantTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->numCore         = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->sizeH           = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->sizeX           = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
    tilingdata->sizeZOut        = (*(const __gm__ uint32_t *)(p_tilingdata + 12));
    tilingdata->sizeCopyRow     = (*(const __gm__ uint32_t *)(p_tilingdata + 16));
    tilingdata->numCopyRow      = (*(const __gm__ uint32_t *)(p_tilingdata + 20));
    tilingdata->numHeadCore     = (*(const __gm__ uint32_t *)(p_tilingdata + 24));
    tilingdata->numTailCore     = (*(const __gm__ uint32_t *)(p_tilingdata + 28));
    tilingdata->numHeadTimes    = (*(const __gm__ uint32_t *)(p_tilingdata + 32));
    tilingdata->numTailTimes    = (*(const __gm__ uint32_t *)(p_tilingdata + 36));
    tilingdata->numLastTailRow  = (*(const __gm__ uint32_t *)(p_tilingdata + 40));
    tilingdata->alignType       = (*(const __gm__ uint32_t *)(p_tilingdata + 44));
    tilingdata->asymmetric      = (*(const __gm__ uint32_t *)(p_tilingdata + 48));
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AsdOps::DynamicQuantTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->numCore         = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->sizeH           = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    tilingdata->sizeX           = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 8));
    tilingdata->sizeZOut        = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 12));
    tilingdata->sizeCopyRow     = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 16));
    tilingdata->numCopyRow      = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 20));
    tilingdata->numHeadCore     = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 24));
    tilingdata->numTailCore     = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 28));
    tilingdata->numHeadTimes    = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 32));
    tilingdata->numTailTimes    = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 36));
    tilingdata->numLastTailRow  = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 40));
    tilingdata->alignType       = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 44));
    tilingdata->asymmetric      = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 48));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                                                        \
    AsdOps::DynamicQuantTilingData tiling_data;                                                      \
    InitTilingData(tiling_arg, &(tiling_data))

extern "C" __global__ __aicore__ void dynamic_quant(GM_ADDR x, GM_ADDR z, GM_ADDR scale, GM_ADDR offset,
    GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    if ((TILING_KEY_IS(10100))) {   // FP16, align
        DynamicQuantAlign<half> op;
        op.InitParams(&tilingData);
        op.InitBuffer(x, z, scale, offset);
        op.InitQueue();
        op.Process();
    }
    if ((TILING_KEY_IS(11000))) {   // FP16, 310p unalign
        DynamicQuantUnalign310P op;
        op.DynamicQuantAlign::InitParams(&tilingData);
        op.InitBuffer(x, z, scale, offset);
        op.DynamicQuantAlign::InitQueue();
        op.Process();
    }
    if ((TILING_KEY_IS(10000))) {   // FP16, 910b unalign
        DynamicQuantUnalign910B<half> op;
        op.InitParams(&tilingData);
        op.InitBuffer(x, z, scale, offset);
        op.InitQueue();
        op.Process();
    }
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if ((TILING_KEY_IS(10110))) {   // BF16, align
        DynamicQuantAlign<bfloat16_t> op;
        op.InitParams(&tilingData);
        op.InitBuffer(x, z, scale, offset);
        op.InitQueue();
        op.Process();
    }
    if ((TILING_KEY_IS(10010))) {   // BF16, 910b unalign
        DynamicQuantUnalign910B<bfloat16_t> op;
        op.InitParams(&tilingData);
        op.InitBuffer(x, z, scale, offset);
        op.InitQueue();
        op.Process();
    }
#endif
}