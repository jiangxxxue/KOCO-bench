/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef NORM_COMMMON_INIT_TILING_H
#define NORM_COMMMON_INIT_TILING_H

#include "kernel_operator.h"
#include "kernels/utils/kernel/kernel_utils.h"
#include "kernels/norm/postrmsnorm/tiling/post_rms_norm_tiling_data.h"

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AsdOps::PostRmsNormTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->numCore = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->numCol = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->numRow = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
    tilingdata->avgFactor = (*(const __gm__ uint32_t *)(p_tilingdata + 12));
    tilingdata->epsilon = (*(const __gm__ uint32_t *)(p_tilingdata + 16));
    tilingdata->sliceSize = (*(const __gm__ uint32_t *)(p_tilingdata + 20));
    tilingdata->mode = (*(const __gm__ uint32_t *)(p_tilingdata + 24));
    tilingdata->supportFast = (*(const __gm__ uint32_t *)(p_tilingdata + 28));
    tilingdata->supportFastLast = (*(const __gm__ uint32_t *)(p_tilingdata + 32));
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AsdOps::PostRmsNormTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->numCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->numCol = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    tilingdata->numRow = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 8));
    tilingdata->avgFactor = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 12));
    tilingdata->epsilon = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 16));
    tilingdata->sliceSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 20));
    tilingdata->mode = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 24));
    tilingdata->supportFast = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 28));
    tilingdata->supportFastLast = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 32));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                                                                       \
    AsdOps::PostRmsNormTilingData tiling_data;                                                                         \
    InitTilingData(tiling_arg, &(tiling_data))

#endif