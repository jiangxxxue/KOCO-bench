/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "rms_norm_grad_split_d.h"
#include "rms_norm_grad_split_n.h"

using namespace AscendC;
using namespace AsdOps;

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata,
                                      RmsNormGradTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->row = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->col = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->avg = (*(const __gm__ float *)(p_tilingdata + 8));
    tilingdata->dataType = (*(const __gm__ uint32_t *)(p_tilingdata + 12));
    tilingdata->blockFactor = (*(const __gm__ uint32_t *)(p_tilingdata + 16));
    tilingdata->ubSplitDim = (*(const __gm__ int32_t *)(p_tilingdata + 20));
    tilingdata->ubFactor = (*(const __gm__ int32_t *)(p_tilingdata + 24));
    tilingdata->coreCalcNum = (*(const __gm__ int32_t *)(p_tilingdata + 28));
    tilingdata->coreCalcTail = (*(const __gm__ int32_t *)(p_tilingdata + 32));
    tilingdata->blockDim = (*(const __gm__ int32_t *)(p_tilingdata + 36));
    tilingdata->ubCalcNum = (*(const __gm__ int32_t *)(p_tilingdata + 40));
    tilingdata->ubCalcTail = (*(const __gm__ int32_t *)(p_tilingdata + 44));
    tilingdata->ubCalcLoop = (*(const __gm__ int32_t *)(p_tilingdata + 48));
    tilingdata->ubCalcTailNum = (*(const __gm__ int32_t *)(p_tilingdata + 52));
    tilingdata->ubCalcTailTail = (*(const __gm__ int32_t *)(p_tilingdata + 56));
    tilingdata->ubCalcTailLoop = (*(const __gm__ int32_t *)(p_tilingdata + 60));
#else
    __ubuf__ uint8_t *tilingdata_in_ub = (__ubuf__ uint8_t *)get_imm(0);
    int32_t tilingBlockNum = sizeof(RmsNormGradTilingData) / 32 + 1;
    copy_gm_to_ubuf(((__ubuf__ uint8_t *)tilingdata_in_ub), p_tilingdata, 0, 1, tilingBlockNum, 0, 0);
    pipe_barrier(PIPE_ALL);
    tilingdata->row = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 0));
    // and so on...
    pipe_barrier(PIPE_ALL);
#endif
}

#define GET_TILING_DATA(tiling_data, tiling_arg)        \
    AsdOps::RmsNormGradTilingData tiling_data;          \
    InitTilingData(tiling_arg, &(tiling_data))

extern "C" __global__ __aicore__ void rms_norm_backward(GM_ADDR dy, GM_ADDR x,
                                                        GM_ADDR rstd, GM_ADDR gamma,
                                                        GM_ADDR dx,
                                                        GM_ADDR dgamma, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(121)) { // 进入FLOAT_SPLITN 分支
        RmsNormGradSplitN<float> rms_norm_grad_split_n;
        rms_norm_grad_split_n.Init(dy, x, rstd, gamma, dx, dgamma, workspace, &tilingData);
        rms_norm_grad_split_n.Process();
    }
    if (TILING_KEY_IS(122)) { // 进入FLOAT_SPLITD 分支
        RmsNormGradSplitD<float> rms_norm_grad_split_d;
        rms_norm_grad_split_d.Init(dy, x, rstd, gamma, dx, dgamma, workspace, &tilingData);
        rms_norm_grad_split_d.Process();
    }
    if (TILING_KEY_IS(221)) { // 进入HALF_SPLITN 分支
        RmsNormGradSplitN<half> rms_norm_grad_split_n;
        rms_norm_grad_split_n.Init(dy, x, rstd, gamma, dx, dgamma, workspace, &tilingData);
        rms_norm_grad_split_n.ProcessFp16();
    }
    if (TILING_KEY_IS(222)) { // 进入HALF_SPLITD 分支
        RmsNormGradSplitD<half> rms_norm_grad_split_d;
        rms_norm_grad_split_d.Init(dy, x, rstd, gamma, dx, dgamma, workspace, &tilingData);
        rms_norm_grad_split_d.ProcessFp16();
    }
    if (TILING_KEY_IS(321)) { // 进入BFLOAT16_SPLITD 分支
        RmsNormGradSplitN<bfloat16_t> rms_norm_grad_split_n;
        rms_norm_grad_split_n.Init(dy, x, rstd, gamma, dx, dgamma, workspace, &tilingData);
        rms_norm_grad_split_n.ProcessBf16();
    }
    if (TILING_KEY_IS(322)) { // 进入BFLOAT16_SPLITD 分支
        RmsNormGradSplitD<bfloat16_t> rms_norm_grad_split_d;
        rms_norm_grad_split_d.Init(dy, x, rstd, gamma, dx, dgamma, workspace, &tilingData);
        rms_norm_grad_split_d.ProcessBf16();
    }
}
