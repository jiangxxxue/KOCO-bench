/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rms_norm.h"
#include "rms_norm_split_d.h"
#include "kernel_operator.h"
#include "kernels/norm/rmsnormforward/tiling/tiling_data.h"

using namespace AscendC;
using namespace AsdOps;


inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata,
                                      RmsNormForwardTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->numRow = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->numCol = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->blockFactor = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
    tilingdata->rowFactor = (*(const __gm__ uint32_t *)(p_tilingdata + 12));
    tilingdata->ubFactor = (*(const __gm__ uint32_t *)(p_tilingdata + 16));
    tilingdata->epsilon = (*(const __gm__ float *)(p_tilingdata + 20));
    tilingdata->avgFactor = (*(const __gm__ float *)(p_tilingdata + 24));

#else
    __ubuf__ uint8_t *tilingdata_in_ub = (__ubuf__ uint8_t *)get_imm(0);
    int32_t tilingBlockNum = sizeof(RmsNormForwardTilingData) / 32 + 1;
    copy_gm_to_ubuf(((__ubuf__ uint8_t *)tilingdata_in_ub), p_tilingdata, 0, 1, tilingBlockNum, 0, 0);
    pipe_barrier(PIPE_ALL);
    tilingdata->numRow = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 0));
    tilingdata->numCol = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 4));
    tilingdata->blockFactor = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 8));
    tilingdata->rowFactor = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 12));
    tilingdata->ubFactor = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 16));
    tilingdata->epsilon = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 20));
    tilingdata->avgFactor = (*(__ubuf__ float *)((__ubuf__ uint8_t *)tilingdata_in_ub + 24));
    pipe_barrier(PIPE_ALL);
#endif
}

#define GET_TILING_DATA(tiling_data, tiling_arg)           \
    AsdOps::RmsNormForwardTilingData tiling_data;          \
    InitTilingData(tiling_arg, &(tiling_data))

extern "C" __global__ __aicore__ void rms_norm_forward(GM_ADDR x, GM_ADDR gamma,
                                                       GM_ADDR y,
                                                       GM_ADDR rstd, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(10)) { // 进入half分支
        KernelRmsNorm<half> op;
        op.Init(x, gamma, y, rstd, tilingData.numRow, tilingData.numCol, tilingData.blockFactor,
                tilingData.rowFactor, tilingData.ubFactor, tilingData.epsilon, tilingData.avgFactor);
        op.Process();
    } else if (TILING_KEY_IS(20)) { // 进入float分支
        KernelRmsNorm<float> op;
        op.Init(x, gamma, y, rstd, tilingData.numRow, tilingData.numCol, tilingData.blockFactor,
                tilingData.rowFactor, tilingData.ubFactor, tilingData.epsilon, tilingData.avgFactor);
        op.Process();
    } else if (TILING_KEY_IS(30)) { // 进入bfloat16分支
        KernelRmsNorm<bfloat16_t> op;
        op.Init(x, gamma, y, rstd, tilingData.numRow, tilingData.numCol, tilingData.blockFactor,
                tilingData.rowFactor, tilingData.ubFactor, tilingData.epsilon, tilingData.avgFactor);
        op.Process();
    } else if (TILING_KEY_IS(11)) { // 进入half_SplitD分支
        KernelRmsNormSplitD<half> op;
        op.Init(x, gamma, y, rstd, tilingData.numRow, tilingData.numCol, tilingData.blockFactor,
                tilingData.rowFactor, tilingData.ubFactor, tilingData.epsilon, tilingData.avgFactor);
        op.Process();
    } else if (TILING_KEY_IS(21)) { // 进入float_SplitD分支
        KernelRmsNormSplitD<float> op;
        op.Init(x, gamma, y, rstd, tilingData.numRow, tilingData.numCol, tilingData.blockFactor,
                tilingData.rowFactor, tilingData.ubFactor, tilingData.epsilon, tilingData.avgFactor);
        op.Process();
    } else if (TILING_KEY_IS(31)) { // 进入bfloat16_SplitD分支
        KernelRmsNormSplitD<bfloat16_t> op;
        op.Init(x, gamma, y, rstd, tilingData.numRow, tilingData.numCol, tilingData.blockFactor,
                tilingData.rowFactor, tilingData.ubFactor, tilingData.epsilon, tilingData.avgFactor);
        op.Process();
    }
}