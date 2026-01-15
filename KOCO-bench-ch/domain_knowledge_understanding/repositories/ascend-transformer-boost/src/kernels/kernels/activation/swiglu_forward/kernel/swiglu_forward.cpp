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
#include "kernels/utils/kernel/kernel_utils.h"
#include "swiglu_forward.h"
using namespace AscendC;

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AsdOps::SwiGluForwardTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->rowLen = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->colLen = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->rowLenPerCore = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
    tilingdata->colLenPerCore = (*(const __gm__ uint32_t *)(p_tilingdata + 12));
    tilingdata->coreNumPerLine = (*(const __gm__ uint32_t *)(p_tilingdata + 16));
    tilingdata->basicRowLen = (*(const __gm__ uint16_t *)(p_tilingdata + 20));
    tilingdata->basicColLen = (*(const __gm__ uint16_t *)(p_tilingdata + 22));
#else
    TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AsdOps::SwiGluForwardTilingData), &pipe);
    PipeBarrier<PIPE_ALL>();
    tilingdata->rowLen = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->colLen = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    tilingdata->rowLenPerCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 8));
    tilingdata->colLenPerCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 12));
    tilingdata->coreNumPerLine = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 16));
    tilingdata->basicRowLen = (*(__ubuf__ uint16_t *)(tilingdata_in_ub + 20));
    tilingdata->basicColLen = (*(__ubuf__ uint16_t *)(tilingdata_in_ub + 22));
    PipeBarrier<PIPE_ALL>();
#endif
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                                                                 \
    AsdOps::SwiGluForwardTilingData tiling_data;                                                                       \
    InitTilingData(tiling_arg, &(tiling_data))

extern "C" __global__ __aicore__ void swiglu_forward(GM_ADDR input_gm, GM_ADDR output_gm, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    if (TILING_KEY_IS(1)) { // 1代表fp16 type
        SwigluForward<half, half> op;
        op.Init(input_gm, output_gm, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(0)) { // 0代表float type
        SwigluForward<float, float> op;
        op.Init(input_gm, output_gm, tiling_data);
        op.Process();
    }
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if (TILING_KEY_IS(27)) { // 27代表bf16 type
        SwigluForward<bfloat16_t, bfloat16_t> op;
        op.Init(input_gm, output_gm, tiling_data);
        op.Process();
    }
#endif
}