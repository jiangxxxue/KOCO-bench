/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gelu_forward.h"

#include "kernel_operator.h"
#include "kernels/activation/gelu_forward/tiling/tiling_data.h"
#include "kernels/utils/kernel/kernel_utils.h"

using namespace AscendC;
using namespace AsdOps;

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, GeluForwardTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->blockLength = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->tileNum = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->tileLength = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
    tilingdata->tailLength = (*(const __gm__ uint32_t *)(p_tilingdata + 12));
    tilingdata->bufferNum = (*(const __gm__ uint32_t *)(p_tilingdata + 16));
#else
    TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AsdOps::GeluForwardTilingData), &pipe);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    tilingdata->blockLength = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->tileNum = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    tilingdata->tileLength = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 8));
    tilingdata->tailLength = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 12));
    tilingdata->bufferNum = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 16));
    PipeBarrier<PIPE_ALL>();
#endif
}

extern "C" __global__ __aicore__ void gelu_forward(GM_ADDR inputAddr, GM_ADDR outputAddr, GM_ADDR tiling)
{
    GeluForwardTilingData tilingData;
    InitTilingData(tiling, &(tilingData));
    if (TILING_KEY_IS(1)) { // 1代表fp16 type
        GeluForward<half, half> op;
        op.Init(inputAddr, outputAddr, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(0)) { // 0代表float type
        GeluForward<float, float> op;
        op.Init(inputAddr, outputAddr, tilingData);
        op.ProcessFP32();
    }
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220) // 220代表910B
    if (TILING_KEY_IS(27)) { // 27代表bf16
        GeluForward<bfloat16_t, bfloat16_t> op;
        op.Init(inputAddr, outputAddr, tilingData);
        op.Process();
    }
#endif
}
