/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faster_gelu_forward.h"
#include "kernel_operator.h"
#include "kernels/activation/faster_gelu_forward/tiling/tiling_data.h"
#include "kernels/utils/kernel/kernel_utils.h"

using namespace AscendC;
using namespace AsdOps;

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, FasterGeluForwardTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->usedCoreNum = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->maxTileLen = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->alignDataNum = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
    for (uint32_t i = 0; i < tilingdata->usedCoreNum; ++i) {
        tilingdata->singleCoreDataLen[i] = (*(const __gm__ uint32_t *)(p_tilingdata + 12 + i * sizeof(uint32_t)));
    }
#else
    TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AsdOps::FasterGeluForwardTilingData), &pipe);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    tilingdata->usedCoreNum = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->maxTileLen = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    tilingdata->alignDataNum = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 8));
    for (uint32_t i = 0; i < tilingdata->usedCoreNum; ++i) {
        tilingdata->singleCoreDataLen[i] = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 12 + i * sizeof(uint32_t)));
    }
    PipeBarrier<PIPE_ALL>();
#endif
}

#define GET_TILING_DATA(tilingData, tilingArg)                                                                         \
    FasterGeluForwardTilingData tilingData;                                                                            \
    InitTilingData(tilingArg, &(tilingData))

extern "C" __global__ __aicore__ void faster_gelu_forward(GM_ADDR inputAddr, GM_ADDR outputAddr, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        FasterGeluForward<half, half, false> op;
        op.Init(inputAddr, outputAddr, tilingData);
        op.Process<>();
    }
    if (TILING_KEY_IS(0)) {
        FasterGeluForward<float, float, false> op;
        op.Init(inputAddr, outputAddr, tilingData);
        // fp32默认走高精度模式
        op.Process<false>();
    }
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if (TILING_KEY_IS(27)) {
        FasterGeluForward<bfloat16_t, bfloat16_t, false> op;
        op.Init(inputAddr, outputAddr, tilingData);
        op.Process<>();
    }
#endif
}
