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
#include "kernels/dynamic_ntk/dynamic_ntk/tiling/tiling_data.h"
#include "kernels/utils/kernel/kernel_utils.h"
#include "dynamic_ntk_calc.h"


inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AsdOps::DynamicNTKTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->numTokens = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->headDim = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->batchNum = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
    tilingdata->freqTileLen = (*(const __gm__ uint32_t *)(p_tilingdata + 12));
    tilingdata->freqTileNum = (*(const __gm__ uint32_t *)(p_tilingdata + 16));
    tilingdata->freqTailLen = (*(const __gm__ int32_t *)(p_tilingdata + 20));
    tilingdata->posTileLen = (*(const __gm__ uint32_t *)(p_tilingdata + 24));
    tilingdata->posLongCores = (*(const __gm__ uint32_t *)(p_tilingdata + 28));
    tilingdata->posShortCores = (*(const __gm__ uint32_t *)(p_tilingdata + 32));
    tilingdata->posShortLen = (*(const __gm__ uint32_t *)(p_tilingdata + 36));
    tilingdata->posLongLen = (*(const __gm__ uint32_t *)(p_tilingdata + 40));
    tilingdata->posTailCoreLen = (*(const __gm__ uint32_t *)(p_tilingdata + 44));
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AsdOps::DynamicNTKTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->numTokens = (*(const __ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->headDim = (*(const __ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    tilingdata->batchNum = (*(const __ubuf__ uint32_t *)(tilingdata_in_ub + 8));
    tilingdata->freqTileLen = (*(const __ubuf__ uint32_t *)(tilingdata_in_ub + 12));
    tilingdata->freqTileNum = (*(const __ubuf__ uint32_t *)(tilingdata_in_ub + 16));
    tilingdata->freqTailLen = (*(const __ubuf__ int32_t *)(tilingdata_in_ub + 20));
    tilingdata->posTileLen = (*(const __ubuf__ uint32_t *)(tilingdata_in_ub + 24));
    tilingdata->posLongCores = (*(const __ubuf__ uint32_t *)(tilingdata_in_ub + 28));
    tilingdata->posShortCores = (*(const __ubuf__ uint32_t *)(tilingdata_in_ub + 32));
    tilingdata->posShortLen = (*(const __ubuf__ uint32_t *)(tilingdata_in_ub + 36));
    tilingdata->posLongLen = (*(const __ubuf__ uint32_t *)(tilingdata_in_ub + 40));
    tilingdata->posTailCoreLen = (*(const __ubuf__ uint32_t *)(tilingdata_in_ub + 44));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}


#define GET_TILING_DATA(tiling_data, tiling_arg)    \
    AsdOps::DynamicNTKTilingData tiling_data;      \
    InitTilingData((tiling_arg), &(tiling_data))


extern "C" __global__ __aicore__ void dynamic_ntk(GM_ADDR positionIds,
                                                  GM_ADDR invFreqs,
                                                  GM_ADDR seqLens,
                                                  GM_ADDR sin,
                                                  GM_ADDR cos,
                                                  GM_ADDR workspace,
                                                  GM_ADDR tiling)
{
    GET_TILING_DATA(tilingDataIn, tiling);
    const AsdOps::DynamicNTKTilingData *__restrict tilingData = &tilingDataIn;

    TPipe tPipe;
    if (TILING_KEY_IS(100)) {
        DynamicNTKCalc<half> op;
        op.Init(tPipe, positionIds, invFreqs, seqLens, sin, cos, workspace, tilingData);
        op.Process();
    }  else if (TILING_KEY_IS(120)) {
        DynamicNTKCalc<float> op;
        op.Init(tPipe, positionIds, invFreqs, seqLens, sin, cos, workspace, tilingData);
        op.Process();
    }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    else if (TILING_KEY_IS(110)) {
        DynamicNTKCalc<bfloat16_t> op;
        op.Init(tPipe, positionIds, invFreqs, seqLens, sin, cos, workspace, tilingData);
        op.Process();
    }
#endif
}
