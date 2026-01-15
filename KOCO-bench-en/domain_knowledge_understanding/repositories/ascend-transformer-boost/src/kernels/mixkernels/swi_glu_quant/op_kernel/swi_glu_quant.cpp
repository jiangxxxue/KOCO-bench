/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "mixkernels/swi_glu_quant/tiling/tiling_data.h"
#include "mixkernels/utils/common/kernel/kernel_utils.h"
#include "swi_glu_quant.h"
using namespace AscendC;
using namespace AtbOps;

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AtbOps::SwiGluQuantTilingData *tilingdata,
                                      AscendC::TPipe *pipe)
{
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AtbOps::SwiGluQuantTilingData), pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->groupLen = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->rowLen = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    tilingdata->colLen = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 8));
    tilingdata->rowLenPerHeadCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 12));
    tilingdata->rowLenPerTailCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 16));
    tilingdata->basicRowLenHeadCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 20));
    tilingdata->basicRowLenTailCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 24));
    tilingdata->basicColLen = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 28));
    tilingdata->headCoreNum = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 32));
    tilingdata->realCoreNum = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 36));
    AscendC::PipeBarrier<PIPE_ALL>();
}

#define GET_TILING_DATA(tiling_data, tiling_arg, pipe)                                                                 \
    AtbOps::SwiGluQuantTilingData tiling_data;                                                                         \
    InitTilingData(tiling_arg, &(tiling_data), &(pipe))

extern "C" __global__ __aicore__ void swi_glu_quant(GM_ADDR x, GM_ADDR y, GM_ADDR scale, GM_ADDR tiling)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    PRELOAD(3);
#endif
    AscendC::TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling, pipe);
    if (TILING_KEY_IS(106)) {
        SwiGluQuant<half, int8_t> op(&pipe);
        op.Init(x, y, scale, &tiling_data);
        op.Process();
    }
    if (TILING_KEY_IS(206)) {
        SwiGluQuant<bfloat16_t, int8_t> op(&pipe);
        op.Init(x, y, scale, &tiling_data);
        op.Process();
    }
}