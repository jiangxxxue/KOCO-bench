/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "mixkernels/rope_q_concat/tiling/tiling_data.h"
#include "mixkernels/utils/common/kernel/kernel_utils.h"
#include "RopeFp16.h"

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AtbOps::RopeQConcatTilingData *tilingdata,
                                      AscendC::TPipe *pipe)
{
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AtbOps::RopeQConcatTilingData), pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->hiddenSizeQ = (*(__ubuf__ uint64_t *)(tilingdata_in_ub + 0));
    tilingdata->headNumQ = (*(__ubuf__ uint64_t *)(tilingdata_in_ub + 8));
    tilingdata->headDim = (*(__ubuf__ uint64_t *)(tilingdata_in_ub + 16));
    tilingdata->concatSize = (*(__ubuf__ uint64_t *)(tilingdata_in_ub + 24));
    tilingdata->rotaryCoeff = (*(__ubuf__ uint64_t *)(tilingdata_in_ub + 32));
    tilingdata->ntokens = (*(__ubuf__ uint64_t *)(tilingdata_in_ub + 40));
    tilingdata->realCore = (*(__ubuf__ uint64_t *)(tilingdata_in_ub + 48));
    tilingdata->nlCoreRun = (*(__ubuf__ uint64_t *)(tilingdata_in_ub + 56));
    tilingdata->lCoreRun = (*(__ubuf__ uint64_t *)(tilingdata_in_ub + 64));
    tilingdata->maxNPerLoopForUb = (*(__ubuf__ uint64_t *)(tilingdata_in_ub + 72));
    tilingdata->preCoreLoopTime = (*(__ubuf__ uint64_t *)(tilingdata_in_ub + 80));
    tilingdata->preCoreLoopNLast = (*(__ubuf__ uint64_t *)(tilingdata_in_ub + 88));
    tilingdata->lastCoreLoopTime = (*(__ubuf__ uint64_t *)(tilingdata_in_ub + 96));
    tilingdata->lastCoreLoopNLast = (*(__ubuf__ uint64_t *)(tilingdata_in_ub + 104));

    AscendC::PipeBarrier<PIPE_ALL>();
}
 
#define GET_TILING_DATA(tiling_data, tiling_arg, pipe)    \
    AtbOps::RopeQConcatTilingData tiling_data; \
    InitTilingData(tiling_arg, &(tiling_data), &(pipe))

extern "C" __global__ __aicore__ void rope_q_concat(GM_ADDR q, GM_ADDR cos, GM_ADDR sin, GM_ADDR concatInput,
                                                    GM_ADDR ropeQConcat, GM_ADDR tiling)
{
    PRELOAD(2);
    AscendC::TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling, pipe);
    if (TILING_KEY_IS(20)) {
        RopeFp16<half, half, false> ropeFp16(&tiling_data, &pipe);
        ropeFp16.RopeInitGm(q, cos, sin, concatInput, ropeQConcat);
        ropeFp16.Process();
    }
    if (TILING_KEY_IS(22)) {
        RopeFp16<bfloat16_t, bfloat16_t, false> ropeFp16(&tiling_data, &pipe);
        ropeFp16.RopeInitGm(q, cos, sin, concatInput, ropeQConcat);
        ropeFp16.Process();
    }
}