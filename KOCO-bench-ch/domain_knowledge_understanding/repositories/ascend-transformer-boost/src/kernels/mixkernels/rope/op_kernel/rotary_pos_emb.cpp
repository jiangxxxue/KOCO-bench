/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "mixkernels/rope/tiling/tiling_data.h"
#include "mixkernels/utils/common/kernel/kernel_utils.h"
#include "rotary_pos_emb_fp16.h"
#include "rotary_pos_emb_fp16_large_ntokens.h"
#include "rotary_pos_emb_fp32.h"
#include "rotary_pos_emb_bf16.h"
#include "rotary_pos_emb_bf16_align.h"

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AtbOps::RopeTilingData *tilingdata,
                                      AscendC::TPipe *pipe)
{
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AtbOps::RopeTilingData), pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->hiddenSizeQ = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->hiddenSizeK = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    tilingdata->headDim = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 8));
    tilingdata->headNumQ = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 12));
    tilingdata->headNumK = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 16));
    tilingdata->rotaryCoeff = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 20));
    tilingdata->ntokens = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 24));
    tilingdata->realCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 28));
    tilingdata->cosFormat = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 32));
    tilingdata->batch = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 36));
    tilingdata->maxUbSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 40));
    tilingdata->multiple = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 44));
    AscendC::PipeBarrier<PIPE_ALL>();
}

#define GET_TILING_DATA(tiling_data, tiling_arg, pipe)                                                                 \
    AtbOps::RopeTilingData tiling_data;                                                                                \
    InitTilingData(tiling_arg, &(tiling_data), &(pipe))

extern "C" __global__ __aicore__ void rotary_pos_emb(GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin,
                                                     GM_ADDR seqLen, GM_ADDR outQ, GM_ADDR outK,
                                                     GM_ADDR workspace, GM_ADDR sync, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling, pipe);
    if (TILING_KEY_IS(30)) {
        RopeFp16<half, half, true> ropeFp16(&tiling_data, &pipe);
        ropeFp16.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeFp16.Process(workspace, sync);
    } else if (TILING_KEY_IS(31)) {
        RopeFp32<half, float, true> ropeFp32(&tiling_data, &pipe);
        ropeFp32.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeFp32.Process(workspace);
    } else if (TILING_KEY_IS(32)) {
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        RopeBf16<bfloat16_t, bfloat16_t, true> ropeBf16(&tiling_data, &pipe);
        ropeBf16.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeBf16.Process(workspace);
#endif
    } else if (TILING_KEY_IS(33)) {
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        RopeBf16Align<bfloat16_t, bfloat16_t, true> ropeBf16(&tiling_data, &pipe);
        ropeBf16.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeBf16.Process(workspace);
#endif
    } else if (TILING_KEY_IS(20)) {
        RopeFp16<half, half, false> ropeFp16(&tiling_data, &pipe);
        ropeFp16.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeFp16.Process(workspace, sync);
    } else if (TILING_KEY_IS(21)) {
        RopeFp32<half, float, false> ropeFp32(&tiling_data, &pipe);
        ropeFp32.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeFp32.Process(workspace);
    } else if (TILING_KEY_IS(22)) {
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        RopeBf16<bfloat16_t, bfloat16_t, false> ropeBf16(&tiling_data, &pipe);
        ropeBf16.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeBf16.Process(workspace);
#endif
    } else if (TILING_KEY_IS(24)) {
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        RopeBf16Align<bfloat16_t, bfloat16_t, false> ropeBf16(&tiling_data, &pipe);
        ropeBf16.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeBf16.Process(workspace);
#endif
    } else if (TILING_KEY_IS(23)) {
        RopeFp16LargeNtokens<half, half, false> ropeFp16LargeNtokens(&tiling_data, &pipe);
        ropeFp16LargeNtokens.RopeInitGm(q, k, cos, sin, seqLen, outQ, outK);
        ropeFp16LargeNtokens.Process(workspace, sync);
    }
}
