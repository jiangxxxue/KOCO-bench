/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "fused_add_topk_div.h"
#include "mixkernels/fused_add_topk_div/tiling/tiling_data.h"
#include "mixkernels/utils/common/kernel/kernel_utils.h"

using namespace AscendC;

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AtbOps::FusedAddTopkDivTilingData *tilingdata,
                                      AscendC::TPipe *pipe)
{
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AtbOps::FusedAddTopkDivTilingData), pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->firstDimSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->secondDimSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    tilingdata->addNumDimSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 8));
    tilingdata->groupNum = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 12));
    tilingdata->groupTopk = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 16));
    tilingdata->n = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 20));
    tilingdata->k = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 24));
    tilingdata->activateType = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 28));
    tilingdata->isNorm = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 32));
    tilingdata->scale = (*(__ubuf__ float *)(tilingdata_in_ub + 36));
    tilingdata->groupEles = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 40));
    tilingdata->blockNum = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 44));
    tilingdata->ubFactorElement = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 48));
    tilingdata->batchPerCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 52));
    tilingdata->tailBatch = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 56));
    tilingdata->tilingKey = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 60));
    tilingdata->dtype = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 64));
    tilingdata->tempSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 68));
    tilingdata->enableExpertMapping = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 72));
    tilingdata->expertNum = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 76));
    tilingdata->tableDim = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 80));
    tilingdata->workspacePerCore = (*(__ubuf__ uint64_t *)(tilingdata_in_ub + 88));
    AscendC::PipeBarrier<PIPE_ALL>();
}

#define GET_TILING_DATA(tilingData, tiling_arg, pipe)                                                      \
    AtbOps::FusedAddTopkDivTilingData tilingData;                                                          \
    InitTilingData(tiling_arg, &(tilingData), &(pipe))

extern "C" __global__ __aicore__ void fused_add_topk_div(GM_ADDR x, GM_ADDR addNum,
                                                         GM_ADDR mappingNum, GM_ADDR mappingTable, GM_ADDR y,
                                                         GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling)
{
    PRELOAD(8);
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr) {
        return;
    }
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling, pipe);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(0)) {
        FusedAddTopkDiv<float, float, 0> op;
        op.InitTilingData(tilingData, x, addNum, mappingNum, mappingTable, y, indices, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        FusedAddTopkDiv<half, float, 0> op;
        op.InitTilingData(tilingData, x, addNum, mappingNum, mappingTable, y, indices, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        FusedAddTopkDiv<bfloat16_t, float, 0> op;
        op.InitTilingData(tilingData, x, addNum, mappingNum, mappingTable, y, indices, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    } else if (TILING_KEY_IS(10)) {
        FusedAddTopkDiv<float, float, 1> op;
        op.InitTilingData(tilingData, x, addNum, mappingNum, mappingTable, y, indices, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    } else if (TILING_KEY_IS(11)) {
        FusedAddTopkDiv<half, float, 1> op;
        op.InitTilingData(tilingData, x, addNum, mappingNum, mappingTable, y, indices, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    } else if (TILING_KEY_IS(12)) {
        FusedAddTopkDiv<bfloat16_t, float, 1> op;
        op.InitTilingData(tilingData, x, addNum, mappingNum, mappingTable, y, indices, workspace);
        op.InitBuffer(&pipe);
        op.Process();
    }
}