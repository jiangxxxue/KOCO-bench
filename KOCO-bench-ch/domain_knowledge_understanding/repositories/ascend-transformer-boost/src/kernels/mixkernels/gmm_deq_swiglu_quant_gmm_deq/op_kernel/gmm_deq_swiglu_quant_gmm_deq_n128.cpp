/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gmm_deq_swiglu_quant_gmm_deq.h"

extern "C" __global__ __aicore__
void gmm_deq_swiglu_quant_gmm_deq_n128(
    GM_ADDR fftsAddr,
    GM_ADDR gmX1,
    GM_ADDR gmPermuteWeight1,
    GM_ADDR gmPermuteScale1,
    GM_ADDR gmPerTokenScale1,
    GM_ADDR gmGroupList,
    GM_ADDR gmWeight2,
    GM_ADDR gmScale2,
    GM_ADDR gmOutput,
    GM_ADDR workspace,
    GM_ADDR tiling
)
{
    icache_preload(1);
    AscendC::SetSyncBaseAddr(reinterpret_cast<uint64_t>(fftsAddr));

    auto tilingData = GetTilingData(tiling);

    uint32_t m = tilingData.m;
    uint32_t n = tilingData.n;
    uint32_t k = tilingData.k;
    uint32_t groupCount = tilingData.groupCount;
    uint32_t nOut = n / 2;

    GemmCoord gmm1ProblemShape{m, n, k};

    layout::RowMajor layoutX1{m, k};
    layout::zN layoutWeight1 = layout::zN::template MakeLayout<int8_t>(k, n);
    layout::VectorLayout layoutScale1{n};
    layout::VectorLayout layoutPerTokenScale1{m};
    layout::RowMajor layoutX2{m, nOut};
    layout::VectorLayout layoutPerTokenScale2{m};

    size_t workspaceOffset = 0;
    GM_ADDR gmX2 = workspace;
    workspaceOffset += RoundUp<GM_ALIGN_BYTE>(static_cast<size_t>(m) * nOut * sizeof(int8_t));
    GM_ADDR gmPerTokenScale2 = workspace + workspaceOffset;
    workspaceOffset += RoundUp<GM_ALIGN_BYTE>(static_cast<size_t>(m) * sizeof(float));
    GM_ADDR gmWorkspace = workspace + workspaceOffset;

    using Gmm1L1TileShape = GemmShape<Gmm1TileArgs<PERMUTE_N128>::L1M, Gmm1TileArgs<PERMUTE_N128>::L1N, GMM1_L1K>;
    using Gmm1L0TileShape = GemmShape<Gmm1L1TileShape::M, Gmm1L1TileShape::N, GMM1_L0K>;
    using Gmm1EpilogueTileShape = MatrixShape<Gmm1TileArgs<PERMUTE_N128>::EPIM, Gmm1L1TileShape::N>;

    GmmDeqSwigluQuant<Gmm1L1TileShape, Gmm1L0TileShape, Gmm1EpilogueTileShape>(
        gmm1ProblemShape, tilingData.groupCount, gmGroupList, gmX1, layoutX1, gmPermuteWeight1, layoutWeight1,
        gmPermuteScale1, layoutScale1, gmPerTokenScale1, layoutPerTokenScale1, gmX2, layoutX2,
        gmPerTokenScale2, layoutPerTokenScale2, gmWorkspace);

    BarrierBetweenUpAndDown();

    uint32_t n2 = k;
    uint32_t k2 = nOut;

    GemmCoord gmm2ProblemShape{m, n2, k2};

    layout::nZ layoutWeight2 = layout::nZ::template MakeLayout<int8_t>(k2, n2);
    layout::VectorLayout layoutScale2{n2};
    layout::RowMajor layoutOutput{m, n2};

    GmmDeq(gmm2ProblemShape, tilingData.groupCount, gmGroupList, gmX2, layoutX2, gmWeight2, layoutWeight2,
        gmScale2, layoutScale2, gmPerTokenScale2, layoutPerTokenScale2, gmOutput, layoutOutput, gmWorkspace);
}

// 控制 C - V 按照 1:1 启动
#if defined(__DAV_C220_CUBE__)
static const struct FunLevelMixCoreType gmm_deq_swiglu_quant_gmm_deq_n128_0_mix_aic_section
__attribute__ ((used, section (".ascend.meta.gmm_deq_swiglu_quant_gmm_deq_n128_0_mix_aic"))) = {
    {
        {
            F_TYPE_KTYPE,
            sizeof(unsigned int)
        },
        K_TYPE_MIX_AIC_MAIN
    },
    {
        {
            F_TYPE_MIX_TASK_RATION,
            sizeof(unsigned int)
        },
        1, 1
    }
};
#endif
#if defined(__DAV_C220_VEC__)
static const struct FunLevelMixCoreType gmm_deq_swiglu_quant_gmm_deq_n128_0_mix_aiv_section
__attribute__ ((used, section (".ascend.meta.gmm_deq_swiglu_quant_gmm_deq_n128_0_mix_aiv"))) = {
    {
        {
            F_TYPE_KTYPE,
            sizeof(unsigned int)
        },
        K_TYPE_MIX_AIC_MAIN
    },
    {
        {
            F_TYPE_MIX_TASK_RATION,
            sizeof(unsigned int)
        },
        1, 1
    }
};
#endif