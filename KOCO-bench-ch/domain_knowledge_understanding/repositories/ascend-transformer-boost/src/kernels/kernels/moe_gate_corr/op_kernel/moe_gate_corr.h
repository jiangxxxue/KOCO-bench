/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_GATE_CORR_H
#define MOE_GATE_CORR_H

#include <kernel_operator.h>
#undef inline

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "kernels/moe_gate_corr/op_kernel/moe_gate_corr_kernel.h"

#define inline __inline__ __attribute__((always_inline))

#include "kernels/moe_gate_corr/tiling/tiling_data.h"

using namespace AsdOps;

template <
    class L1TileShape_,
    class L0TileShape_,
    class LayoutA,
    class LayoutB,
    class LayoutC,
    class ElementInput,
    class ElementOutput
>
inline __aicore__ void MoeGateCorr(
    GemmCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmC, LayoutC layoutC
)
{
    constexpr uint32_t preloadStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;
    constexpr bool enableUnitFlag = true;
    constexpr bool enableShuffleK = true;
    using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsyncWithCallback<
        preloadStages,
        l1Stages, l0AStages, l0BStages, l0CStages,
        enableUnitFlag, enableShuffleK
    >;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;

    using AType = Gemm::GemmType<ElementInput, LayoutA>;
    using BType = Gemm::GemmType<ElementInput, LayoutB>;
    using CType = Gemm::GemmType<ElementOutput, LayoutC>;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockEpilogue = void;

    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<4, 1>;

    // kernel level
    using MoeGateCorrKernel = MoeGateCorrKernel<BlockMmad, BlockEpilogue, BlockScheduler>;

    typename MoeGateCorrKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC};

    // call a kernel
    MoeGateCorrKernel moeGateCorr;
    moeGateCorr(params);
}

inline __aicore__ MoeGateCorrTilingData GetTilingData(GM_ADDR tiling)
{
    auto tilingDataPointer = reinterpret_cast<__gm__ MoeGateCorrTilingData *>(tiling);
    MoeGateCorrTilingData tilingData;
    tilingData.m = tilingDataPointer->m;
    tilingData.n = tilingDataPointer->n;
    tilingData.k = tilingDataPointer->k;
    return tilingData;
}

#endif  // MOE_GATE_CORR_H