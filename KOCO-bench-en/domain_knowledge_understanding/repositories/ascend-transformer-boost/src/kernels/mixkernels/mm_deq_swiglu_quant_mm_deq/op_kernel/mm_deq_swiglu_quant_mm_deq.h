/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#define L2_CACHE_HINT
#include <kernel_operator.h>
#undef inline

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/gemm_type.hpp"

#include "epilogue/dispatch_policy.h"
#include "gemm/dispatch_policy.h"

#include "epilogue/block/block_epilogue.h"
#include "gemm/block/block_mmad.h"
#include "gemm/kernel/matmul_per_token_dequant_multistage_workspace.h"
#include "gemm/kernel/matmul_per_token_dequant_swiglu_quant_multistage_workspace.h"

#define inline __inline__ __attribute__((always_inline))

#include "../tiling/mm_deq_swiglu_quant_mm_deq_tiling_data.h"

#ifdef __DAV_C220_CUBE__
__gm__ struct OpSystemRunCfg g_opSystemRunCfg{0};
#else
extern __gm__ struct OpSystemRunCfg g_opSystemRunCfg;
#endif

using namespace Catlass;
using namespace AtbOps;

using MmadAtlasA2Custom = Gemm::MmadAtlasA2PreloadAsyncWithCallback<
    CUSTOM_PRELOAD_STAGES, CUSTOM_L1_STAGES,
    CUSTOM_L0A_STAGES, CUSTOM_L0B_STAGES, CUSTOM_L0C_STAGES,
    CUSTOM_ENABLE_UNIT_FLAG, CUSTOM_ENABLE_SHUFFLE_K
>;

using Mm1BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<MM1_SWIZZLE_OFFSET, MM1_SWIZZLE_DIRECTION>;

using Mm2L1TileShape = GemmShape<MM2_L1M, MM2_L1N, MM2_L1K>;
using Mm2L0TileShape = GemmShape<Mm2L1TileShape::M, Mm2L1TileShape::N, MM2_L0K>;
using Mm2EpilogueTileShape = MatrixShape<MM2_EPIM, Mm2L1TileShape::N>;
using Mm2BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<MM2_SWIZZLE_OFFSET, MM2_SWIZZLE_DIRECTION>;
using Mm2DispatchPolicy = Gemm::MmadAtlasA2PreloadAsyncWithCallbackResidentA<
    CUSTOM_PRELOAD_STAGES, MM2_L1A_STAGES, MM2_L1B_STAGES,
    MM2_L0A_STAGES, MM2_L0B_STAGES, CUSTOM_L0C_STAGES,
    CUSTOM_ENABLE_UNIT_FLAG, CUSTOM_ENABLE_SHUFFLE_K
>;

template <
    class L1TileShape_,
    class L0TileShape_,
    class EpilogueTileShape_,
    class BlockScheduler_ = Mm1BlockScheduler,
    class DispatchPolicy_ = MmadAtlasA2Custom
>
CATLASS_DEVICE
void MmDeqSwigluQuant(
    GemmCoord problemShape,
    GM_ADDR gmA, layout::RowMajor layoutA,
    GM_ADDR gmB, layout::zN layoutB,
    GM_ADDR gmScale, layout::VectorLayout layoutScale,
    GM_ADDR gmPerTokenScale, layout::VectorLayout layoutPerTokenScale,
    GM_ADDR gmD, layout::RowMajor layoutD,
    GM_ADDR gmDequantScale, layout::VectorLayout layoutDequantScale,
    GM_ADDR gmWorkspace
)
{
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = DispatchPolicy_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;

    using AType = Gemm::GemmType<int8_t, layout::RowMajor>;
    using BType = Gemm::GemmType<int8_t, layout::zN>;
    using CType = Gemm::GemmType<int32_t, layout::RowMajor>;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    constexpr uint32_t ubStages = 1;
    using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2PerTokenDequantSwiglu<ubStages>;
    using ScaleType = Gemm::GemmType<float, layout::VectorLayout>;
    using PerTokenScaleType = Gemm::GemmType<float, layout::VectorLayout>;
    using DType = Gemm::GemmType<float, layout::RowMajor>;

    using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
    using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;

    using EpilogueTileShape = EpilogueTileShape_;
    using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
    using TileBroadcastOneBlk = Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType,
        EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMul = Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag,
        OneBlkColumnBroadcastMulType, EpilogueTileShape>;
    using TileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, DType>;
    using TileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using BlockEpilogue = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType, ScaleType, PerTokenScaleType,
        DType, TileRowBroadcastMul, TileBroadcastOneBlk, TileOneBlkColumnBroadcastMul, TileCopy, TileScheduler>;

    using BlockScheduler = BlockScheduler_;

    // kernel level
    using GemmKernel = Gemm::Kernel::MatmulPerTokenDequantSwigluQuantMultiStageWorkspace<
        BlockMmad, BlockEpilogue, BlockScheduler, WORKSPACE_STAGES
    >;

    typename GemmKernel::Params params{
        problemShape,
        gmA, layoutA,
        gmB, layoutB,
        gmScale, layoutScale,
        gmPerTokenScale, layoutPerTokenScale,
        gmD, layoutD,
        gmDequantScale, layoutDequantScale,
        gmWorkspace
    };

    // call a kernel
    GemmKernel gemm;
    gemm(params);
}

template <
    class L1TileShape_ = Mm2L1TileShape,
    class L0TileShape_ = Mm2L0TileShape,
    class EpilogueTileShape_ = Mm2EpilogueTileShape,
    class BlockScheduler_ = Mm2BlockScheduler,
    class DispatchPolicy_ = Mm2DispatchPolicy
>
CATLASS_DEVICE
void MmDeq(
    GemmCoord problemShape,
    GM_ADDR gmA, layout::RowMajor layoutA,
    GM_ADDR gmB, layout::nZ layoutB,
    GM_ADDR gmScale, layout::VectorLayout layoutScale,
    GM_ADDR gmPerTokenScale, layout::VectorLayout layoutPerTokenScale,
    GM_ADDR gmD, layout::RowMajor layoutD,
    GM_ADDR gmWorkspace
)
{
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = DispatchPolicy_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;

    using AType = Gemm::GemmType<int8_t, layout::RowMajor>;
    using BType = Gemm::GemmType<int8_t, layout::nZ>;
    using CType = Gemm::GemmType<int32_t, layout::RowMajor>;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    constexpr uint32_t ubStages = 1;
    using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
    using ScaleType = Gemm::GemmType<float, layout::VectorLayout>;
    using PerTokenScaleType = Gemm::GemmType<float, layout::VectorLayout>;
    using DType = Gemm::GemmType<half, layout::RowMajor>;

    using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
    using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;

    using EpilogueTileShape = EpilogueTileShape_;
    using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
    using TileBroadcastOneBlk = Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType,
        EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMul = Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag,
        OneBlkColumnBroadcastMulType, EpilogueTileShape>;
    using TileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, DType>;
    using TileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using BlockEpilogue = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType, ScaleType, PerTokenScaleType,
        DType, TileRowBroadcastMul, TileBroadcastOneBlk, TileOneBlkColumnBroadcastMul, TileCopy, TileScheduler>;

    using BlockScheduler = BlockScheduler_;

    // kernel level
    using GemmKernel = Gemm::Kernel::MatmulPerTokenDequantMultiStageWorkspace<
        BlockMmad, BlockEpilogue, BlockScheduler, WORKSPACE_STAGES
    >;

    typename GemmKernel::Params params{
        problemShape,
        gmA, layoutA,
        gmB, layoutB,
        gmScale, layoutScale,
        gmPerTokenScale, layoutPerTokenScale,
        gmD, layoutD,
        gmWorkspace
    };

    // call a kernel
    GemmKernel gemm;
    gemm(params);
}

CATLASS_DEVICE
void BarrierBetweenUpAndDown()
{
    AscendC::PipeBarrier<PIPE_ALL>();
    Arch::CrossCoreFlag mm1AivFinished{0};
    if constexpr (g_coreType == AscendC::AIV) {
        Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(mm1AivFinished);
    } else {
        Arch::CrossCoreWaitFlag(mm1AivFinished);
    }
}

inline __aicore__ MmDeqSwigluQuantMmDeqTilingData GetTilingData(GM_ADDR tiling)
{
    auto tilingDataPointer = reinterpret_cast<__gm__ MmDeqSwigluQuantMmDeqTilingData *>(tiling);
    MmDeqSwigluQuantMmDeqTilingData tilingData;
    tilingData.m = tilingDataPointer->m;
    tilingData.n = tilingDataPointer->n;
    tilingData.k = tilingDataPointer->k;
    return tilingData;
}
