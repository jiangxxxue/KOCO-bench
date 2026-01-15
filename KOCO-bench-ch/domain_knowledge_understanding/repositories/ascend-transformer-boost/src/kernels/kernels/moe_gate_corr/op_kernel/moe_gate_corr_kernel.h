/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_GATE_CORR_KERNEL_H
#define MOE_GATE_CORR_KERNEL_H

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

using namespace Catlass;

template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class MoeGateCorrKernel {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using BlockScheduler = BlockScheduler_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;

        // Methods
        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
        Params(GemmCoord const &problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_,
               LayoutB layoutB_, GM_ADDR ptrC_, LayoutC layoutC_)
            : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
              ptrC(ptrC_), layoutC(layoutC_) {}
    };

    // Methods
    CATLASS_DEVICE
    MoeGateCorrKernel() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params) {
        BlockScheduler blockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = blockScheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        for (uint32_t loopIdx = coreIdx; loopIdx < coreLoops; loopIdx += coreNum) {
            // Compute block location
            GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

            // Compute initial location in logical coordinates
            MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
            MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
            MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
            int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
            int64_t gmOffsetC = params.layoutC.GetOffset(offsetC);

            // Compute block-scoped matrix multiply-add
            blockMmad(gmA[gmOffsetA], params.layoutA,
                      gmB[gmOffsetB], params.layoutB,
                      gmC[gmOffsetC], params.layoutC,
                      actualBlockShape, Callback{}, Callback{});
        }
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params) {}
};

#endif  // MOE_GATE_CORR_KERNEL_H