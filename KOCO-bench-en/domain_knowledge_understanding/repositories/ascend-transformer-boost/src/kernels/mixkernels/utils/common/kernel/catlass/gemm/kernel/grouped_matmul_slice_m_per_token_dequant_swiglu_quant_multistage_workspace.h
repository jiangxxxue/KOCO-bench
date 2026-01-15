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
#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/detail/callback.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"

namespace Catlass::Gemm::Kernel {

template <class ArchTag>
class BlockQuant {
public:
    using ElementInput = float;
    using LayoutInput = layout::RowMajor;
    using ElementDequantScale = float;
    using LayoutDequantScale = layout::VectorLayout;
    using ElementOutput = int8_t;
    using LayoutOutput = layout::RowMajor;

    using InputType = GemmType<ElementInput, LayoutInput>;
    using DequantScaleType = GemmType<ElementDequantScale, LayoutDequantScale>;
    using OutputType = GemmType<ElementOutput, LayoutOutput>;

    constexpr static uint32_t TILE_ROW = 8;
    constexpr static uint32_t TILE_COLUMN = 2048;
    constexpr static uint32_t HALF_TILE_COLUMN = 1024;
    using TileShape = MatrixShape<TILE_ROW, TILE_COLUMN>;
    using HalfTileShape = MatrixShape<TILE_ROW, HALF_TILE_COLUMN>;

    using EpilogueTileSwizzle = Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    struct Params {
        __gm__ ElementInput *ptrInput{nullptr};
        LayoutInput layoutInput;
        __gm__ ElementDequantScale *ptrDequantScale{nullptr};
        LayoutDequantScale layoutDequantScale;
        __gm__ ElementOutput *ptrOutput{nullptr};
        LayoutOutput layoutOutput;

        CATLASS_DEVICE
        Params() {};

        CATLASS_DEVICE
        Params(
            __gm__ ElementInput *ptrInput_, LayoutInput const &layoutInput_,
            __gm__ ElementDequantScale *ptrQuantScale_, LayoutDequantScale const &layoutQuantScale_,
            __gm__ ElementOutput *ptrOutput_, LayoutOutput const layoutOutput_
        ) : ptrInput(ptrInput_), layoutInput(layoutInput_),
            ptrDequantScale(ptrQuantScale_), layoutDequantScale(layoutQuantScale_),
            ptrOutput(ptrOutput_), layoutOutput(layoutOutput_) {}
    };

    CATLASS_DEVICE
    BlockQuant(Arch::Resource<ArchTag> const &resource, Params const &params_) : params(params_)
    {
        int64_t ubOffset = 0;
        ubInput = resource.ubBuf.template GetBufferByByte<ElementInput>(ubOffset);
        ubOffset += TileShape::COUNT * sizeof(ElementInput);
        ubDequantScale = resource.ubBuf.template GetBufferByByte<ElementDequantScale>(ubOffset);
        ubOffset += TileShape::ROW * sizeof(ElementDequantScale);
        ubOutput = resource.ubBuf.template GetBufferByByte<ElementOutput>(ubOffset);
        ubOffset += TileShape::COUNT * sizeof(ElementOutput);

        ubAbs = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TileShape::COUNT * sizeof(float);
        ubMax = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += HalfTileShape::COUNT * sizeof(float);
        ubReduceMax = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TileShape::ROW * sizeof(float);
        ubQuantScale = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TileShape::ROW * sizeof(float);
        ubInputTmp = ubAbs;
        ubQuantF32 = ubAbs;
        ubQuantS32 = ubAbs.ReinterpretCast<int32_t>();
        ubQuantF16 = ubAbs.ReinterpretCast<half>();

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
    }

    CATLASS_DEVICE
    ~BlockQuant()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
    }

    CATLASS_DEVICE
    void operator() (
        MatrixCoord const &blockShape,
        MatrixCoord const &blockCoord,
        MatrixCoord const &actualBlockShape
    )
    {
        MatrixCoord blockOffset = blockCoord * blockShape;

        AscendC::GlobalTensor<ElementInput> gmInput;
        gmInput.SetGlobalBuffer(params.ptrInput);
        AscendC::GlobalTensor<ElementDequantScale> gmDequantScale;
        gmDequantScale.SetGlobalBuffer(params.ptrDequantScale);
        AscendC::GlobalTensor<ElementOutput> gmOutput;
        gmOutput.SetGlobalBuffer(params.ptrOutput);

        auto ubTileStride = MakeCoord(static_cast<int64_t>(TileShape::COLUMN), 1L);
        auto ubHalfTileStride = MakeCoord(static_cast<int64_t>(HalfTileShape::COLUMN), 1L);
        auto tileShape = TileShape::ToCoord();
        EpilogueTileSwizzle epilogueTileSwizzle(actualBlockShape, tileShape);
        uint32_t tileLoops = epilogueTileSwizzle.GetLoops();
        uint32_t subblockIdx = AscendC::GetSubBlockIdx();
        uint32_t subblockNum = AscendC::GetSubBlockNum();
        for (uint32_t loopIdx = subblockIdx; loopIdx < tileLoops; loopIdx += subblockNum) {
            auto tileCoord = epilogueTileSwizzle.GetTileCoord(loopIdx);
            auto actualTileShape = epilogueTileSwizzle.GetActualTileShape(tileCoord);
            auto tileOffsetInBlock = tileCoord * tileShape;
            auto tileOffset = blockOffset + tileOffsetInBlock;

            auto gmTileInput = gmInput[params.layoutInput.GetOffset(tileOffset)];
            auto layoutGmTileInput = params.layoutInput.GetTileLayout(actualTileShape);

            layout::RowMajor layoutUbInput{actualTileShape, ubTileStride};

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            copyGmToUbInput(ubInput, gmTileInput, layoutUbInput, layoutGmTileInput);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Abs(ubAbs, ubInput, TileShape::COUNT);
            AscendC::PipeBarrier<PIPE_V>();

            for (uint32_t rowIdx = 0; rowIdx < HalfTileShape::ROW; ++rowIdx) {
                AscendC::Max(ubMax[rowIdx * HalfTileShape::COLUMN],
                    ubAbs[rowIdx * TileShape::COLUMN],
                    ubAbs[rowIdx * TileShape::COLUMN + HalfTileShape::COLUMN],
                    HalfTileShape::COLUMN);
            }

            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(ubInputTmp, ubInput, 127.f, TileShape::COUNT);

            constexpr uint32_t elementPerBlk = BYTE_PER_BLK / sizeof(float);
            constexpr int32_t mask = 64;

            AscendC::BinaryRepeatParams maxParams;
            maxParams.dstBlkStride = HalfTileShape::COLUMN / elementPerBlk;
            maxParams.src0BlkStride = HalfTileShape::COLUMN / elementPerBlk;
            maxParams.src1BlkStride = HalfTileShape::COLUMN / elementPerBlk;
            maxParams.dstRepStride = 1;
            maxParams.src0RepStride = 1;
            maxParams.src1RepStride = 1;
            constexpr uint32_t colNumPerCompute = BYTE_PER_VECTOR_FRACTAL / sizeof(float);
            uint32_t reduceWidth = HalfTileShape::COLUMN;
            while (reduceWidth > (BLK_NUM_PER_VECTOR_FRACTAL * BYTE_PER_BLK / sizeof(float))) {
                reduceWidth >>= 1;
                AscendC::Max(ubMax, ubMax, ubMax[reduceWidth], mask, reduceWidth / elementPerBlk, maxParams);
                AscendC::PipeBarrier<PIPE_V>();
            }

            AscendC::WholeReduceMax(ubReduceMax, ubMax, mask, HalfTileShape::ROW,
                1, 1, HalfTileShape::COLUMN / elementPerBlk, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::Muls(ubDequantScale, ubReduceMax, 1.0f / 127.0f, TileShape::ROW);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            auto dequantScaleTileOffset = tileOffset.template GetCoordByAxis<0>();
            auto dequantScaleTileShape = actualTileShape.template GetCoordByAxis<0>();

            auto gmTileDequantScale = gmDequantScale[params.layoutDequantScale.GetOffset(dequantScaleTileOffset)];
            auto layoutGmTileDequantScale = params.layoutDequantScale.GetTileLayout(dequantScaleTileShape);

            auto layoutUbDequantScale = LayoutDequantScale::template MakeLayoutInUb<ElementDequantScale>(
                dequantScaleTileShape);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            copyUbToGmDequantScale(gmTileDequantScale, ubDequantScale, layoutGmTileDequantScale, layoutUbDequantScale);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);

            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            for (uint32_t rowIdx = 0; rowIdx < TileShape::ROW; ++rowIdx) {
                AscendC::Muls(ubQuantF32[rowIdx * TileShape::COLUMN], ubInputTmp[rowIdx * TileShape::COLUMN],
                    1.f / ubReduceMax.GetValue(rowIdx), TileShape::COLUMN);
            }

            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(ubQuantS32, ubQuantF32, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetDeqScale(static_cast<half>(1.0));
            AscendC::Cast(ubQuantF16, ubQuantS32, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::Cast(ubOutput, ubQuantF16, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);

            auto gmTileOutput = gmOutput[params.layoutOutput.GetOffset(tileOffset)];
            auto layoutGmTileOutput = params.layoutOutput.GetTileLayout(actualTileShape);

            LayoutOutput layoutUbOutput{actualTileShape, ubTileStride};

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            copyUbToGmOutput(gmTileOutput, ubOutput, layoutGmTileOutput, layoutUbOutput);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        }
    }

private:
    Params params;

    AscendC::LocalTensor<ElementInput> ubInput;
    AscendC::LocalTensor<ElementDequantScale> ubDequantScale;
    AscendC::LocalTensor<ElementOutput> ubOutput;

    AscendC::LocalTensor<float> ubAbs;
    AscendC::LocalTensor<float> ubMax;
    AscendC::LocalTensor<float> ubReduceMax;
    AscendC::LocalTensor<float> ubQuantScale;
    AscendC::LocalTensor<float> ubQuantScaleBrcb;
    AscendC::LocalTensor<float> ubInputTmp;
    AscendC::LocalTensor<float> ubQuantF32;
    AscendC::LocalTensor<int32_t> ubQuantS32;
    AscendC::LocalTensor<half> ubQuantF16;

    Epilogue::Tile::CopyGm2Ub<ArchTag, InputType> copyGmToUbInput;
    Epilogue::Tile::CopyUb2Gm<ArchTag, DequantScaleType> copyUbToGmDequantScale;
    Epilogue::Tile::CopyUb2Gm<ArchTag, OutputType> copyUbToGmOutput;
};

template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    uint32_t WORKSPACE_STAGES_,
    class ElementGroupList_
>
class GroupedMatmulSliceMPerTokenDequantSwigluQuantMultiStageWorkspace {
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

    using BlockEpilogue = BlockEpilogue_;
    using ElementScale = typename BlockEpilogue::ElementScale;
    using LayoutScale = typename BlockEpilogue::LayoutScale;
    using ElementPerTokenScale = typename BlockEpilogue::ElementPerTokenScale;
    using LayoutPerTokenScale = typename BlockEpilogue::LayoutPerTokenScale;
    using ElementD = typename BlockEpilogue::ElementD;
    using LayoutD = typename BlockEpilogue::LayoutD;
    using EpilogueParams = typename BlockEpilogue::Params;

    using ElementDequantScale = typename BlockQuant<ArchTag>::ElementDequantScale;
    using LayoutDequantScale = typename BlockQuant<ArchTag>::LayoutDequantScale;
    using ElementOutput = typename BlockQuant<ArchTag>::ElementOutput;
    using LayoutOutput = typename BlockQuant<ArchTag>::LayoutOutput;

    using BlockScheduler = BlockScheduler_;
    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;
    using ElementGroupList = ElementGroupList_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        uint32_t problemCount;
        __gm__ ElementGroupList_ *ptrGroupList;
        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB;
        LayoutB layoutB;
        __gm__ ElementScale *ptrScale;
        LayoutScale layoutScale;
        __gm__ ElementPerTokenScale *ptrPerTokenScale;
        LayoutPerTokenScale layoutPerTokenScale;
        __gm__ ElementOutput *ptrOutput;
        LayoutOutput layoutOutput;
        __gm__ ElementDequantScale *ptrDequantScale;
        LayoutDequantScale layoutDequantScale;
        GM_ADDR ptrWorkspace;

        // Methods
        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
        Params(
            GemmCoord problemShape_, uint32_t problemCount_, GM_ADDR ptrGroupList_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrScale_, LayoutScale const &layoutScale_,
            GM_ADDR ptrPerTokenScale_, LayoutPerTokenScale const &layoutPerTokenScale_,
            GM_ADDR ptrOutput_, LayoutOutput const &layoutOutput_,
            GM_ADDR ptrDequantScale_, LayoutDequantScale const &layoutDequantScale_,
            GM_ADDR ptrWorkspace_
        ) : problemShape(problemShape_),
            problemCount(problemCount_), ptrGroupList(reinterpret_cast<__gm__ ElementGroupList *>(ptrGroupList_)),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrScale(reinterpret_cast<__gm__ ElementScale *>(ptrScale_)), layoutScale(layoutScale_),
            ptrPerTokenScale(reinterpret_cast<__gm__ ElementPerTokenScale *>(ptrPerTokenScale_)),
            layoutPerTokenScale(layoutPerTokenScale_),
            ptrOutput(reinterpret_cast<__gm__ ElementOutput *>(ptrOutput_)), layoutOutput(layoutOutput_),
            ptrDequantScale(reinterpret_cast<__gm__ ElementDequantScale *>(ptrDequantScale_)),
            layoutDequantScale(layoutDequantScale_),
            ptrWorkspace(ptrWorkspace_)
        {
        }
    };

    // Methods
    CATLASS_DEVICE
    GroupedMatmulSliceMPerTokenDequantSwigluQuantMultiStageWorkspace()
    {
        Arch::FlagID flagId = 0;
        for (uint32_t stageId = 0; stageId < WORKSPACE_STAGES; ++stageId) {
            flagAicFinishStoreList[stageId] = Arch::CrossCoreFlag(flagId++);
            flagAivFinishComputeList[stageId] = Arch::CrossCoreFlag(flagId++);
            aicWaitFuncList[stageId] = {this, stageId};
            aicSetFuncList[stageId] = {this, stageId};
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));
        auto layoutC = layout::RowMajor{L1TileShape::M * coreNum * WORKSPACE_STAGES, L1TileShape::N};

        uint32_t stageId = 0;
        uint32_t stageUsed = 0;
        uint32_t startCoreIdx = 0;
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = (groupIdx == 0) ? groupList.GetValue(groupIdx) :
                (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

            AscendC::GlobalTensor<ElementB> gmBInGroup;
            gmBInGroup.SetGlobalBuffer(params.ptrB + gmGroupOffsetB);
            if (currentM <= L1TileShape::M) {
                gmBInGroup.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }

            LayoutA layoutA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutB = params.layoutB;

            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
            // Loop through the matmul of each groupIdx
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                // Compute block location
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

                Callback callbackBeforeFixpipe{};
                if (stageUsed == WORKSPACE_STAGES) {
                    callbackBeforeFixpipe = MakeCallback(&aicWaitFuncList[stageId]);
                } else {
                    ++stageUsed;
                }
                Callback callbackAfterFixpipe = MakeCallback(&aicSetFuncList[stageId]);

                // Compute initial location in logical coordinates
                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{(stageId * coreNum + coreIdx) * L1TileShape::M, 0};
                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);

                // Compute block-scoped matrix multiply-add
                if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                    blockMmad(
                        gmA[gmGroupOffsetA + gmOffsetA], layoutA,
                        gmBInGroup[gmOffsetB], layoutB,
                        gmC[gmOffsetC], layoutC,
                        actualBlockShape,
                        callbackBeforeFixpipe, callbackAfterFixpipe
                    );
                } else {
                    callbackBeforeFixpipe();
                    blockMmad(
                        gmA[gmGroupOffsetA + gmOffsetA], layoutA,
                        gmBInGroup[gmOffsetB], layoutB,
                        gmC[gmOffsetC], layoutC,
                        actualBlockShape
                    );
                    callbackAfterFixpipe();
                }

                stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
            }

            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }

        while (stageUsed > 0) {
            uint32_t aivComputeStageId = (stageId >= stageUsed) ?
                (stageId - stageUsed) : (stageId + WORKSPACE_STAGES - stageUsed);
            Arch::CrossCoreWaitFlag(flagAivFinishComputeList[aivComputeStageId]);
            --stageUsed;
        }
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t gmGroupOffsetScale = 0;
        int64_t gmGroupOffsetPerTokenScale = 0;
        int64_t gmGroupOffsetD = 0;

        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrWorkspace));
        auto layoutC = layout::RowMajor{L1TileShape::M * coreNum * WORKSPACE_STAGES, L1TileShape::N};

        auto ptrD = reinterpret_cast<__gm__ float *>(
            params.ptrWorkspace + sizeof(int32_t) * (L1TileShape::M * coreNum * WORKSPACE_STAGES * L1TileShape::N));

        uint32_t mActual = groupList.GetValue(params.problemCount - 1);
        uint32_t nOut = params.problemShape.n() / 2;

        {
            BlockScheduler blockScheduler;
            BlockEpilogue blockEpilogue(resource);

            uint32_t stageId = 0;
            uint32_t startCoreIdx = 0;
            for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
                uint32_t currentM = (groupIdx == 0) ? groupList.GetValue(groupIdx) :
                    (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
                GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

                LayoutScale layoutScale = params.layoutScale;
                LayoutPerTokenScale layoutPerTokenScale =
                    params.layoutPerTokenScale.GetTileLayout(inGroupProblemShape.template GetCoordByAxis<0>());
                LayoutD layoutD = params.layoutOutput.GetTileLayout(MakeCoord(currentM, nOut));

                EpilogueParams epilogueParams{
                    params.ptrScale + gmGroupOffsetScale, layoutScale,
                    params.ptrPerTokenScale + gmGroupOffsetPerTokenScale, layoutPerTokenScale,
                    ptrD + gmGroupOffsetD, layoutD
                };

                blockScheduler.Update(inGroupProblemShape, L1TileShape::ToCoordMN());
                blockEpilogue.UpdateParams(epilogueParams);
                uint32_t coreLoops = blockScheduler.GetCoreLoops();

                GemmCoord blockShapeMNK = L1TileShape::ToCoord();
                uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
                for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                    GemmCoord blockCoordMNK = blockScheduler.GetBlockCoord(loopIdx);
                    GemmCoord actualBlockShapeMNK = blockScheduler.GetActualBlockShape(blockCoordMNK);

                    MatrixCoord offsetC{(stageId * coreNum + coreIdx) * L1TileShape::M, 0};
                    int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                    auto gmBlockC = gmC[gmOffsetC];
                    auto layoutBlockC = layoutC.GetTileLayout(actualBlockShapeMNK.GetCoordMN());

                    Arch::CrossCoreWaitFlag(flagAicFinishStoreList[stageId]);
                    blockEpilogue(blockShapeMNK, blockCoordMNK, actualBlockShapeMNK, gmBlockC, layoutBlockC);
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComputeList[stageId]);

                    stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
                }

                gmGroupOffsetScale += inGroupProblemShape.n();
                gmGroupOffsetPerTokenScale += inGroupProblemShape.m();
                gmGroupOffsetD += currentM * nOut;

                startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
            }
        }

        Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();

        {
            typename BlockQuant<ArchTag>::Params quantParams{
                ptrD, params.layoutOutput,
                params.ptrDequantScale, params.layoutDequantScale,
                params.ptrOutput, params.layoutOutput
            };

            BlockQuant<ArchTag> blockQuant(resource, quantParams);
            MatrixCoord quantShape(mActual, nOut);
            MatrixCoord quantBlockShape(16U, 2048U);
            Epilogue::Tile::EpilogueHorizontalTileSwizzle quantSwizzle(quantShape, quantBlockShape);
            for (uint32_t loopIdx = coreIdx; loopIdx < quantSwizzle.GetLoops(); loopIdx += coreNum) {
                auto blockCoord = quantSwizzle.GetTileCoord(loopIdx);
                auto actualBlockShape = quantSwizzle.GetActualTileShape(blockCoord);

                blockQuant(quantBlockShape, blockCoord, actualBlockShape);
            }
        }
    }

private:
    friend struct AicWaitFunc;
    friend struct AicSetFunc;

    struct AicWaitFunc {
        using MatmulKernel = GroupedMatmulSliceMPerTokenDequantSwigluQuantMultiStageWorkspace<
            BlockMmad, BlockEpilogue, BlockScheduler, WORKSPACE_STAGES, ElementGroupList
        >;

        CATLASS_DEVICE
        AicWaitFunc() = default;

        CATLASS_DEVICE
        void operator()() const
        {
            Arch::CrossCoreWaitFlag(ptr->flagAivFinishComputeList[stageId]);
        }

        MatmulKernel *ptr{nullptr};
        uint32_t stageId;
    };

    struct AicSetFunc {
        using MatmulKernel = GroupedMatmulSliceMPerTokenDequantSwigluQuantMultiStageWorkspace<
            BlockMmad, BlockEpilogue, BlockScheduler, WORKSPACE_STAGES, ElementGroupList
        >;

        CATLASS_DEVICE
        AicSetFunc() = default;

        CATLASS_DEVICE
        void operator()() const
        {
            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(ptr->flagAicFinishStoreList[stageId]);
        }

        MatmulKernel *ptr{nullptr};
        uint32_t stageId;
    };

    Arch::CrossCoreFlag flagAicFinishStoreList[WORKSPACE_STAGES];
    Arch::CrossCoreFlag flagAivFinishComputeList[WORKSPACE_STAGES];

    AicWaitFunc aicWaitFuncList[WORKSPACE_STAGES];
    AicSetFunc aicSetFuncList[WORKSPACE_STAGES];
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel