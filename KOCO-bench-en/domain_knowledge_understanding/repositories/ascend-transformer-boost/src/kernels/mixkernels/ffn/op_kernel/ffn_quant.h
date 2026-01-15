/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ASCEND_OPS_FFN_QUANT_H
#define ASCEND_OPS_FFN_QUANT_H

#include "kernel_operator.h"
#include "ffn_common.h"
#include "ffn_gelu.h"
#include "ffn_matmul.h"
#include "mixkernels/ffn/tiling/tiling_data.h"
#include "kernels/utils/kernel/common.h"
#include "kernels/utils/kernel/common_func.h"
#include "kernels/utils/kernel/hardware.h"
#include "kernels/utils/kernel/simd.h"
#include "kernels/utils/kernel/iterator.h"
#include "kernels/utils/kernel/mma.h"
#include "kernels/utils/kernel/utils.h"

namespace FFN {
template <
    typename DESCALE_DTYPE,
    typename BIAS_DTYPE,
    typename ACCUMULATOR_DTYPE,
    typename OUT_DTYPE,
    CubeFormat IN_FORMAT,
    CubeFormat OUT_FORMAT,
    bool HIGH_PRECISION,
    bool HIGH_PERFORMANCE
>
struct DefaultFFN<ArchType::ASCEND_V200, int8_t, DESCALE_DTYPE, BIAS_DTYPE, ACCUMULATOR_DTYPE, OUT_DTYPE,
    IN_FORMAT, OUT_FORMAT, false, true, true, HIGH_PRECISION, HIGH_PERFORMANCE> {
    __aicore__ inline explicit DefaultFFN()
    {
        AscendC::SetLoadDataPaddingValue<uint64_t>((uint64_t)0x0);
        AscendC::SetAtomicNone();
        AscendC::SetMaskNorm();
    };

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *weight1, __gm__ uint8_t *weight2,
        __gm__ uint8_t *scaleQuant, __gm__ uint8_t *scaleDequant1, __gm__ uint8_t *scaleDequant2,
        __gm__ uint8_t *biasQuant, __gm__ uint8_t *biasDequant1, __gm__ uint8_t *biasDequant2,
        __gm__ uint8_t *y, __gm__ uint8_t *workspace, const AtbOps::FFNTilingData *tiling)
    {
        gmX.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DTYPE *>(x));
        gmW1.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DTYPE *>(weight1));
        gmW2.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DTYPE *>(weight2));
        gmScaleQuant.SetGlobalBuffer(reinterpret_cast<__gm__ ACCUMULATOR_DTYPE *>(scaleQuant));
        gmScaleDq1.SetGlobalBuffer(reinterpret_cast<__gm__ DESCALE_DTYPE *>(scaleDequant1));
        gmScaleDq2.SetGlobalBuffer(reinterpret_cast<__gm__ DESCALE_DTYPE *>(scaleDequant2));
        gmBiasQuant.SetGlobalBuffer(reinterpret_cast<__gm__ ACCUMULATOR_DTYPE *>(biasQuant));
        gmBiasDq1.SetGlobalBuffer(reinterpret_cast<__gm__ BIAS_DTYPE *>(biasDequant1));
        gmBiasDq2.SetGlobalBuffer(reinterpret_cast<__gm__ BIAS_DTYPE *>(biasDequant2));
        gmYAccu.SetGlobalBuffer(reinterpret_cast<__gm__ ACCUMULATOR_DTYPE *>(y));
        gmY.SetGlobalBuffer(reinterpret_cast<__gm__ OUT_DTYPE *>(y));
        gmSync.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace));
        gmYTmp.SetGlobalBuffer(reinterpret_cast<__gm__ IN_DTYPE *>(workspace + FRACTAL_SIZE));
        const AtbOps::FFNTilingData *tilingData = tiling;
        m1 = tilingData->mm1.m;
        k1 = tilingData->mm1.k;
        n1 = tilingData->mm1.n;
        baseM1 = tilingData->mm1.baseM;
        baseK1 = tilingData->mm1.baseK;
        baseN1 = tilingData->mm1.baseN;
        m1Loops = tilingData->mm1.mLoops;
        k1Loops = tilingData->mm1.kLoops;
        n1Loops = tilingData->mm1.nLoops;
        coreLoops1 = tilingData->mm1.coreLoops;
        swizzlCnt1 = tilingData->mm1.swizzlCount;
        swizzlDirect1 = tilingData->mm1.swizzlDirect;

        m2 = tilingData->mm2.m;
        k2 = tilingData->mm2.k;
        n2 = tilingData->mm2.n;
        baseM2 = tilingData->mm2.baseM;
        baseK2 = tilingData->mm2.baseK;
        baseN2 = tilingData->mm2.baseN;
        m2Loops = tilingData->mm2.mLoops;
        k2Loops = tilingData->mm2.kLoops;
        n2Loops = tilingData->mm2.nLoops;
        coreLoops2 = tilingData->mm2.coreLoops;
        swizzlCnt2 = tilingData->mm2.swizzlCount;
        swizzlDirect2 = tilingData->mm2.swizzlDirect;

        activationType = tilingData->activationType;
        mOrgUp1 = RoundUp(m1, BLOCK_SIZE_16);
        nOrgUp1 = RoundUp(n1, BLOCK_SIZE_16);
        mOrgUp2 = RoundUp(m2, BLOCK_SIZE_16);
        nOrgUp2 = RoundUp(n2, BLOCK_SIZE_16);
        
        ubScaleQuant = buf.GetBuffer<BufferType::ASCEND_UB, ACCUMULATOR_DTYPE>(tilingData->scaleOffset);
        ubBiasQuant = buf.GetBuffer<BufferType::ASCEND_UB, ACCUMULATOR_DTYPE>(tilingData->biasOffset);
        ubSync = buf.GetBuffer<BufferType::ASCEND_UB, int32_t>(tilingData->syncOffset);
        ubC1 = buf.GetBuffer<BufferType::ASCEND_UB, ACCUMULATOR_DTYPE>(tilingData->cOffset);
        ubC2 = buf.GetBuffer<BufferType::ASCEND_UB, OUT_DTYPE>(tilingData->cOffset);
        ubCQuant = buf.GetBuffer<BufferType::ASCEND_UB, IN_DTYPE>(tilingData->cOffset);
        ubTrans1 = buf.GetBuffer<BufferType::ASCEND_UB, ACCUMULATOR_DTYPE>(tilingData->geluOffset);
        ubTrans2 = buf.GetBuffer<BufferType::ASCEND_UB, OUT_DTYPE>(tilingData->geluOffset);
        ubGelu = buf.GetBuffer<BufferType::ASCEND_UB, uint8_t>(tilingData->geluOffset);
        ubGelu.SetSize(UB_HALF_BUFFER_LEN - tilingData->cOffset);
    }

    __aicore__ void Process()
    {
        uint32_t eachCoreNum = BLOCK_SIZE_32 / sizeof(int32_t);
        Duplicate(ubSync, 0, eachCoreNum);
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);
        DataCopy(gmSync[eachCoreNum * block_idx], ubSync, eachCoreNum);
        AscendC::PipeBarrier<PIPE_ALL>();
        Stage1MmVec();
        SyncAll(gmSync, ubSync);
        Stage2Mm();
    }

private:
    using IN_DTYPE = int8_t;
    using MM1 = DefaultMatmul<ArchType::ASCEND_V200, IN_DTYPE, IN_FORMAT, IN_DTYPE, IN_FORMAT,
        half, CubeFormat::NZ, BIAS_DTYPE, BIAS_DTYPE, DESCALE_DTYPE, false, true, true, true>;
    using MM2 = DefaultMatmul<ArchType::ASCEND_V200, IN_DTYPE, CubeFormat::NZ, IN_DTYPE, IN_FORMAT,
        OUT_DTYPE, CubeFormat::NZ, BIAS_DTYPE, BIAS_DTYPE, DESCALE_DTYPE, false, true, true, false>;

    __aicore__ inline void GetIdx(uint32_t loopIdx, matmulInfo &mmInfo,
        const uint32_t &swizzlDirect, const uint32_t &swizzlCnt)
    {
        uint32_t inBatchIdx = loopIdx % (mmInfo.mLoops * mmInfo.nLoops);
        uint32_t mIdx = 0;
        uint32_t nIdx = 0;
        if (swizzlDirect == 0) { // Zn
            uint32_t tileBlockLoop = (mmInfo.mLoops + swizzlCnt - 1) / swizzlCnt;
            uint32_t tileBlockIdx = inBatchIdx / (swizzlCnt * mmInfo.nLoops);
            uint32_t inTileBlockIdx = inBatchIdx % (swizzlCnt * mmInfo.nLoops);

            uint32_t nRow = swizzlCnt;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = mmInfo.mLoops - swizzlCnt * tileBlockIdx;
            }
            mIdx = tileBlockIdx * swizzlCnt + inTileBlockIdx % nRow;
            nIdx = inTileBlockIdx / nRow;
        } else if (swizzlDirect == 1) { // Nz
            uint32_t tileBlockLoop = (mmInfo.nLoops + swizzlCnt - 1) / swizzlCnt;
            uint32_t tileBlockIdx = inBatchIdx / (swizzlCnt * mmInfo.mLoops);
            uint32_t inTileBlockIdx = inBatchIdx % (swizzlCnt * mmInfo.mLoops);

            uint32_t nCol = swizzlCnt;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = mmInfo.nLoops - swizzlCnt * tileBlockIdx;
            }
            mIdx = inTileBlockIdx / nCol;
            nIdx = tileBlockIdx * swizzlCnt + inTileBlockIdx % nCol;
        }
        mmInfo.updateIdx(mIdx, nIdx);
    }

    __aicore__ inline void Stage2AscendQuantNz(AscendC::LocalTensor<IN_DTYPE> &dst,
        AscendC::LocalTensor<ACCUMULATOR_DTYPE> &src, AscendC::LocalTensor<ACCUMULATOR_DTYPE> &scaleQuantUb,
        AscendC::LocalTensor<ACCUMULATOR_DTYPE> &biasQuantUb, const uint32_t &height, const uint32_t &width)
    {
        uint64_t mask0 = 128;
        uint32_t calCount = height * width;
        WAIT_FLAG(MTE2, V, EVENT_ID3);
        for (uint32_t i = 0; i < CONST_2; i++) {
            Mul(src[i * calCount / CONST_2], src[i * calCount / CONST_2], scaleQuantUb[i * calCount / CONST_16],
                mask0, calCount / CONST_256, { 1, 1, 0, BLOCK_NUM_PER_VEC, BLOCK_NUM_PER_VEC, 1 });
        }

        for (uint32_t i = 0; i < width / BLOCK_SIZE_16; i += CONST_2) {
            Add(ubTrans1[i * height * BLOCK_SIZE_16], src[i * height * BLOCK_SIZE_16],
                biasQuantUb[i * BLOCK_SIZE_16], mask0, height / BLOCK_NUM_PER_VEC,
                { CONST_2, 1, 0, CONST_2 * BLOCK_NUM_PER_VEC, BLOCK_NUM_PER_VEC, 0 });
            Add(ubTrans1[i * height * BLOCK_SIZE_16 + BLOCK_SIZE_16], src[(i + 1) * height * BLOCK_SIZE_16],
                biasQuantUb[(i + 1) * BLOCK_SIZE_16], mask0, height / BLOCK_NUM_PER_VEC,
                { CONST_2, 1, 0, CONST_2 * BLOCK_NUM_PER_VEC, BLOCK_NUM_PER_VEC, 0 });
        }
        SET_FLAG(V, MTE2, EVENT_ID3);
        
        Cast(dst, ubTrans1, AscendC::RoundMode::CAST_NONE, calCount);
    }

    __aicore__ inline void GeluQuant(
        AscendC::LocalTensor<IN_DTYPE> ubCQuantTensor,  AscendC::LocalTensor<ACCUMULATOR_DTYPE> ubCTensor,
        const matmulInfo &mmInfo, const uint32_t &l0cPingPong)
    {
        WAIT_FLAG(V, MTE2, EVENT_ID3);
        for (uint32_t i = 0; i < mmInfo.mRound / BLOCK_NUM_PER_VEC; i++) {
            AscendC::DataCopy(ubScaleQuant[i * BLOCK_SIZE_16], gmScaleQuant[mmInfo.nIdx * baseN1],
                {static_cast<uint16_t>(mmInfo.nRound / BLOCK_SIZE_16), 1, 0,
                static_cast<uint16_t>(mmInfo.mRound / BLOCK_NUM_PER_VEC - 1)});
        }
        AscendC::DataCopy(ubBiasQuant, gmBiasQuant[mmInfo.nIdx * baseN1], mmInfo.nRound);
        SET_FLAG(MTE2, V, EVENT_ID3);
        
        ActivationType active = static_cast<ActivationType>(activationType);
        if (active == ActivationType::GELU) {
            AscendC::Gelu<ACCUMULATOR_DTYPE, HIGH_PRECISION, HIGH_PERFORMANCE>(
                ubCTensor, ubCTensor, ubGelu, mmInfo.mRound * mmInfo.nRound);
        } else if (active == ActivationType::FASTGELU) {
            AscendC::FasterGelu<ACCUMULATOR_DTYPE, HIGH_PRECISION, HIGH_PERFORMANCE>(
                ubCTensor, ubCTensor, ubGelu, mmInfo.mRound * mmInfo.nRound);
        } else if (active == ActivationType::FASTGELUV2) {
            FFN::FasterGeluV2<ACCUMULATOR_DTYPE, HIGH_PRECISION, HIGH_PERFORMANCE>(
                ubCTensor, ubCTensor, ubGelu, mmInfo.mRound * mmInfo.nRound);
        }
        
        Stage2AscendQuantNz(ubCQuantTensor, ubCTensor, ubScaleQuant, ubBiasQuant, mmInfo.mRound, mmInfo.nRound);
    }

    __aicore__ inline void CopyOut1(AscendC::LocalTensor<IN_DTYPE> ubCQuantTensor,
        const matmulInfo &mmInfo, const uint32_t &l0cPingPong)
    {
        event_t l0cEvent = l0cPingPong ? EVENT_ID0 : EVENT_ID1;
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);

        int64_t mOffset = static_cast<int64_t>(mmInfo.mIdx) * mmInfo.baseM * BLOCK_SIZE_32 +
            static_cast<int64_t>(mmInfo.nIdx) * mmInfo.baseN * mOrgUp1;
        uint32_t dstStride = mOrgUp1 - mmInfo.mRound;
        uint16_t blockCount = CeilDiv(mmInfo.nRound, BLOCK_SIZE_32);
        ub_to_gm<ArchType::ASCEND_V200, IN_DTYPE, DataFormat::NZ, DataFormat::NZ>(
            gmYTmp[mOffset], ubCQuantTensor, mmInfo.mRound, mmInfo.mRound, mOrgUp1, mmInfo.nRound, mmInfo.nRound, 0);
        
        SET_FLAG(MTE3, V, l0cEvent);
    }

    __aicore__ inline void Stage1MmVec()
    {
        SET_FLAG(MTE1, MTE2, EVENT_ID0);
        SET_FLAG(MTE1, MTE2, EVENT_ID1);
        SET_FLAG(MTE1, MTE2, EVENT_ID2);
        SET_FLAG(MTE1, MTE2, EVENT_ID3);
        SET_FLAG(M, MTE1, EVENT_ID0);
        SET_FLAG(M, MTE1, EVENT_ID1);
        SET_FLAG(MTE3, V, EVENT_ID0);
        SET_FLAG(MTE3, V, EVENT_ID1);
        SET_FLAG(V, MTE2, EVENT_ID0);
        SET_FLAG(V, MTE2, EVENT_ID1);
        SET_FLAG(V, MTE2, EVENT_ID2);
        SET_FLAG(V, MTE2, EVENT_ID3);
        uint32_t l0cPingPong = 1;
        matmulInfo mmInfo = {0, 0, m1, k1, n1, baseM1, baseK1, baseN1, m1Loops, k1Loops, n1Loops};
        matmulInfo nextMmInfo = {0, 0, m1, k1, n1, baseM1, baseK1, baseN1, m1Loops, k1Loops, n1Loops};
        GetIdx(block_idx, mmInfo, swizzlDirect1, swizzlCnt1);
        mm1.LoadBias(gmBiasDq1, mmInfo, l0cPingPong);
        mm1.BrcBias(mmInfo, l0cPingPong);
        for (uint32_t loopIdx = block_idx; loopIdx < coreLoops1; loopIdx += block_num) {
            AscendC::LocalTensor<ACCUMULATOR_DTYPE> ubCTensor = l0cPingPong ? ubC1 : ubC1[MATRIX_C_PINGPONG_LEN];
            AscendC::LocalTensor<IN_DTYPE> ubCQuantTensor =
                l0cPingPong ? ubCQuant : ubCQuant[MATRIX_C_PINGPONG_LEN * CONST_2];
            if (loopIdx + block_num < coreLoops1) {
                GetIdx(loopIdx + block_num, nextMmInfo, swizzlDirect1, swizzlCnt1);
                mm1.LoadBias(gmBiasDq1, nextMmInfo, 1 - l0cPingPong);
            }
            mm1.IterateAll(gmX, gmW1, ubCTensor, gmScaleDq1, mmInfo, l0cPingPong);
            if (loopIdx + block_num < coreLoops1) {
                mm1.BrcBias(nextMmInfo, 1 - l0cPingPong);
            }
            GeluQuant(ubCQuantTensor, ubCTensor, mmInfo, l0cPingPong);
            CopyOut1(ubCQuantTensor, mmInfo, l0cPingPong);
            mmInfo.updateIdx(nextMmInfo.mIdx, nextMmInfo.nIdx);
            l0cPingPong = 1 - l0cPingPong;
        }
        WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
        WAIT_FLAG(M, MTE1, EVENT_ID0);
        WAIT_FLAG(M, MTE1, EVENT_ID1);
        WAIT_FLAG(MTE3, V, EVENT_ID0);
        WAIT_FLAG(MTE3, V, EVENT_ID1);
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID1);
        WAIT_FLAG(V, MTE2, EVENT_ID2);
        WAIT_FLAG(V, MTE2, EVENT_ID3);
    }

    __aicore__ inline void CopyOut2(const matmulInfo &mmInfo)
    {
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);
        if constexpr (OUT_FORMAT == CubeFormat::ND) {
            mm2.CopyCo22GMNZ2ND(gmY, ubC2, ubTrans2, mmInfo.mIdx * mmInfo.baseM, mmInfo.nIdx * mmInfo.baseN,
                mmInfo.mActual, mmInfo.nActual, mmInfo.n);
        } else {
            int64_t dstOffset = static_cast<int64_t>(mmInfo.nIdx) * mmInfo.baseN * mOrgUp2 +
                static_cast<int64_t>(mmInfo.mIdx) * mmInfo.baseM * BLOCK_SIZE_16;
            ub_to_gm<ArchType::ASCEND_V200, OUT_DTYPE, DataFormat::NZ, DataFormat::NZ>(
                gmY[dstOffset], ubC2, mmInfo.mRound, mmInfo.mRound, mOrgUp2, mmInfo.nRound, mmInfo.nRound, 0);
        }
        SET_FLAG(MTE3, V, EVENT_ID0);
    }

    __aicore__ inline void Stage2Mm()
    {
        ubC2 = buf.GetBuffer<BufferType::ASCEND_UB, OUT_DTYPE>(UB_HALF_BUFFER_LEN);
        ubTrans2 = buf.GetBuffer<BufferType::ASCEND_UB, OUT_DTYPE>(
            baseN2 * sizeof(DESCALE_DTYPE) + baseN2 * sizeof(BIAS_DTYPE));
        SET_FLAG(MTE1, MTE2, EVENT_ID0);
        SET_FLAG(MTE1, MTE2, EVENT_ID1);
        SET_FLAG(MTE1, MTE2, EVENT_ID2);
        SET_FLAG(MTE1, MTE2, EVENT_ID3);
        SET_FLAG(M, MTE1, EVENT_ID0);
        SET_FLAG(M, MTE1, EVENT_ID1);
        SET_FLAG(MTE3, V, EVENT_ID0);
        SET_FLAG(V, MTE2, EVENT_ID0);
        SET_FLAG(V, MTE2, EVENT_ID2);
        matmulInfo mmInfo = {0, 0, m2, k2, n2, baseM2, baseK2, baseN2, m2Loops, k2Loops, n2Loops};
        for (uint32_t loopIdx = block_idx; loopIdx < coreLoops2; loopIdx += block_num) {
            GetIdx(loopIdx, mmInfo, swizzlDirect2, swizzlCnt2);
            mm2.LoadBias(gmBiasDq2, mmInfo, 0);
            mm2.BrcBias(mmInfo, 0);
            mm2.IterateAll(gmYTmp, gmW2, ubC2, gmScaleDq2, mmInfo, 0);
            CopyOut2(mmInfo);
        }
        WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
        WAIT_FLAG(M, MTE1, EVENT_ID0);
        WAIT_FLAG(M, MTE1, EVENT_ID1);
        WAIT_FLAG(MTE3, V, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID2);
    }

    AsdopsBuffer<ArchType::ASCEND_V200> buf;
    AscendC::GlobalTensor<IN_DTYPE> gmX;
    AscendC::GlobalTensor<IN_DTYPE> gmW1;
    AscendC::GlobalTensor<IN_DTYPE> gmW2;
    AscendC::GlobalTensor<ACCUMULATOR_DTYPE> gmScaleQuant;
    AscendC::GlobalTensor<DESCALE_DTYPE> gmScaleDq1;
    AscendC::GlobalTensor<DESCALE_DTYPE> gmScaleDq2;
    AscendC::GlobalTensor<ACCUMULATOR_DTYPE> gmBiasQuant;
    AscendC::GlobalTensor<BIAS_DTYPE> gmBiasDq1;
    AscendC::GlobalTensor<BIAS_DTYPE> gmBiasDq2;
    AscendC::GlobalTensor<OUT_DTYPE> gmY;
    AscendC::GlobalTensor<ACCUMULATOR_DTYPE> gmYAccu;
    AscendC::GlobalTensor<IN_DTYPE> gmYTmp;
    AscendC::GlobalTensor<int32_t> gmSync;

    AscendC::LocalTensor<ACCUMULATOR_DTYPE> ubC1{buf.GetBuffer<BufferType::ASCEND_UB, ACCUMULATOR_DTYPE>(0)};
    AscendC::LocalTensor<OUT_DTYPE> ubC2{buf.GetBuffer<BufferType::ASCEND_UB, OUT_DTYPE>(0)};
    AscendC::LocalTensor<IN_DTYPE> ubCQuant{buf.GetBuffer<BufferType::ASCEND_UB, IN_DTYPE>(0)};
    AscendC::LocalTensor<int32_t> ubSync{buf.GetBuffer<BufferType::ASCEND_UB, int32_t>(0)};
    AscendC::LocalTensor<ACCUMULATOR_DTYPE> ubScaleQuant{buf.GetBuffer<BufferType::ASCEND_UB, ACCUMULATOR_DTYPE>(0)};
    AscendC::LocalTensor<ACCUMULATOR_DTYPE> ubBiasQuant{buf.GetBuffer<BufferType::ASCEND_UB, ACCUMULATOR_DTYPE>(0)};
    AscendC::LocalTensor<ACCUMULATOR_DTYPE> ubTrans1{buf.GetBuffer<BufferType::ASCEND_UB, ACCUMULATOR_DTYPE>(0)};
    AscendC::LocalTensor<OUT_DTYPE> ubTrans2{buf.GetBuffer<BufferType::ASCEND_UB, OUT_DTYPE>(0)};
    AscendC::LocalTensor<uint8_t> ubGelu{buf.GetBuffer<BufferType::ASCEND_UB, uint8_t>(0)};
    MM1 mm1;
    MM2 mm2;
    
    uint32_t m1{0};
    uint32_t k1{0};
    uint32_t n1{0};
    uint32_t baseM1{0};
    uint32_t baseK1{0};
    uint32_t baseN1{0};
    uint32_t m1Loops{0};
    uint32_t k1Loops{0};
    uint32_t n1Loops{0};
    uint32_t coreLoops1{0};
    uint32_t swizzlCnt1{0};
    uint32_t swizzlDirect1{0};
    uint32_t mOrgUp1{0};
    uint32_t nOrgUp1{0};
    uint32_t mOrgUp2{0};
    uint32_t nOrgUp2{0};
    uint32_t m2{0};
    uint32_t k2{0};
    uint32_t n2{0};
    uint32_t baseM2{0};
    uint32_t baseK2{0};
    uint32_t baseN2{0};
    uint32_t m2Loops{0};
    uint32_t k2Loops{0};
    uint32_t n2Loops{0};
    uint32_t coreLoops2{0};
    uint32_t swizzlCnt2{0};
    uint32_t swizzlDirect2{0};
    uint32_t activationType{0};
};
} // namespace FFN
#endif