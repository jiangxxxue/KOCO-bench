/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ASCEND_OPS_FFN_MATMUL_H
#define ASCEND_OPS_FFN_MATMUL_H

#include "kernel_operator.h"
#include "ffn_common.h"
#include "kernels/utils/kernel/common.h"
#include "kernels/utils/kernel/common_func.h"
#include "kernels/utils/kernel/hardware.h"
#include "kernels/utils/kernel/simd.h"
#include "kernels/utils/kernel/iterator.h"
#include "kernels/utils/kernel/mma.h"
#include "kernels/utils/kernel/utils.h"

namespace FFN {
constexpr uint32_t L0AB_PINGPONG_BUFFER_LEN_INT8 = 32768;
constexpr uint32_t CUBE_MATRIX_SIZE_256 = 256;
constexpr uint32_t CUBE_MATRIX_SIZE_512 = 16 * 32;
constexpr uint32_t L1_PINGPONG_BUFFER_LEN_INT8 = 131072;
constexpr uint32_t L1_DESCALE_BUFFER_LEN = 40960;
constexpr uint32_t L1_BIAS_BUFFER_LEN = 20480;
constexpr uint32_t UB_HALF_BUFFER_LEN = 131072;
constexpr uint32_t MATRIX_C_PINGPONG_LEN = 256 * 128;

struct matmulInfo {
    uint32_t mIdx{0};
    uint32_t nIdx{0};
    uint32_t m{0};
    uint32_t k{0};
    uint32_t n{0};
    uint32_t baseM{0};
    uint32_t baseK{0};
    uint32_t baseN{0};
    uint32_t mLoops{0};
    uint32_t kLoops{0};
    uint32_t nLoops{0};
    uint32_t mActual{0};
    uint32_t nActual{0};
    uint32_t mRound{0};
    uint32_t nRound{0};
    __aicore__ inline void updateIdx(uint32_t newMIdx, uint32_t newNIdx)
    {
        mIdx = newMIdx;
        nIdx = newNIdx;
        mActual = (mIdx == (mLoops - 1)) ? (m - mIdx * baseM) : baseM;
        nActual = (nIdx == (nLoops - 1)) ? (n - nIdx * baseN) : baseN;
        mRound = RoundUp(mActual, BLOCK_SIZE_16);
        nRound = RoundUp(nActual, BLOCK_SIZE_16);
    }
};

template <typename DTYPE_A, CubeFormat FORMAT_A, typename DTYPE_B, CubeFormat FORMAT_B,
    typename DTYPE_C, CubeFormat FORMAT_C, typename DTYPE_ACCUMULATOR, typename DTYPE_BIAS,
    typename DTYPE_SCALE, bool TRANS_A, bool TRANS_B, bool WITH_BIAS, bool EN_L0C_DB
>
struct DefaultMatmul<ArchType::ASCEND_V200,
    DTYPE_A, FORMAT_A, DTYPE_B, FORMAT_B, DTYPE_C, FORMAT_C,
    DTYPE_ACCUMULATOR, DTYPE_BIAS, DTYPE_SCALE,
    TRANS_A, TRANS_B, WITH_BIAS, EN_L0C_DB> {
    __aicore__ explicit DefaultMatmul() {};

    __aicore__ inline void IterateAll(
        AscendC::GlobalTensor<DTYPE_A> refA, AscendC::GlobalTensor<DTYPE_B> refB, AscendC::LocalTensor<DTYPE_C> refC,
        AscendC::GlobalTensor<DTYPE_SCALE> refScale, const matmulInfo &mmInfo, const uint32_t &l0cPingPong)
    {
        AsdopsBuffer<ArchType::ASCEND_V200> buf;
        AscendC::LocalTensor<DTYPE_A> l1aPingTensor =
            buf.GetBuffer<BufferType::ASCEND_CB, DTYPE_A>(0);
        AscendC::LocalTensor<DTYPE_A> l1aPongTensor =
            buf.GetBuffer<BufferType::ASCEND_CB, DTYPE_A>(L1_PINGPONG_BUFFER_LEN_INT8);
        AscendC::LocalTensor<DTYPE_B> l1bPingTensor =
            buf.GetBuffer<BufferType::ASCEND_CB, DTYPE_B>(L1_PINGPONG_BUFFER_LEN_INT8 * CONST_2);
        AscendC::LocalTensor<DTYPE_B> l1bPongTensor =
            buf.GetBuffer<BufferType::ASCEND_CB, DTYPE_B>(L1_PINGPONG_BUFFER_LEN_INT8 * CONST_3);
        AscendC::LocalTensor<DTYPE_A> l0aPingTensor =
            buf.GetBuffer<BufferType::ASCEND_L0A, DTYPE_A>(0);
        AscendC::LocalTensor<DTYPE_A> l0aPongTensor =
            buf.GetBuffer<BufferType::ASCEND_L0A, DTYPE_A>(L0AB_PINGPONG_BUFFER_LEN_INT8);
        AscendC::LocalTensor<DTYPE_B> l0bPingTensor =
            buf.GetBuffer<BufferType::ASCEND_L0B, DTYPE_B>(0);
        AscendC::LocalTensor<DTYPE_B> l0bPongTensor =
            buf.GetBuffer<BufferType::ASCEND_L0B, DTYPE_B>(L0AB_PINGPONG_BUFFER_LEN_INT8);
        AscendC::LocalTensor<DTYPE_SCALE> ubScaleDequant =
            buf.GetBuffer<BufferType::ASCEND_UB, DTYPE_SCALE>(0);
        AscendC::LocalTensor<DTYPE_BIAS> l0cTensor =
            buf.GetBuffer<BufferType::ASCEND_UB, DTYPE_BIAS>(0);
        event_t l0cEvent;
        if constexpr (EN_L0C_DB) {
            l0cTensor = buf.GetBuffer<BufferType::ASCEND_L0C, DTYPE_BIAS>(
                l0cPingPong * MATRIX_C_PINGPONG_LEN * sizeof(float));
            l0cEvent = l0cPingPong ? EVENT_ID0 : EVENT_ID1;
        } else {
            l0cTensor = buf.GetBuffer<BufferType::ASCEND_L0C, DTYPE_BIAS>(0);
            l0cEvent = EVENT_ID0;
        }

        uint32_t kActual = (mmInfo.kLoops == 1) ? mmInfo.k : mmInfo.baseK;
        uint32_t kRound = RoundUp(kActual, BLOCK_SIZE_32);

        uint32_t mOrgUp = RoundUp(mmInfo.m, BLOCK_SIZE_16);
        uint32_t nOrgUp = RoundUp(mmInfo.n, BLOCK_SIZE_16);
        uint32_t kOrgUp = RoundUp(mmInfo.k, BLOCK_SIZE_32);

        AscendC::LocalTensor<DTYPE_A> l1aTensor = l1PingPong ? l1aPingTensor : l1aPongTensor;
        AscendC::LocalTensor<DTYPE_B> l1bTensor = l1PingPong ? l1bPingTensor : l1bPongTensor;
        event_t l1aEvent = l1PingPong ? EVENT_ID0 : EVENT_ID1;
        event_t l1bEvent = l1PingPong ? EVENT_ID2 : EVENT_ID3;
        WAIT_FLAG(MTE1, MTE2, l1aEvent);
        if constexpr (FORMAT_A == CubeFormat::NZ) {
            gm_to_l1<ArchType::ASCEND_V200, DTYPE_A, DataFormat::NZ, DataFormat::NZ>(
                l1aTensor, refA[mmInfo.mIdx * mmInfo.baseM * BLOCK_SIZE_32],
                mmInfo.mRound, mmInfo.mRound, mOrgUp, kRound, kRound, 0);
        } else {
            CopyND2NZOnTheFly(l1aTensor, refA, mmInfo.mIdx * mmInfo.baseM, 0, mmInfo.mActual, kActual, mmInfo.k);
        }
        SET_FLAG(MTE2, MTE1, l1aEvent);

        WAIT_FLAG(MTE1, MTE2, l1bEvent);
        if constexpr (FORMAT_B == CubeFormat::NZ) {
            gm_to_l1<ArchType::ASCEND_V200, DTYPE_B, DataFormat::NZ, DataFormat::NZ>(
                l1bTensor, refB[mmInfo.nIdx * mmInfo.baseN * BLOCK_SIZE_32],
                mmInfo.nRound, mmInfo.nRound, nOrgUp, kRound, kRound, 0);
        } else {
            CopyND2NZOnTheFly(l1bTensor, refB, mmInfo.nIdx * mmInfo.baseN, 0, mmInfo.nActual, kActual, mmInfo.k);
        }
        SET_FLAG(MTE2, MTE1, l1bEvent);

        uint32_t srcOffset = 0;
        uint32_t mnMax = mmInfo.mRound > mmInfo.nRound ? mmInfo.mRound : mmInfo.nRound;
        uint32_t kPartLen = L0AB_PINGPONG_BUFFER_LEN_INT8 / mnMax / BLOCK_SIZE_32 * BLOCK_SIZE_32;

        uint32_t lastKLoop = mmInfo.kLoops - 1;
        uint32_t offsetANext = 0;
        uint32_t offsetBNext = 0;
        uint32_t kActualNext = 0;
        uint32_t kRoundNext = 0;

        for (uint32_t kIdx = 0; kIdx < mmInfo.kLoops; ++kIdx) {
            kActual = (kIdx == mmInfo.kLoops - 1) ? mmInfo.k - kIdx * mmInfo.baseK : mmInfo.baseK;
            kRound = RoundUp(kActual, BLOCK_SIZE_32);

            AscendC::LocalTensor<DTYPE_A> l1aTensor = l1PingPong ? l1aPingTensor : l1aPongTensor;
            AscendC::LocalTensor<DTYPE_B> l1bTensor = l1PingPong ? l1bPingTensor : l1bPongTensor;

            event_t l1aEvent = l1PingPong ? EVENT_ID0 : EVENT_ID1;
            event_t l1bEvent = l1PingPong ? EVENT_ID2 : EVENT_ID3;

            if (kIdx < lastKLoop) {
                AscendC::LocalTensor<DTYPE_A> l1BufANextTensor = (1 - l1PingPong) ? l1aPingTensor : l1aPongTensor;
                AscendC::LocalTensor<DTYPE_B> l1BufBNextTensor = (1 - l1PingPong) ? l1bPingTensor : l1bPongTensor;
                event_t l1aEventNext = (1 - l1PingPong) ? EVENT_ID0 : EVENT_ID1;
                event_t l1bEventNext = (1 - l1PingPong) ? EVENT_ID2 : EVENT_ID3;

                uint32_t kActualNext =
                    ((kIdx + 1) == lastKLoop) ? (mmInfo.k - (kIdx + 1) * mmInfo.baseK) : mmInfo.baseK;
                uint32_t kRoundNext = RoundUp(kActualNext, BLOCK_SIZE_32);

                WAIT_FLAG(MTE1, MTE2, l1aEventNext);
                if constexpr (FORMAT_A == CubeFormat::NZ) {
                    offsetANext = (kIdx + 1) * mmInfo.baseK * mOrgUp + mmInfo.mIdx * mmInfo.baseM * BLOCK_SIZE_32;
                    gm_to_l1<ArchType::ASCEND_V200, DTYPE_A, DataFormat::NZ, DataFormat::NZ>(
                        l1BufANextTensor, refA[offsetANext],
                        mmInfo.mRound, mmInfo.mRound, mOrgUp, kRoundNext, kRoundNext, 0);
                } else {
                    CopyND2NZOnTheFly(l1BufANextTensor, refA, mmInfo.mIdx * mmInfo.baseM, (kIdx + 1) * mmInfo.baseK,
                        mmInfo.mActual, kActual, mmInfo.k);
                }
                SET_FLAG(MTE2, MTE1, l1aEventNext);

                WAIT_FLAG(MTE1, MTE2, l1bEventNext);
                if constexpr (FORMAT_B == CubeFormat::NZ) {
                    offsetBNext = (kIdx + 1) * mmInfo.baseK * nOrgUp + mmInfo.nIdx * mmInfo.baseN * BLOCK_SIZE_32;
                    gm_to_l1<ArchType::ASCEND_V200, DTYPE_B, DataFormat::NZ, DataFormat::NZ>(
                        l1BufBNextTensor, refB[offsetBNext],
                        mmInfo.nRound, mmInfo.nRound, nOrgUp, kRoundNext, kRoundNext, 0);
                } else {
                    CopyND2NZOnTheFly(l1BufBNextTensor, refB, mmInfo.nIdx * mmInfo.baseN, (kIdx + 1) * mmInfo.baseK,
                        mmInfo.nActual, kActual, mmInfo.k);
                }
                SET_FLAG(MTE2, MTE1, l1bEventNext);
            }

            uint32_t kPartLoops = CeilDiv(kActual, kPartLen);
            AscendC::PipeBarrier<PIPE_MTE2>();
            for (uint32_t kPartIdx = 0; kPartIdx < kPartLoops; ++kPartIdx) {
                uint32_t k0Round = kPartIdx < kPartLoops - 1 ? kPartLen : kRound - kPartIdx * kPartLen;
                uint32_t k0Actual = kPartIdx < kPartLoops - 1 ? kPartLen : kActual - kPartIdx * kPartLen;
                uint32_t l0PingPong = 1 - kPartIdx % 2;
                event_t l0Event = l0PingPong ? EVENT_ID0 : EVENT_ID1;
                AscendC::LocalTensor<DTYPE_A> l0aTensor = l0PingPong ? l0aPingTensor : l0aPongTensor;
                AscendC::LocalTensor<DTYPE_B> l0bTensor = l0PingPong ? l0bPingTensor : l0bPongTensor;
                if (kPartIdx == 0) {
                    WAIT_FLAG(MTE2, MTE1, l1aEvent);
                }
                WAIT_FLAG(M, MTE1, l0Event);
                l1_to_l0_a<ArchType::ASCEND_V200, DTYPE_A, false, DataFormat::ZN, DataFormat::ZZ>(
                    l0aTensor,
                    l1aTensor[kPartIdx * kPartLen * mmInfo.mRound],
                    mmInfo.mRound,
                    k0Round,
                    1,
                    mmInfo.mRound / 16,
                    k0Round / 32,
                    1);
                if (kPartIdx == kPartLoops - 1) {
                    SET_FLAG(MTE1, MTE2, l1aEvent);
                }
                if (kPartIdx == 0) {
                    WAIT_FLAG(MTE2, MTE1, l1bEvent);
                }

                srcOffset = kPartIdx * kPartLen * mmInfo.nRound;
                l1_to_l0_b<ArchType::ASCEND_V200, DTYPE_B, false, DataFormat::VECTOR, DataFormat::VECTOR>(
                    l0bTensor,                                      // dst
                    l1bTensor[srcOffset],                           // src
                    0,
                    mmInfo.nRound * k0Round / CUBE_MATRIX_SIZE_512, // repeat
                    0,
                    1,                                              // srcStride
                    0,
                    0);                                             // dstStride
                if (kPartIdx == kPartLoops - 1) {
                    SET_FLAG(MTE1, MTE2, l1bEvent);
                }
                uint32_t mMmadActual = (mmInfo.mActual == 1) ? CONST_2 : mmInfo.mActual;
                SET_FLAG(MTE1, M, l0Event);
                WAIT_FLAG(MTE1, M, l0Event);
                if (kIdx == 0 && kPartIdx == 0) {
                    WAIT_FLAG(V, M, l0cEvent);
                }
                AscendC::PipeBarrier<PIPE_M>();
                mmad<ArchType::ASCEND_V200, DTYPE_A, DTYPE_B, DTYPE_BIAS, false>(
                    l0cTensor,
                    l0aTensor,
                    l0bTensor,
                    mMmadActual,        // mmInfo.m
                    mmInfo.nActual,     // mmInfo.n
                    k0Actual,           // mmInfo.k
                    0);                 // cmatrixInitVal
                SET_FLAG(M, MTE1, l0Event);
            }
            l1PingPong = 1 - l1PingPong;
        }
        WAIT_FLAG(V, MTE2, EVENT_ID2);
        gm_to_ub<ArchType::ASCEND_V200, DTYPE_SCALE>(
            ubScaleDequant,
            refScale[mmInfo.nIdx * mmInfo.baseN],
            0,                                              // sid
            1,                                              // nBurst
            mmInfo.nRound / 4,                              // lenBurst
            0,                                              // srcStride
            0);                                             // dstStride
        SET_FLAG(MTE2, V, EVENT_ID2);
        WAIT_FLAG(MTE2, V, EVENT_ID2);
        SET_FLAG(M, V, l0cEvent);
        WAIT_FLAG(M, V, l0cEvent);
        WAIT_FLAG(MTE3, V, l0cEvent);
        l0c_to_ub<ArchType::ASCEND_V200, DTYPE_BIAS, DTYPE_C>(
            refC,
            l0cTensor,
            (uint16_t)(mmInfo.nRound / BLOCK_SIZE_16),      // nBurst
            (uint16_t)(mmInfo.mRound / BLOCK_SIZE_16),      // lenBurst
            (uint16_t)0,                                    // srcStride
            (uint16_t)0);                                   // dstStride
        SET_FLAG(V, MTE2, l0cEvent);
        SET_FLAG(V, MTE2, EVENT_ID2);
        if (mmInfo.mActual == 1) {
            DTYPE_C zero = 0;
            SetVectorMask<int8_t>((uint64_t)0x0, (uint64_t)0xffff);
            for (uint32_t i = 0; i < mmInfo.nRound / BLOCK_SIZE_16; i++) {
                uint64_t curr_offset_c = i * mmInfo.mRound * BLOCK_SIZE_16 + mmInfo.mActual * BLOCK_SIZE_16;
                muls_v<ArchType::ASCEND_V200, DTYPE_C>(refC[curr_offset_c], refC[curr_offset_c], zero, 1, 1, 1,
                        2, 2);
            }
        }
    }

    __aicore__ inline void LoadBias(AscendC::GlobalTensor<DTYPE_BIAS> refBias,
        const matmulInfo &mmInfo, const uint32_t &l0cPingPong)
    {
        AsdopsBuffer<ArchType::ASCEND_V200> buf;
        AscendC::LocalTensor<DTYPE_BIAS> ubBiasDequant;
        event_t l0cEvent;
        if constexpr (EN_L0C_DB) {
            ubBiasDequant = buf.GetBuffer<BufferType::ASCEND_UB, DTYPE_BIAS>(mmInfo.baseN * sizeof(DTYPE_SCALE) +
                l0cPingPong * mmInfo.baseN * (sizeof(DTYPE_SCALE) +sizeof(DTYPE_BIAS)));
            l0cEvent = l0cPingPong ? EVENT_ID0 : EVENT_ID1;
        } else {
            ubBiasDequant = buf.GetBuffer<BufferType::ASCEND_UB, DTYPE_BIAS>(mmInfo.baseN * sizeof(DTYPE_SCALE));
            l0cEvent = EVENT_ID0;
        }

        WAIT_FLAG(V, MTE2, l0cEvent);
        gm_to_ub<ArchType::ASCEND_V200, DTYPE_BIAS>(ubBiasDequant,
            refBias[mmInfo.nIdx * mmInfo.baseN],
            0,                                          // sid
            1,                                          // nBurst
            mmInfo.nRound / CONST_8,                    // lenBurst
            0,                                          // srcStride
            0);                                         // dstStride
        SET_FLAG(MTE2, V, l0cEvent);
    }

    __aicore__ inline void BrcBias(const matmulInfo &mmInfo, const uint32_t &l0cPingPong)
    {
        AsdopsBuffer<ArchType::ASCEND_V200> buf;
        AscendC::LocalTensor<DTYPE_BIAS> ubBiasDequant;
        AscendC::LocalTensor<DTYPE_BIAS> l0cTensor;
        event_t l0cEvent;
        if constexpr (EN_L0C_DB) {
            ubBiasDequant = buf.GetBuffer<BufferType::ASCEND_UB, DTYPE_BIAS>(
                mmInfo.baseN * sizeof(DTYPE_SCALE) + l0cPingPong * mmInfo.baseN *
                (sizeof(DTYPE_SCALE) + sizeof(DTYPE_BIAS)));
            l0cTensor = buf.GetBuffer<BufferType::ASCEND_L0C, DTYPE_BIAS>(
                l0cPingPong * MATRIX_C_PINGPONG_LEN * sizeof(float));
            l0cEvent = l0cPingPong ? EVENT_ID0 : EVENT_ID1;
        } else {
            ubBiasDequant = buf.GetBuffer<BufferType::ASCEND_UB, DTYPE_BIAS>(
                mmInfo.baseN * sizeof(DTYPE_SCALE));
            l0cTensor = buf.GetBuffer<BufferType::ASCEND_L0C, DTYPE_BIAS>(0);
            l0cEvent = EVENT_ID0;
        }
        
        WAIT_FLAG(MTE2, V, l0cEvent);
        for (uint32_t i = 0; i < mmInfo.mRound / BLOCK_SIZE_16; i++) {
            AscendC::BroadCastVecToMM(l0cTensor[i * CONST_256], ubBiasDequant,
                mmInfo.nRound / BLOCK_SIZE_16, 1, 0, mmInfo.mRound / BLOCK_SIZE_16 - 1);
        }
        SET_FLAG(V, M, l0cEvent);
    }

    template <typename DTYPE>
    __aicore__ inline void CopyND2NZOnTheFly(const AscendC::LocalTensor<DTYPE> &dst,
        const AscendC::GlobalTensor<DTYPE> &src, const uint32_t row,
        const uint32_t col, const uint32_t height, const uint32_t width, const uint32_t gCol)
    {
        const uint32_t c0Size = BLOCK_SIZE_32 / sizeof(DTYPE);
        uint32_t calcWidth = width / c0Size;
        int64_t dstOffset = 0;
        int64_t srcOffset = ((int64_t)row * (int64_t)gCol + (int64_t)col);
        uint32_t calcWidthExr = CeilDiv(width, c0Size);
        uint32_t calcHeightExr = CeilDiv(height, BLOCK_NUM_PER_FRACTAL);
        if (height % BLOCK_NUM_PER_FRACTAL != 0) {
            uint32_t repeat = calcWidthExr * calcHeightExr;
            AscendC::LocalTensor<int16_t> tmp = dst.template ReinterpretCast<int16_t>();
            AscendC::InitConstValueParams<int16_t> initConstValueParams;
            initConstValueParams.repeatTimes = (uint16_t)repeat;
            initConstValueParams.initValue = 0;
            InitConstValue(tmp, initConstValueParams);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        uint32_t srcGap = gCol * sizeof(DTYPE) / BLOCK_SIZE_32 - 1;
        if (gCol % c0Size != 0 || srcGap >= UINT16_MAX) {
            int64_t oriSrcOffset = srcOffset;
            int64_t oriDstOffset = dstOffset;
            for (uint32_t i = 0; i < calcWidth; i++) {
                for (uint32_t j = 0; j < height; j++) {
                    AscendC::DataCopy(dst[dstOffset], src[srcOffset], { 1, 1, 0, 0 });
                    dstOffset += c0Size;
                    srcOffset += gCol;
                }
                srcOffset = oriSrcOffset + (i + 1) * c0Size;
                dstOffset = oriDstOffset + (i + 1) * calcHeightExr * BLOCK_NUM_PER_FRACTAL * c0Size;
            }
        } else {
            for (uint32_t i = 0; i < calcWidth; i++) {
                AscendC::DataCopy(dst[dstOffset], src[srcOffset],
                    { static_cast<uint16_t>(height), 1, static_cast<uint16_t>(srcGap), 0 });
                    dstOffset += calcHeightExr * BLOCK_NUM_PER_FRACTAL * c0Size;
                    srcOffset += c0Size;
            }
        }
    }
    
    template <typename DTYPE>
    __aicore__ inline void TransNZ2ND(
        const AscendC::LocalTensor<DTYPE>& dst, const AscendC::LocalTensor<DTYPE>& src,
        const uint32_t blockHigh, const uint32_t blockWidth, const DTYPE scalar)
    {
        uint32_t blockCount = BLOCK_SIZE_32 / sizeof(DTYPE);
        ASCENDC_ASSERT((blockWidth <= MAX_REPEAT_LIMIT), {
            KERNEL_LOG(KERNEL_ERROR, "blockWidth is %d, blockCount is %d, repeat time exceed max time %d", blockWidth,
                blockCount, MAX_REPEAT_LIMIT);
        });
        struct AscendC::UnaryRepeatParams intriParams;

        int64_t dstOffset = 0;
        int64_t srcOffset = 0;
        uint32_t highBlock = MAX_REPEAT_LIMIT;
        uint32_t highBlocks = 0;
        uint32_t highTail = 0;
        uint32_t srcStride = highBlock * blockCount;
        uint32_t dstStride = blockWidth * blockCount * highBlock;
        bool isBeyondMaxStride = false;
        uint64_t mask[2] = {uint64_t(-1), uint64_t(-1)};

        intriParams.dstBlkStride = blockWidth;
        intriParams.srcBlkStride = 1;
        uint32_t dstRepStride = (blockWidth * blockCount * sizeof(DTYPE) / BLOCK_SIZE_32) * BLOCK_NUM_PER_VEC;
        intriParams.dstRepStride = dstRepStride;
        if (dstRepStride > 255) {
            isBeyondMaxStride = true;
        }
        intriParams.srcRepStride = (blockCount * sizeof(DTYPE) / BLOCK_SIZE_32) * BLOCK_NUM_PER_VEC;
        highBlocks = (blockHigh * blockCount) / BLOCK_NUM_PER_VEC / highBlock;
        highTail = (blockHigh * blockCount) / BLOCK_NUM_PER_VEC % highBlock;
        srcStride *= BLOCK_NUM_PER_VEC;
        dstStride *= BLOCK_NUM_PER_VEC;
        
        AscendC::SetVectorMask<DTYPE>(mask[1], mask[0]);
        for (uint32_t i = 0; i < blockWidth; i++) {
            int64_t dstMulsOffset = dstOffset;
            for (uint32_t j = 0; j < highBlocks; j++) {
                AscendC::Muls<DTYPE, false>(dst[dstMulsOffset], src[srcOffset], scalar, mask, highBlock, intriParams);
                srcOffset += srcStride;
                dstMulsOffset += dstStride;
            }
            if (highTail) {
                if (isBeyondMaxStride) {
                    uint32_t srcOffsetStride = blockCount * BLOCK_NUM_PER_VEC;
                    uint32_t dstOffsetStride = blockWidth * BLOCK_NUM_PER_FRACTAL * BLOCK_NUM_PER_VEC;
                    for (uint32_t j = 0; j < highTail; j++) {
                        AscendC::Muls<DTYPE, false>(dst[dstMulsOffset + j * dstOffsetStride],
                            src[srcOffset + j * srcOffsetStride], scalar, mask, 1, intriParams);
                    }
                } else {
                    AscendC::Muls<DTYPE, false>(
                        dst[dstMulsOffset], src[srcOffset], scalar, mask, highTail, intriParams);
                }
                srcOffset += highTail * blockCount * BLOCK_NUM_PER_VEC;
            }
            dstOffset += blockCount;
        }
        return;
    }

    template <typename DTYPE>
    __aicore__ inline void CopyCo22GMNZ2ND(const AscendC::GlobalTensor<DTYPE>& gmC,
        const AscendC::LocalTensor<DTYPE>& src, AscendC::LocalTensor<DTYPE> &trans,
        const uint32_t row, const uint32_t col, const uint32_t height, const uint32_t width,
        const uint32_t gCol)
    {
        const uint32_t blockCount = BLOCK_SIZE_32 / sizeof(DTYPE);
        uint32_t mRound = RoundUp(height, BLOCK_SIZE_16);
        uint32_t nRound = RoundUp(width, blockCount);
        bool isTragetAligned = (width % blockCount) == 0;
        bool isGmAligned = (gCol % blockCount) == 0;
        int dstStride = (gCol - nRound) * sizeof(DTYPE) / BLOCK_SIZE_32;
        int dstOffset = row * gCol + col;
        ASCENDC_ASSERT((gCol >= nRound),
            {KERNEL_LOG(KERNEL_ERROR, "n is %d, nRound is %d, n should be no less than nRound", gCol, nRound);});
        const bool isComputeLineByLine = (!isGmAligned || dstStride >= UINT16_MAX);

        SET_FLAG(MTE3, V, EVENT_ID2);
        WAIT_FLAG(MTE3, V, EVENT_ID2);
        TransNZ2ND(trans, src, mRound / BLOCK_NUM_PER_FRACTAL, nRound / blockCount, (DTYPE)1.0);
        SET_FLAG(V, MTE3, EVENT_ID2);
        WAIT_FLAG(V, MTE3, EVENT_ID2);

        int32_t blocklen = nRound / blockCount;
        if (isComputeLineByLine) {
            bool needPipe = gCol < BLOCK_NUM_PER_FRACTAL;
            for (int i = 0; i < height; ++i) {
                AscendC::DataCopy(gmC[dstOffset], trans[i * blocklen * BLOCK_SIZE_32 / sizeof(DTYPE)],
                    { 1, static_cast<uint16_t>(blocklen), 0, 0 });
                dstOffset += gCol;
                AscendC::PipeBarrier<PIPE_MTE3>();
            }
        } else {
            AscendC::DataCopy(gmC[dstOffset], trans,
                { static_cast<uint16_t>(height), static_cast<uint16_t>(blocklen), 0,
                static_cast<uint16_t>(dstStride) });
        }
    }
private:
    uint32_t l1PingPong{0};
};
} // namespace FFN
#endif