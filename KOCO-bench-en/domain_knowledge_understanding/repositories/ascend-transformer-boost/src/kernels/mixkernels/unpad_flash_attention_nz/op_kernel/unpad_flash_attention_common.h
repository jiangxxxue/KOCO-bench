/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "kernels/utils/kernel/common.h"
#include "kernels/utils/kernel/iterator.h"
#ifndef UNPAD_FLASH_ATTENTION_COMMON_H
#define UNPAD_FLASH_ATTENTION_COMMON_H

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

#if __CCE_AICORE__ == 100
constexpr int32_t BATCH_TILING_OFFSET = 192;
#else
constexpr int32_t BATCH_TILING_OFFSET = 128;
#endif
constexpr int32_t LS_PINGPONG_SIZE = 4096;
constexpr int32_t TILING_PARA_SIZE = 20;
constexpr int32_t LOWER_MASK = 64;
constexpr int32_t FLOAT_VECTOR_SIZE = 64;
constexpr int32_t VECTOR_SIZE = 128;
constexpr int32_t BLOCK_SIZE = 16;
constexpr int32_t L0AB_HALF_BUF_SIZE = 16384;
constexpr int32_t UB_FLOAT_BUF_SIZE = 8192;
constexpr int32_t FLOAT_BLOCK_SIZE = 8;
constexpr int32_t CUBE_MATRIX_SIZE = 256;
constexpr int32_t BASE_MASK_SIZE = 128;
constexpr int32_t STRIDE_UPPER_BOUND = 65535;
constexpr int64_t L1_UINT8_BLOCK_SIZE = 131072;   // 128K
constexpr int64_t UB_UINT8_BLOCK_SIZE = 32768;    // 128 * 128 * 2 = 32K
constexpr int32_t DEC_UB_UINT8_BLOCK_SIZE = 8192; // 16 * 128 * 2
constexpr int64_t UB_UINT8_LINE_SIZE = 1024;
constexpr int64_t UB_FLOAT_LINE_SIZE = 256;
constexpr int64_t UB_HALF_LINE_SIZE = 512;
constexpr int32_t CUBE_UPDATE_O_ENABLED = 1;

enum AttentonMaskType {
    MASK_TYPE_NONE = 0,
    MASK_TYPE_NORM = 1,
    MASK_TYPE_ALIBI = 2,
    MASK_TYPE_LOOK_AHEAD = 3,
    MASK_TYPE_SWA_NORM = 4,
    MASK_TYPE_SWA_COMPRESS = 5
};
enum PrecType { BMM1_FP16_EXP_FP32 = 0, BMM1_FP32_EXP_FP32 = 1, BMM2_ONLINE_SOFTMAX_FP16 = 4 };
__aicore__ inline void SyncStart()
{
    SET_FLAG(M, MTE1, EVENT_ID0);
    SET_FLAG(M, MTE1, EVENT_ID1);
    SET_FLAG(M, MTE1, EVENT_ID2);
    SET_FLAG(M, MTE1, EVENT_ID3);
    SET_FLAG(V, M, EVENT_ID0);
    SET_FLAG(V, M, EVENT_ID1);
    SET_FLAG(V, M, EVENT_ID2);
    SET_FLAG(V, M, EVENT_ID3);
    SET_FLAG(V, MTE1, EVENT_ID0);
    SET_FLAG(V, MTE1, EVENT_ID1);
    SET_FLAG(MTE3, V, EVENT_ID0);
    SET_FLAG(MTE3, V, EVENT_ID1);
    SET_FLAG(MTE3, V, EVENT_ID2);
    SET_FLAG(MTE3, V, EVENT_ID3);
    SET_FLAG(MTE1, MTE3, EVENT_ID0);
    SET_FLAG(MTE1, MTE3, EVENT_ID1);

    SET_FLAG(MTE1, MTE2, EVENT_ID0);
    SET_FLAG(MTE1, MTE2, EVENT_ID1);
    SET_FLAG(MTE1, MTE2, EVENT_ID2);
    SET_FLAG(MTE1, MTE2, EVENT_ID3);
#if __CCE_AICORE__ == 100
    SET_FLAG(V, MTE2, EVENT_ID0);
#else
    SET_FLAG(MTE1, MTE2, EVENT_ID4);
    SET_FLAG(MTE1, MTE2, EVENT_ID5);
    SET_FLAG(MTE1, MTE2, EVENT_ID6);
    SET_FLAG(MTE1, MTE2, EVENT_ID7);
    SET_FLAG(V, MTE2, EVENT_ID6);
#endif
}

__aicore__ inline void SyncEnd()
{
    WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
#if __CCE_AICORE__ != 100
    WAIT_FLAG(MTE1, MTE2, EVENT_ID4);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID5);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID6);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID7);
#endif
    WAIT_FLAG(V, MTE1, EVENT_ID0);
    WAIT_FLAG(V, MTE1, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE3, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE3, EVENT_ID1);
    WAIT_FLAG(MTE3, V, EVENT_ID0);
    WAIT_FLAG(MTE3, V, EVENT_ID1);
    WAIT_FLAG(MTE3, V, EVENT_ID2);
    WAIT_FLAG(MTE3, V, EVENT_ID3);
    WAIT_FLAG(V, M, EVENT_ID0);
    WAIT_FLAG(V, M, EVENT_ID1);
    WAIT_FLAG(V, M, EVENT_ID2);
    WAIT_FLAG(V, M, EVENT_ID3);
    WAIT_FLAG(M, MTE1, EVENT_ID0);
    WAIT_FLAG(M, MTE1, EVENT_ID1);
    WAIT_FLAG(M, MTE1, EVENT_ID2);
    WAIT_FLAG(M, MTE1, EVENT_ID3);
#if __CCE_AICORE__ == 100
    WAIT_FLAG(V, MTE2, EVENT_ID0);
#else
    WAIT_FLAG(V, MTE2, EVENT_ID6);
#endif
    PIPE_BARRIER(ALL);
}

template <typename T, typename SType, PrecType prec_type1>
__aicore__ inline void SoftmaxExp(AscendC::LocalTensor<half> dst, AscendC::LocalTensor<SType> src,
                                  AscendC::LocalTensor<T> tmp, uint32_t pSize);

template <>
__aicore__ inline void
SoftmaxExp<float, float, PrecType::BMM1_FP32_EXP_FP32>(AscendC::LocalTensor<half> dst, AscendC::LocalTensor<float> src,
                                                       AscendC::LocalTensor<float> tmp, uint32_t pSize)
{
    ub_to_ub<ArchType::ASCEND_V200, float>(tmp, src, 0, 1, pSize / 8, 0, 0);
    PIPE_BARRIER(V);
    // 2 for Repeatimes above 255
    for (int32_t vexpIdx = 0; vexpIdx < 2; ++vexpIdx) {
        exp_v<ArchType::ASCEND_V200, float>(tmp[vexpIdx * pSize / 2], tmp[vexpIdx * pSize / 2],
                                            pSize / 2 / FLOAT_VECTOR_SIZE, // repeat
                                            1,                             // dstBlockStride
                                            1,                             // srcBlockStride
                                            8,                             // dstRepeatStride
                                            8                              // srcRepeatStride
        );
    }
    PIPE_BARRIER(V);
    // 2 for Repeatimes above 255
    for (int32_t vconvIdx = 0; vconvIdx < 2; ++vconvIdx) {
        conv_v<ArchType::ASCEND_V200, float, half>(dst[vconvIdx * pSize / 2], tmp[vconvIdx * pSize / 2],
                                                   pSize / 2 / FLOAT_VECTOR_SIZE, // repeat
                                                   1,                             // dstBlockStride
                                                   1,                             // srcBlockStride
                                                   uint16_t(4),                   // dstRepeatStride
                                                   uint16_t(8)                    // srcRepeatStride
        );
    }
}

template <typename T, PrecType prec_type2>
__aicore__ inline void UpdateExp(AscendC::LocalTensor<T> src, uint32_t repeat);

template <>
__aicore__ inline void UpdateExp<float, PrecType::BMM1_FP16_EXP_FP32>(AscendC::LocalTensor<float> src, uint32_t repeat)
{
    exp_v<ArchType::ASCEND_V200, float>(src, src, repeat, 1, 1, uint16_t(8), uint16_t(8));
}

template <>
__aicore__ inline void UpdateExp<float, PrecType::BMM1_FP32_EXP_FP32>(AscendC::LocalTensor<float> src, uint32_t repeat)
{
    exp_v<ArchType::ASCEND_V200, float>(src, src, repeat, 1, 1, uint16_t(8), uint16_t(8));
}

template <>
__aicore__ inline void UpdateExp<float, PrecType::BMM2_ONLINE_SOFTMAX_FP16>(AscendC::LocalTensor<float> src,
                                                                            uint32_t repeat)
{
    exp_v<ArchType::ASCEND_V200, float>(src, src, repeat, 1, 1, uint16_t(8), uint16_t(8));
}

// tensor type, tiling type, exp type, softmax type
template <typename T = half, typename SType = half, PrecType prec_type1 = PrecType::BMM1_FP16_EXP_FP32,
          PrecType prec_type2 = PrecType::BMM1_FP16_EXP_FP32>
class UnpadFlashAttentionCommon {
public:
    __aicore__ inline UnpadFlashAttentionCommon(
        __gm__ uint8_t *__restrict__ gmSrcq, __gm__ uint8_t *__restrict__ gmSrck, __gm__ uint8_t *__restrict__ gmSrcv,
        __gm__ uint8_t *__restrict__ gmSrcm, __gm__ uint8_t *__restrict__ gmSrcLayerid,
        __gm__ uint8_t *__restrict__ gmSrcAlibiCoeff, __gm__ uint8_t *__restrict__ gmSrcLogn,
        __gm__ uint8_t *__restrict__ gmDsto)
        : gmSrcq(gmSrcq), gmSrck(gmSrck), gmSrcv(gmSrcv), gmSrcm(gmSrcm), gmSrcLayerid(gmSrcLayerid),
          gmSrcAlibiCoeff(gmSrcAlibiCoeff), gmSrcLogn(gmSrcLogn), gmDsto(gmDsto)
    {
    }

    __aicore__ inline void Init(int32_t mReal, int32_t nReal, int32_t kReal, int64_t srcqOffsetReal,
                                int64_t srckOffsetReal, int64_t srcvOffsetReal, int64_t srcmOffsetReal0,
                                int64_t srcmOffsetReal1, int64_t dstoOffsetReal, int32_t initGReal, int32_t wrapOReal,
                                int32_t ntokensQReal, int32_t maskStrideReal, int64_t lognOffsetReal,
                                int32_t cubeUpdateSwitch = 0)
    {
        m = mReal;
        n = nReal;
        k = kReal;
        d = kReal;

        srcqOffset = srcqOffsetReal;
        srckOffset = srckOffsetReal;
        srcvOffset = srcvOffsetReal;
        srcmOffset0 = srcmOffsetReal0;
        srcmOffset1 = srcmOffsetReal1;
        dstoOffset = dstoOffsetReal;
        maskStride = maskStrideReal;

        initG = initGReal;
        wrapO = wrapOReal;
        ntokensQ = ntokensQReal;
        cubeUpdateO = cubeUpdateSwitch;

        lognOffset = lognOffsetReal;
    }

    __aicore__ inline void SetParams(half torIn, int32_t kvCopyStrideIn)
    {
        tor = torIn;
        kvCopyStride = kvCopyStrideIn;
    }
    __aicore__ inline void SetEncoderParams(SType torIn, int32_t kvCopyStrideIn, int32_t isSqrtIn, int32_t precTypeIn)
    {
        tor = torIn;
        kvCopyStride = kvCopyStrideIn;
        isSqrt = isSqrtIn;
        precType = precTypeIn;
        if (precType == PrecType::BMM1_FP32_EXP_FP32) {
            lpUbufSize = UB_FLOAT_BUF_SIZE;
        }
    }

    __aicore__ inline void InitKVgmBatchwise(uint64_t kBatchPtr, uint64_t vBatchPtr)
    {
        gmSrck = reinterpret_cast<__gm__ uint8_t *>(kBatchPtr);
        gmSrcv = reinterpret_cast<__gm__ uint8_t *>(vBatchPtr);
    }

    __aicore__ inline void InitDec(int32_t mReal, int32_t nReal, int32_t kReal, int64_t srcqOffsetReal,
                                   int64_t srckOffsetReal, int64_t srcvOffsetReal, int64_t srcmOffsetReal,
                                   int64_t dstoOffsetReal, int32_t initGReal, int32_t wrapOReal, int32_t ntokensQReal,
                                   int32_t maskStrideReal)
    {
        m = mReal;
        n = nReal;
        k = kReal;
        d = kReal;

        srcqOffset = srcqOffsetReal;
        srckOffset = srckOffsetReal;
        srcvOffset = srcvOffsetReal;
        srcmOffset0 = srcmOffsetReal;
        dstoOffset = dstoOffsetReal;
        maskStride = maskStrideReal;

        initG = initGReal;
        wrapO = wrapOReal;
        ntokensQ = ntokensQReal;

        const uint32_t lsUbuf_offset = 0;
        const uint32_t lpUbuf_offset = DEC_UB_UINT8_BLOCK_SIZE * 2;
        const uint32_t ls32Ubuf_offset = DEC_UB_UINT8_BLOCK_SIZE * 4;
        const uint32_t maskUbuf_offset = DEC_UB_UINT8_BLOCK_SIZE * 8;
        const uint32_t loUbuf_offset = DEC_UB_UINT8_BLOCK_SIZE * 10;

        lsUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(lsUbuf_offset);
        lpUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(lpUbuf_offset);
        ls32Ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(ls32Ubuf_offset);
        maskUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(maskUbuf_offset);
        loUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(loUbuf_offset);
    }

public:
    __aicore__ inline void FlashAttentionNzPrefillCompute(
        const int32_t fm, const int32_t fn, const int32_t fk, const int32_t bn, const int32_t __m0, const int32_t __n0,
        const int32_t __n1, const int32_t pp_n_scalar, const int32_t q_tight, const int32_t add_mask_n0,
        const int32_t add_mask_n1, const int32_t long_seq, const SType alibi_coeff, const SType delta0,
        const SType delta1, const uint32_t scale_type, const uint32_t alibi_left_align);
    __aicore__ inline void FlashAttentionNzDecoderCompute(const int32_t fm, const int32_t fn, const int32_t fk,
                                                          const int32_t bn, const int32_t __m0, const int32_t __n0,
                                                          const int32_t __n1, const int32_t pp_n_scalar, half local_tor,
                                                          const uint32_t scale_type);

public:
    __aicore__ inline void
    Run(AscendC::LocalTensor<int32_t> tiling_para_ub_tensor, AscendC::GlobalTensor<int32_t> tiling_para_gm_tensor,
        AscendC::LocalTensor<int32_t> layerId_ub_tensor, __gm__ uint8_t *__restrict__ alibi_coeff_gm,
        uint32_t mask_type, uint32_t window_len, uint32_t long_seq, uint64_t stride_qo, uint64_t stride_kv,
        int64_t head_mask_stride, int64_t batch_mask_stride, uint32_t start_batch, uint32_t end_batch,
        int32_t start_blk, int32_t end_blk, uint32_t is_triu, uint32_t alibi_compress_offset, int32_t group_num,
        uint32_t mask_stride, uint32_t q_tokens, int32_t embd, uint32_t q_tight, uint32_t scaleType, SType tor,
        int32_t kv_copy_stride, uint32_t is_sqrt, int64_t heads, uint32_t max_seqlen, uint32_t batch_size,
        int32_t kv_real_heads, const uint32_t alibi_left_align);
    __aicore__ inline void RowSum(const int32_t __n0, const int32_t fm, int32_t Pingflag);
    __aicore__ inline void SoftmaxUpdate(int32_t fm, int32_t fk, int32_t oSize, int32_t Pingflag, int32_t initGgO,
                                         int32_t mD64);
    __aicore__ inline void UpdateOutput(int32_t fm, int32_t fk, int32_t oSize, int32_t mD64, int32_t __m0);
    __aicore__ inline void UpdateFP32(const int32_t fm, const int32_t fk, const int32_t __n1, uint32_t initGgO);
    __aicore__ inline void UpdateFP16(const int32_t fm, const int32_t fk, const int32_t __n1, uint32_t initGgO);
    __aicore__ inline void SoftMax(const int32_t fm, const int32_t fn, const int32_t fk, const int32_t bn,
                                   const int32_t __m0, const int32_t __n0, const int32_t __n1,
                                   const int32_t add_mask_n0, const int32_t add_mask_n1, const SType alibi_coeff,
                                   const SType delta0, const SType delta1, const uint32_t scale_type,
                                   const uint32_t alibi_left_align, uint32_t initGgDm);

public:
    __aicore__ void __set_mask(int32_t len)
    {
        if (len >= 128) {
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        int32_t highMask = len - 64 > 0 ? len - 64 : 0;
        int32_t lowMask = len - 64 >= 0 ? 64 : len;
        if (len < 64) {
            SetVectorMask<int8_t>(0x0, ((uint64_t)1 << lowMask) - 1);
        } else {
            SetVectorMask<int8_t>(((uint64_t)1 << highMask) - 1, 0xffffffffffffffff);
        }
    }

    __aicore__ void __set_vcg_mask(int32_t len)
    {
        if (len > 16) {
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        uint64_t subMask = ((uint64_t)1 << len) - 1;
        uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask;
        SetVectorMask<int8_t>(maskValue, maskValue);
    }
    __aicore__ void __set_vcg_mask_fp32(int32_t len)
    {
        if (len > 8) {
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            return;
        }
        uint64_t subMask = ((uint64_t)1 << len) - 1;
        uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask + (subMask << 56) +
                             (subMask << 40) + (subMask << 24) + (subMask << 8);
        SetVectorMask<int8_t>(0x0, maskValue);
    }
    __aicore__ void ExpandToBlockHalf(AscendC::LocalTensor<half> dst_tensor, AscendC::LocalTensor<half> src_tensor,
                                      int32_t len)
    {
        for (int32_t vaddsIdx = 0; vaddsIdx < 2; ++vaddsIdx) { // (len,) -> len / 16 个 (16, 16)
            adds_v<ArchType::ASCEND_V200, half>(dst_tensor[vaddsIdx * 8 * BLOCK_SIZE], src_tensor, (half)(0.0),
                                                len / BLOCK_SIZE, // repeat
                                                1, 0, 16, 1);
        }
        PIPE_BARRIER(V);
        for (int32_t vtransIdx = 0; vtransIdx < (len / BLOCK_SIZE); ++vtransIdx) { // (16, len) -> (len, 16)
            tranpose_v<ArchType::ASCEND_V200, half>(dst_tensor[vtransIdx * CUBE_MATRIX_SIZE],
                                                    dst_tensor[vtransIdx * CUBE_MATRIX_SIZE]);
        }
        PIPE_BARRIER(V);
    }
    __aicore__ void ExpandToBlockFloat(AscendC::LocalTensor<float> dst_tensor, AscendC::LocalTensor<float> src_tensor,
                                       int32_t len)
    {
        for (int32_t rowIdx = 0; rowIdx < len; rowIdx++) {
            float scale = (float)*((__ubuf__ float *)src_tensor.GetPhyAddr() + rowIdx);
            SET_FLAG(S, V, EVENT_ID0);
            WAIT_FLAG(S, V, EVENT_ID0);
            PIPE_BARRIER(V);
            vector_dup((__ubuf__ float *)dst_tensor.GetPhyAddr() + rowIdx * 16, scale, 1, 1, 1, 8, 8);
        }
        PIPE_BARRIER(V);
    }

public:
    int32_t vmPingpongFlag = 1;

    __gm__ uint8_t *__restrict__ gmSrcq;
    __gm__ uint8_t *__restrict__ gmSrck;
    __gm__ uint8_t *__restrict__ gmSrcv;
    __gm__ uint8_t *__restrict__ gmSrcm;
    __gm__ uint8_t *__restrict__ gmSrcLayerid;
    __gm__ uint8_t *__restrict__ gmSrcAlibiCoeff;
    __gm__ uint8_t *__restrict__ gmSrcLogn;
    __gm__ uint8_t *__restrict__ gmDsto;

    const uint32_t lq_buf_offset = 0;
    const uint32_t lk_buf_offset = 2 * UB_UINT8_BLOCK_SIZE;
    const uint32_t lv_buf_offset = 2 * (L1_UINT8_BLOCK_SIZE + UB_UINT8_BLOCK_SIZE);
    const uint32_t lp_buf_offset = 2 * L1_UINT8_BLOCK_SIZE;
    const uint32_t lmask_buf_offset = 4 * L1_UINT8_BLOCK_SIZE;
    const uint32_t lalibi_coeff_buf_offset = 4 * (L1_UINT8_BLOCK_SIZE + UB_UINT8_BLOCK_SIZE);
    const uint32_t ldiag_buf_offset = 5 * L1_UINT8_BLOCK_SIZE; // 128 * 128 * 2(fp16) * 2(PingPong) = 64k
    const uint32_t lo_buf_offset = 6 * L1_UINT8_BLOCK_SIZE;    // 128(qSeqStep) * 128(embedDim) * 2(fp16) = 32k

    const uint32_t ls_ubuf_offset = 0;
    const uint32_t lp_ubuf_offset = 0;
    const uint32_t ls32_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE;
    const uint32_t lo_ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE;
    const uint32_t lm_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE;
    const uint32_t hm_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE + 1 * UB_UINT8_LINE_SIZE;
    const uint32_t gm_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE + 2 * UB_UINT8_LINE_SIZE;
    const uint32_t dm_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE + 3 * UB_UINT8_LINE_SIZE;
    const uint32_t ll_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE + 5 * UB_UINT8_LINE_SIZE;
    const uint32_t gl_ubuf_offset = 4 * UB_UINT8_BLOCK_SIZE + 7 * UB_UINT8_LINE_SIZE;
    const uint32_t tiling_para_ub_offset = 4 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_LINE_SIZE;
    const uint32_t logn_ub_offset = 4 * UB_UINT8_BLOCK_SIZE + 31 * UB_UINT8_LINE_SIZE;
    const uint32_t tv_ubuf_offset = 5 * UB_UINT8_BLOCK_SIZE;
    const uint32_t go_ubuf_offset = 6 * UB_UINT8_BLOCK_SIZE;
    const uint32_t mask_ubuf_offset = DEC_UB_UINT8_BLOCK_SIZE * 8;

    __cbuf__ uint8_t *l1qBufAddr;
    __cbuf__ uint8_t *l1kBufAddr;
    __cbuf__ uint8_t *l1pBufAddr;
    __cbuf__ uint8_t *l1vBufAddr;
    __cbuf__ uint8_t *l1maskBufAddr;

    AsdopsBuffer<ArchType::ASCEND_V200> buf;

    AscendC::LocalTensor<half> l1qBufAddr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(lq_buf_offset);
    AscendC::LocalTensor<half> l1kBufAddr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(lk_buf_offset);
    AscendC::LocalTensor<half> l1vBufAddr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(lv_buf_offset);
    AscendC::LocalTensor<half> l1pBufAddr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(lp_buf_offset);
    AscendC::LocalTensor<half> l1maskBufAddr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(lmask_buf_offset);
    AscendC::LocalTensor<half> l1diagBufAddr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(ldiag_buf_offset);
    AscendC::LocalTensor<half> l1oBufAddr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(lo_buf_offset);

    AscendC::LocalTensor<half> lsUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(ls_ubuf_offset);
    // diagUbuf_tensor复用lsUbuf_tensor空间，128*128*2*2=64k
    AscendC::LocalTensor<half> diagUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(ls_ubuf_offset);
    AscendC::LocalTensor<half> lpUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(lp_ubuf_offset);
    AscendC::LocalTensor<T> ls32Ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, T>(ls32_ubuf_offset);
    AscendC::LocalTensor<float> loUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(lo_ubuf_offset);
    AscendC::LocalTensor<half> lmUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(lm_ubuf_offset);
    AscendC::LocalTensor<half> hmUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(hm_ubuf_offset);
    AscendC::LocalTensor<half> gmUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(gm_ubuf_offset);
    AscendC::LocalTensor<half> dmUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(dm_ubuf_offset);
    AscendC::LocalTensor<T> llUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, T>(ll_ubuf_offset);
    AscendC::LocalTensor<T> glUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, T>(gl_ubuf_offset);
    AscendC::LocalTensor<half> tvUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(tv_ubuf_offset);
    AscendC::LocalTensor<T> goUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, T>(go_ubuf_offset);
    // toUbuf_tensor复用goUbuf_tensor空间，128*128*2=32k
    AscendC::LocalTensor<half> toUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(go_ubuf_offset);
    AscendC::LocalTensor<half> maskUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(mask_ubuf_offset);
    AscendC::LocalTensor<half> logn_ub_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(logn_ub_offset);

    AscendC::LocalTensor<half> l0aBuf_tensor = buf.GetBuffer<BufferType::ASCEND_L0A, half>(0);
    AscendC::LocalTensor<half> l0bBuf_tensor = buf.GetBuffer<BufferType::ASCEND_L0B, half>(0);
    AscendC::LocalTensor<float> l0cBuf_tensor = buf.GetBuffer<BufferType::ASCEND_L0C, float>(0);

    AscendC::GlobalTensor<half> gmSrcq_tensor;
    AscendC::GlobalTensor<half> gmSrck_tensor;
    AscendC::GlobalTensor<half> gmSrcv_tensor;
    AscendC::GlobalTensor<half> gmSrcm_tensor;
    AscendC::GlobalTensor<half> gmDsto_tensor;

    SType tor = 0;
    int32_t precType = 0;
    int32_t m = 0;
    int32_t n = 0;
    int32_t k = 0;
    int32_t d = 0;
    int32_t ntokensQ = 0;
    int32_t kvCopyStride = 0;
    int64_t srcqOffset = 0;
    int64_t srckOffset = 0;
    int64_t srcvOffset = 0;
    int64_t dstoOffset = 0;
    int64_t srcmOffset0 = 0;
    int64_t srcmOffset1 = 0;
    int64_t lognOffset = 0;
    int32_t maskStride = 0;
    int32_t initG = 0;
    int32_t wrapO = 0;
    int32_t isSqrt = 0;
    int32_t cubeUpdateO = 0;
    int32_t lpUbufSize = L0AB_HALF_BUF_SIZE;
};
#endif