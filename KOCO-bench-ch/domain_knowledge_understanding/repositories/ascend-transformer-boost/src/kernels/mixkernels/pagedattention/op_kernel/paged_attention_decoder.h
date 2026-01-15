/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "kernels/utils/kernel/common.h"
#include "kernels/utils/kernel/common_func.h"
#include "kernels/utils/kernel/simd.h"
#include "kernels/utils/kernel/iterator.h"
#include "kernel_operator.h"

namespace {

constexpr uint32_t FLOAT_VECTOR_SIZE = 64; // float32
constexpr int32_t VECTOR_SIZE = 128;      // float16
constexpr int32_t BLOCK_SIZE = 16;
constexpr int32_t L0AB_HALF_BUF_SIZE = 16384; // 128 * 128 = 16K
constexpr int32_t CUBE_MATRIX_SIZE = 256;
constexpr int32_t LS_PINGPONG_SIZE = 4096; // 16 * 128
constexpr int32_t STRIDE_UPPER_BOUND = 65535;

constexpr int64_t L1_UINT8_BLOCK_SIZE = 131072; // 128K
constexpr int64_t UB_UINT8_BLOCK_SIZE = 32768;  // 128 * 128 * 2 = 32K
constexpr int64_t UB_UINT8_LINE_SIZE = 1024;
constexpr int64_t UB_FLOAT_LINE_SIZE = 256;
constexpr int64_t UB_HALF_LINE_SIZE = 512;
constexpr int64_t MAX_LEN_64_BYTES = 64;
constexpr int32_t DEC_UB_UINT8_BLOCK_SIZE = 8192; // 16 * 128 * 2
#if __CCE_AICORE__ == 100
constexpr int32_t TILING_HEAD_SIZE = 192;
#else
constexpr int32_t TILING_HEAD_SIZE = 128;
#endif
constexpr int32_t TILING_PARA_SIZE = 8;


} // namespace

enum class CalcMode {
    CALC_MODE_DEFAULT = 0,
    CALC_MODE_PREFILL = 1,
};

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
#else
    SET_FLAG(MTE1, MTE2, EVENT_ID4);
    SET_FLAG(MTE1, MTE2, EVENT_ID5);
    SET_FLAG(MTE1, MTE2, EVENT_ID6);
    SET_FLAG(MTE1, MTE2, EVENT_ID7);
#endif
}

__aicore__ inline void SyncEnd()
{
    WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
#if __CCE_AICORE__ == 100
#else
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
    PIPE_BARRIER(ALL);
}

__aicore__ inline void __set_mask(int32_t len)
{
    if (len >= VECTOR_SIZE) {
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        return;
    }
    int32_t highMask = len - MAX_LEN_64_BYTES > 0 ? len - MAX_LEN_64_BYTES : 0;
    int32_t lowMask = len - MAX_LEN_64_BYTES >= 0 ? MAX_LEN_64_BYTES : len;
    if (len < MAX_LEN_64_BYTES) {
        SetVectorMask<int8_t>(0x0, ((uint64_t)1 << lowMask) - 1);
        return;
    }
    SetVectorMask<int8_t>(((uint64_t)1 << highMask) - 1, 0xffffffffffffffff);
}

__aicore__ inline void __set_vcg_mask(int32_t len)
{
    if (len > BLOCK_SIZE) {
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        return;
    }
    uint64_t subMask = ((uint64_t)1 << len) - 1;
    uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask;
    SetVectorMask<int8_t>(maskValue, maskValue);
}

__aicore__ inline void ExpandToBlockHalf(AscendC::LocalTensor<half> dst_tensor,
    AscendC::LocalTensor<half> src_tensor, int32_t len)
{
    const uint32_t BLOCK_TWO = 2;
    const uint32_t BLOCK_NUM = 8;
    // (len,) -> len / 16 个 (16, 16)
    for (int32_t vaddsIdx = 0; vaddsIdx < BLOCK_TWO; ++vaddsIdx) {
        adds_v<ArchType::ASCEND_V200, half>(dst_tensor[vaddsIdx * BLOCK_NUM * BLOCK_SIZE], src_tensor, 0.0,
                                            len / BLOCK_SIZE, // repeat
                                            1, 0, BLOCK_TWO * BLOCK_NUM, 1);
    }
    PIPE_BARRIER(V);
    // (16, len) -> (len, 16)
    for (int32_t vtransIdx = 0; vtransIdx < (len / BLOCK_SIZE); ++vtransIdx) {
        tranpose_v<ArchType::ASCEND_V200, half>(dst_tensor[vtransIdx * CUBE_MATRIX_SIZE],
                                                dst_tensor[vtransIdx * CUBE_MATRIX_SIZE]);
    }
    PIPE_BARRIER(V);
}


template <CalcMode DECODE_MODE = CalcMode::CALC_MODE_DEFAULT>
class PagedAttentionDecoder {
public:
    __aicore__ inline PagedAttentionDecoder(__gm__ uint8_t *__restrict__ gmSrcq, __gm__ uint8_t *__restrict__ gmSrck,
                                            __gm__ uint8_t *__restrict__ gmSrcv, __gm__ uint8_t *__restrict__ gmSrcm,
                                            __gm__ uint8_t *__restrict__ gmDsto, half tor, uint32_t maxS,
                                            uint32_t ntokensQ, uint32_t blockSize, uint32_t maskStride)
        : gmSrcq(gmSrcq), gmSrck(gmSrck), gmSrcv(gmSrcv), gmSrcm(gmSrcm), gmDsto(gmDsto), tor(tor), maxS(maxS),
          ntokensQ(ntokensQ), blockSize(blockSize), maskStride(maskStride)
    {
        gmSrcq_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmSrcq));
        gmSrck_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmSrck));
        gmSrcv_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmSrcv));
        gmSrcm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmSrcm));
        gmDsto_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gmDsto));

        switch (DECODE_MODE) {
            case (CalcMode::CALC_MODE_PREFILL):{
                InitOffsetPrefill();
                break;
            }
            case (CalcMode::CALC_MODE_DEFAULT):{
            }
            default: {
                InitOffsetDefault();
            }
        }
        lsUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(lsUbuf_offset); // 32K * 2
        lpUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(lpUbuf_offset);
        // 2 for float32
        ls32Ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(ls32Ubuf_offset); // 32K * 4
        // 2 for UB memory offset
        loUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(loUbuf_offset);
        // 4 for UB memory offset
        lmUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(lmUbuf_offset); // 32K * 5
        // 4 for UB memory offset and 1 for local m memory offset(fp16)
        hmUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(hmUbuf_offset);
        // 4 for UB memory offset and 2 for hat m memory offset(fp16)
        gmUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(gmUbuf_offset);
        // 4 for UB memory offset and 3 for global m memory offset(fp16)
        dmUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(dmUbuf_offset);
        // 4 for UB memory offset and 5 for global m memory offset(fp32&fp16)
        llUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(llUbuf_offset);
        // 4 for UB memory offset and 7 for local l memory offset(fp32)
        glUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(glUbuf_offset);
        // 5 for UB memory offset, to save temp vector
        tvUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(tvUbuf_offset); // 32K * 6
        // 6 for UB memory offset, to save global O(fp32)
        goUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(goUbuf_offset);
        // 6 for UB memory offset, to save global O(fp32)
        maskUbuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, half>(maskUbuf_offset);
    }

    __aicore__ inline void InitOffsetPrefill()
    {
        lsUbuf_offset = 0;
        lpUbuf_offset = 0;
        ls32Ubuf_offset = 2 * UB_UINT8_BLOCK_SIZE; // 32K * 2
        loUbuf_offset = 2 * UB_UINT8_BLOCK_SIZE; // 32K * 2
        lmUbuf_offset = 4 * UB_UINT8_BLOCK_SIZE; // 32K * 4
        hmUbuf_offset = 4 * UB_UINT8_BLOCK_SIZE + 1 * UB_UINT8_LINE_SIZE; // 32K * 4 + 1K
        gmUbuf_offset = 4 * UB_UINT8_BLOCK_SIZE + 2 * UB_UINT8_LINE_SIZE; // 32K * 4 + 2K
        dmUbuf_offset = 4 * UB_UINT8_BLOCK_SIZE + 3 * UB_UINT8_LINE_SIZE; // 32K * 4 + 3K
        llUbuf_offset = 4 * UB_UINT8_BLOCK_SIZE + 5 * UB_UINT8_LINE_SIZE; // 32K * 4 + 5K
        glUbuf_offset = 4 * UB_UINT8_BLOCK_SIZE + 7 * UB_UINT8_LINE_SIZE; // 32K * 4 + 7K
        tvUbuf_offset = 5 * UB_UINT8_BLOCK_SIZE; // 32K * 5
        goUbuf_offset = 6 * UB_UINT8_BLOCK_SIZE; // 32K * 6
    }

    __aicore__ inline void InitOffsetDefault()
    {
        lsUbuf_offset = 0; // 8K * 2
        lpUbuf_offset = DEC_UB_UINT8_BLOCK_SIZE * 2; // 8K * 2
        ls32Ubuf_offset = 4 * DEC_UB_UINT8_BLOCK_SIZE; // 8K * 4
        maskUbuf_offset = 8 * DEC_UB_UINT8_BLOCK_SIZE; // 8K * 8
        loUbuf_offset = 10 * DEC_UB_UINT8_BLOCK_SIZE; // 8K * 10
        // 4 for UB memory offset
        lmUbuf_offset = 4 * UB_UINT8_BLOCK_SIZE; // 32K * 5
        // 4 for UB memory offset and 1 for local m memory offset(fp16)
        hmUbuf_offset = 4 * UB_UINT8_BLOCK_SIZE + 1 * UB_UINT8_LINE_SIZE;
        // 4 for UB memory offset and 2 for global m memory offset(fp16)
        dmUbuf_offset = 4 * UB_UINT8_BLOCK_SIZE + 2 * UB_UINT8_LINE_SIZE;
        // 4 for UB memory offset and 4 for global m memory offset(fp32&fp16)
        llUbuf_offset = 4 * UB_UINT8_BLOCK_SIZE + 4 * UB_UINT8_LINE_SIZE;
        // 4 for UB memory offset and 6 for hat m memory offset(fp16)
        gmUbuf_offset = 4 * UB_UINT8_BLOCK_SIZE + 6 * UB_UINT8_LINE_SIZE;
        // 5 for UB memory offset, to save temp vector
        tvUbuf_offset = 5 * UB_UINT8_BLOCK_SIZE; // 32K * 6
        // 5 for UB memory offset and 16 for local l memory offset(fp32)
        glUbuf_offset = 5 * UB_UINT8_BLOCK_SIZE + 16 * UB_UINT8_LINE_SIZE;
        // 6 for UB memory offset, to save global O(fp32)
        goUbuf_offset = 6 * UB_UINT8_BLOCK_SIZE; // 32K * (7~8)
    }

    __aicore__ inline void Init(int64_t srcqOffsetReal, int64_t srckOffsetReal, int64_t srcvOffsetReal,
                                int64_t srckOffsetReal1, int64_t srcvOffsetReal1, int64_t srcmOffsetReal,
                                int64_t dstoOffsetReal, int32_t initGReal, int32_t wrapOReal)
    {
        srcqOffset = srcqOffsetReal;
        srckOffset = srckOffsetReal;
        srcvOffset = srcvOffsetReal;
        srckOffset1 = srckOffsetReal1;
        srcvOffset1 = srcvOffsetReal1;
        srcmOffset = srcmOffsetReal;
        dstoOffset = dstoOffsetReal;

        initG = initGReal;
        wrapO = wrapOReal;
    }

public:
    __aicore__ inline void Decode(const int32_t fm, const int32_t fn, const int32_t fk, const int32_t bn,
                                  const int32_t m_actual, const int32_t n0_actual, const int32_t n1_actual,
                                  const uint32_t maskType, const uint32_t initKVE, const uint32_t headOffset = 0,
                                  const uint32_t initKV = 1, half localTor = 1, const uint32_t scaleType = 0);

    __aicore__ inline void DecodeParallel(const int32_t fm, const int32_t fn, const int32_t fk,
                                          const int32_t bn, const int32_t m_actual, const int32_t n0_actual,
                                          const int32_t n1_actual, const uint32_t maskType, const int32_t nIdx, const int32_t mask_n, const int32_t is_ping);

private:
    int32_t l1PingpongFlag = 0;
    int32_t ibPingpongFlag = 0;
    int32_t vmPingpongFlag = 1;

    __gm__ uint8_t *__restrict__ gmSrcq;
    __gm__ uint8_t *__restrict__ gmSrck;
    __gm__ uint8_t *__restrict__ gmSrcv;
    __gm__ uint8_t *__restrict__ gmSrcm;
    __gm__ uint8_t *__restrict__ gmDsto;
    AscendC::GlobalTensor<half> gmSrcq_tensor;
    AscendC::GlobalTensor<half> gmSrck_tensor;
    AscendC::GlobalTensor<half> gmSrcv_tensor;
    AscendC::GlobalTensor<half> gmSrcm_tensor;
    AscendC::GlobalTensor<half> gmDsto_tensor;

    AsdopsBuffer<ArchType::ASCEND_V200> buf;

    // L1 offsets
    uint32_t l1qBufAddr_offset = 0;
    uint32_t l1kBufAddr_offset = 2 * UB_UINT8_BLOCK_SIZE;
    uint32_t l1pBufAddr_offset = 2 * L1_UINT8_BLOCK_SIZE;
    uint32_t l1vBufAddr_offset = 2 * (L1_UINT8_BLOCK_SIZE + UB_UINT8_BLOCK_SIZE);
    uint32_t l1maskBufAddr_offset = 4 * L1_UINT8_BLOCK_SIZE;

    // L0 offsets
    uint32_t l0aBuf_offset = 0;
    uint32_t l0bBuf_offset = 0;
    uint32_t l0cBuf_offset = 0;

    // UB offsets
    uint32_t lsUbuf_offset = 0;
    uint32_t lpUbuf_offset = 0;
    uint32_t ls32Ubuf_offset = 0;
    uint32_t maskUbuf_offset = 0;
    uint32_t loUbuf_offset = 0;
    uint32_t lmUbuf_offset = 0;
    uint32_t hmUbuf_offset = 0;
    uint32_t gmUbuf_offset = 0;
    uint32_t dmUbuf_offset = 0;
    uint32_t llUbuf_offset = 0;
    uint32_t glUbuf_offset = 0;
    uint32_t tvUbuf_offset = 0;
    uint32_t goUbuf_offset = 0;

    AscendC::LocalTensor<half> l1qBufAddr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(l1qBufAddr_offset);
    AscendC::LocalTensor<half> l1kBufAddr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(l1kBufAddr_offset);
    AscendC::LocalTensor<half> l1pBufAddr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(l1pBufAddr_offset);
    AscendC::LocalTensor<half> l1vBufAddr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, half>(l1vBufAddr_offset);
    AscendC::LocalTensor<half> l1maskBufAddr_tensor =
        buf.GetBuffer<BufferType::ASCEND_CB, half>(l1maskBufAddr_offset);

    AscendC::LocalTensor<half> l0aBuf_tensor = buf.GetBuffer<BufferType::ASCEND_L0A, half>(l0aBuf_offset);
    AscendC::LocalTensor<half> l0bBuf_tensor = buf.GetBuffer<BufferType::ASCEND_L0B, half>(l0bBuf_offset);
    AscendC::LocalTensor<float> l0cBuf_tensor = buf.GetBuffer<BufferType::ASCEND_L0C, float>(l0cBuf_offset);

    AscendC::LocalTensor<half> lsUbuf_tensor;
    AscendC::LocalTensor<half> lpUbuf_tensor;
    AscendC::LocalTensor<float> ls32Ubuf_tensor;
    AscendC::LocalTensor<float> loUbuf_tensor;
    AscendC::LocalTensor<half> lmUbuf_tensor;
    AscendC::LocalTensor<half> hmUbuf_tensor;
    AscendC::LocalTensor<half> gmUbuf_tensor;
    AscendC::LocalTensor<half> dmUbuf_tensor;
    AscendC::LocalTensor<float> llUbuf_tensor;
    AscendC::LocalTensor<float> glUbuf_tensor;
    AscendC::LocalTensor<half> tvUbuf_tensor;
    AscendC::LocalTensor<float> goUbuf_tensor;
    AscendC::LocalTensor<half> maskUbuf_tensor;

    half tor = 1.0;
    uint32_t ntokensQ = 0;
    uint32_t maxS = 0;
    uint32_t blockSize = 128;

    int64_t srcqOffset = 0;
    int64_t srckOffset = 0;
    int64_t srcvOffset = 0;
    int64_t srckOffset1 = 0;
    int64_t srcvOffset1 = 0;
    int64_t dstoOffset = 0;
    int64_t srcmOffset = 0;
    int64_t maskStride = 0;

    int32_t initG = 0;
    int32_t wrapO = 0;
};
