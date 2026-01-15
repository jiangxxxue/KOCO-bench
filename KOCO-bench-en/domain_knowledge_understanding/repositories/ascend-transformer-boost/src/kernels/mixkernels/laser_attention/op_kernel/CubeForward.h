/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef __CUBEFORWARD_H__
#define __CUBEFORWARD_H__

#define USE_ASCENDC
#include "ppmatmul_const.h"
#include "kernels/utils/kernel/common.h"
#include "kernels/utils/kernel/utils.h"
#include "kernels/utils/kernel/iterator.h"

using namespace AscendC;
#define SET_FLAG(trigger, waiter, e) AscendC::SetFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define WAIT_FLAG(trigger, waiter, e) AscendC::WaitFlag<AscendC::HardEvent::trigger##_##waiter>((e))

#ifdef __DAV_C220_CUBE__

constexpr int32_t SIZE_16 = 16;
constexpr int32_t SIZE_32 = 32;
constexpr int32_t SIZE_64 = 64;
constexpr int32_t SIZE_128 = 128;
constexpr int32_t SIZE_256 = 256;
constexpr int32_t SIZE_384 = 384;
constexpr int32_t SIZE_504 = 504;
constexpr int32_t SIZE_ONE_K = 1024;
constexpr int64_t BASE_LEN = 128;
constexpr int64_t BASE_BLOCK = 128 * 128;
constexpr int64_t REPEAT_TIME_8 = 8;
constexpr int64_t REPEAT_TIME_64 = 64;
constexpr int64_t SRC_STRIDE_1 = 1;
constexpr int64_t SRC_STRIDE_8 = 8;

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
class CubeForward {
public:
    __aicore__ inline CubeForward(){};
    __aicore__ inline void Init(__gm__ uint8_t *__restrict__ a_cube1, __gm__ uint8_t *__restrict__ b_cube1,
        __gm__ uint8_t *__restrict__ b_cube2, __gm__ uint8_t *__restrict__ c_cube1, __gm__ float *__restrict__ c_cube2,
        __gm__ uint8_t *__restrict__ ones_rowsum, __gm__ float *__restrict__ gm_rowsum, int32_t Y, int32_t F, int32_t B,
        int32_t N, int32_t S1, int32_t S2, int32_t D, int32_t nG, int32_t isTriangle, int32_t sparseMode,
        int32_t windowLength);
    __aicore__ inline void Run();
    __aicore__ inline void PresetFlag();
    __aicore__ inline void ClearFlag();
    __aicore__ inline void SetHighPrecision(bool isHighPrecision)
    {
        this->isHighPrecision = isHighPrecision;
    };

protected:
    __aicore__ __inline__ void unified_addrMapping(
        int32_t roundId, PhyAddrCube1<TYPE, WORKSPACE_TYPE> *src, PhyAddrCube1<TYPE, WORKSPACE_TYPE> *remain);
    __aicore__ __inline__ void addrMapping(
        int32_t roundId, PhyAddrCube1<TYPE, WORKSPACE_TYPE> *src, PhyAddrCube1<TYPE, WORKSPACE_TYPE> *remain);
    __aicore__ __inline__ void addrMapping_nomask(
        int32_t roundId, PhyAddrCube1<TYPE, WORKSPACE_TYPE> *src, PhyAddrCube1<TYPE, WORKSPACE_TYPE> *remain);
    __aicore__ __inline__ void cube1(
        PhyAddrCube1<TYPE, WORKSPACE_TYPE> *src, PhyAddrCube1<TYPE, WORKSPACE_TYPE> *remain);
    __aicore__ __inline__ void base_block_mad(PhyAddrCube1<TYPE, WORKSPACE_TYPE> addr_1,
        PhyAddrCube1<TYPE, WORKSPACE_TYPE> addr_2, int32_t l0a_offset_remain = -1);

    // 以下方法用于混算：
    __aicore__ __inline__ void loadRowSumOnesblock();  // 加载rowsum右全1矩阵
    __aicore__ __inline__ void addrMapping_cube2_rowsum(int32_t roundId, int64_t &src_length, int64_t &remain_length,
        PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *src,
        PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *remain);  // 混算中cube2、rowsum寻址（倒三角+sparse）
    __aicore__ __inline__ void addrMapping_cube2_rowsum_nomask(int32_t roundId, int64_t &src_length,
        int64_t &remain_length, PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *src,
        PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *remain);  // 混算中cube2、rowsum寻址（nomask+不等长）
    __aicore__ __inline__ void unified_addrMapping_cube2mix(int32_t roundId, int64_t &src_length,
        int64_t &remain_length, PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *src,
        PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *remain);  // cube2和rowsum混算的寻址
    __aicore__ __inline__ void cube2_rowsum_mix_only(const PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *cube_addr,
        int32_t cube_length, int32_t m_length, int32_t vcore_num_per_head);  // cube2和row_sum混算的基本单元
    __aicore__ __inline__ void mix_cube2_rowsum(int32_t cube2_roundId);      // cube2、rowsum的混算

protected:
    __gm__ TYPE *__restrict__ gm_a_cube1;
    __gm__ TYPE *__restrict__ gm_b_cube1;
    __gm__ TYPE *__restrict__ gm_b_cube2;
    __gm__ WORKSPACE_TYPE *__restrict__ gm_c_cube1;
    __gm__ float *__restrict__ gm_c_cube2;
    __gm__ TYPE *__restrict__ gm_ones;
    __gm__ float *__restrict__ rowsum_gm;

    AsdopsBuffer<ArchType::ASCEND_V220> asdopsBuf;

    // init gm input tensor
    GlobalTensor<TYPE> gm_Q_tensor;
    GlobalTensor<TYPE> gm_K_tensor;
    GlobalTensor<TYPE> gm_V_tensor;
    GlobalTensor<WORKSPACE_TYPE> gm_c_cube1_tensor;
    GlobalTensor<TYPE> gm_c_cube1_tensor_type;
    GlobalTensor<float> gm_c_cube2_tensor;
    GlobalTensor<TYPE> gm_ones_tensor;
    GlobalTensor<float> gm_rowsum_tensor;

    LocalTensor<TYPE> l1_base_b_cube1_tensor;
    LocalTensor<TYPE> l1_base_a_cube2_tensor;
    LocalTensor<TYPE> l1_base_b_cube2_tensor;

    LocalTensor<TYPE> l1_a_ping_tensor;
    LocalTensor<TYPE> l1_a_pong_tensor;
    LocalTensor<TYPE> l1_b_ping_tensor;
    LocalTensor<TYPE> l1_b_pong_tensor;
    LocalTensor<TYPE> l1_row_sum_1_buf_tensor;

    LocalTensor<TYPE> l0_a_ping_tensor;
    LocalTensor<TYPE> l0_a_pong_tensor;
    LocalTensor<TYPE> l0_b_ping_tensor;
    LocalTensor<TYPE> l0_b_pong_tensor;
    LocalTensor<float> l0_c_ping_tensor;
    LocalTensor<float> l0_c_pong_tensor;

    uint32_t l1_a_ping_pong_flag_ = 0;
    uint32_t l1_b_ping_pong_flag_ = 0;

    uint32_t l0_a_ping_pong_flag_ = 0;
    uint32_t l0_b_ping_pong_flag_ = 0;
    uint32_t l0_c_ping_pong_flag_ = 0;

    // Y个core处理一个完整行，所有core分成 F 组， block_dim = F * Y
    int32_t Y;
    int32_t F;

    // Q K V shape : [B,N,S,D] 其中 D 固定为128
    int32_t B;
    int32_t N;
    // int32_t S; // 256 - 128k (256的倍数) 方阵的行
    int32_t S1;  // 256 - 128k (256的倍数) 方阵的行
    int32_t S2;  // 256 - 128k (256的倍数) 方阵的行
    int32_t D;
    int32_t nG;
    int32_t G;              // GQA 场景  k v shape [B,G,S,D]
    int32_t qk_triangle;    // GQA 场景  k v shape [B,G,S,D]
    int32_t sparseMode;     // sparseMode: 0:dense, 1:sparse
    int32_t windowLength;  // sparse场景下，滑动窗口的长度

    // int32_t H;
    int32_t H1;
    int32_t H2;
    int32_t L;  // 负责均衡
    int32_t W;

    int32_t curCoreIndex;
    int32_t local_block_index;  // 组内 core id [0, Y-1], 组内第几个 0
    int32_t core_group_index;   // [0, F-1], 第几组

    int32_t row_per_batch;
    int32_t column_per_core;
    int32_t column_remain;

    bool isHighPrecision = true;
};

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ inline void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::Init(__gm__ uint8_t *__restrict__ a_cube1,
    __gm__ uint8_t *__restrict__ b_cube1, __gm__ uint8_t *__restrict__ b_cube2, __gm__ uint8_t *__restrict__ c_cube1,
    __gm__ float *__restrict__ c_cube2, __gm__ uint8_t *__restrict__ ones_rowsum, __gm__ float *__restrict__ gm_rowsum,
    int32_t y_cube_num_per_line, int32_t group_num, int32_t batch, int32_t n_batch,
    // int32_t total_length,
    int32_t qSeqLength, int32_t kSeqLength, int32_t base_length, int32_t headGroupNum, int32_t isTriangle,
    int32_t sparseMode, int32_t windowLength)
{
    this->gm_a_cube1 = (__gm__ TYPE *__restrict__)a_cube1;
    this->gm_b_cube1 = (__gm__ TYPE *__restrict__)b_cube1;
    this->gm_b_cube2 = (__gm__ TYPE *__restrict__)b_cube2;
    this->gm_c_cube1 = (__gm__ WORKSPACE_TYPE *__restrict__)c_cube1;
    this->gm_c_cube2 = c_cube2;
    this->gm_ones = (__gm__ TYPE *__restrict__)ones_rowsum;
    this->rowsum_gm = gm_rowsum;

    this->Y = y_cube_num_per_line;
    this->F = group_num;
    this->B = batch;
    this->N = n_batch;
    this->S1 = qSeqLength;
    this->S2 = kSeqLength;
    this->D = base_length;
    this->nG = headGroupNum;
    this->G = this->N / this->nG;

    this->qk_triangle = isTriangle;
    this->sparseMode = sparseMode;
    this->windowLength = windowLength;

    // 无mask
    if (this->qk_triangle == 0) {
        this->H1 = this->S1 / SIZE_128;
        this->H2 = this->S2 / SIZE_128;
        this->L = this->H1;  // 负责均衡
        this->column_per_core = this->H2 / this->Y;
        this->column_remain = this->H2 % this->Y;
    }
    // 有mask
    else {
        this->H1 = this->S1 / SIZE_128;
        this->H2 = this->S2 / SIZE_128;
        this->L = this->H1 / 2;  // 负责均衡
        this->column_per_core = (this->H2 + 1) / this->Y;
        this->column_remain = (this->H2 + 1) % this->Y;

        // sparse场景：行数、列数以及尾块需要重新设置 update
        if (this->sparseMode == 1) {
            this->H1 = this->S1 / SIZE_128;
            this->H2 = this->S2 / SIZE_128;
            this->W = this->windowLength / SIZE_128;
            this->L = this->H1 - this->W / 2;
            this->column_per_core = (this->W + 1) / this->Y;
            this->column_remain = (this->W + 1) % this->Y;
        }
    }

    this->curCoreIndex = get_block_idx();
    this->core_group_index = this->curCoreIndex / this->Y;
    this->local_block_index = this->curCoreIndex % this->Y;
    this->row_per_batch = this->N * this->L;

    // init input gm tensor
    gm_Q_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(this->gm_a_cube1));
    gm_K_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(this->gm_b_cube1));
    gm_V_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(this->gm_b_cube2));

    // init gm output tensor
    gm_c_cube1_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ WORKSPACE_TYPE *>(this->gm_c_cube1));
    gm_c_cube1_tensor_type.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(this->gm_c_cube1));
    gm_c_cube2_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(this->gm_c_cube2));
    gm_ones_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(this->gm_ones));
    gm_rowsum_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(this->rowsum_gm));

    // init L1 tensor
    l1_base_b_cube1_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_128 * SIZE_ONE_K);
    l1_base_a_cube2_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_256 * SIZE_ONE_K);
    l1_base_b_cube2_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_384 * SIZE_ONE_K);

    l1_a_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(0);
    l1_a_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_128 * SIZE_ONE_K);
    l1_b_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_256 * SIZE_ONE_K);
    l1_b_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_384 * SIZE_ONE_K);
    l1_row_sum_1_buf_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_504 * SIZE_ONE_K);

    // init L0A/L0B/L0C tensor
    l0_a_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0A, TYPE>(0);
    l0_a_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0A, TYPE>(SIZE_32 * SIZE_ONE_K);
    l0_b_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0B, TYPE>(0);
    l0_b_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0B, TYPE>(SIZE_32 * SIZE_ONE_K);
    l0_c_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(0);
    l0_c_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(SIZE_64 * SIZE_ONE_K);
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ inline void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::PresetFlag()
{
    SET_FLAG(MTE1, MTE2, EVENT_ID0);
    SET_FLAG(MTE1, MTE2, EVENT_ID1);
    SET_FLAG(MTE1, MTE2, EVENT_ID2);
    SET_FLAG(MTE1, MTE2, EVENT_ID3);

    SET_FLAG(M, MTE1, EVENT_ID0);
    SET_FLAG(M, MTE1, EVENT_ID1);
    SET_FLAG(M, MTE1, EVENT_ID2);
    SET_FLAG(M, MTE1, EVENT_ID3);
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ inline void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::ClearFlag()
{
    WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID3);

    WAIT_FLAG(M, MTE1, EVENT_ID0);
    WAIT_FLAG(M, MTE1, EVENT_ID1);
    WAIT_FLAG(M, MTE1, EVENT_ID2);
    WAIT_FLAG(M, MTE1, EVENT_ID3);
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ inline void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::Run()
{
    SetPadding(0);
    uint64_t config = 0x1;
    AscendC::SetNdParaImpl(config);

    // 等待vector workspace的初始化
    WaitFlagDev(AIV2AICFLAGID);
    loadRowSumOnesblock();  // 先加载全1矩阵，使其常驻L1

    PresetFlag();
    // 计算每个核的总轮次 Z，  注意：存在余数需要计算当前核释放参与尾行的计算
    int32_t Z = B * N * L / F;
    int32_t remain = B * N * L % F;
    if (remain != 0) {
        Z += 1;
    }

    if (Z == 1) {
        if (core_group_index >= B * N * L) {
            if (Y > 1) {
                FftsCrossCoreSync<PIPE_FIX, 0>(AICFLAGID);
                WaitFlagDev(AICFLAGID);
            }
        } else {
            PhyAddrCube1<TYPE, WORKSPACE_TYPE> src[2];
            PhyAddrCube1<TYPE, WORKSPACE_TYPE> remain[2];
            unified_addrMapping(0, src, remain);
            cube1(src, remain);

            if (Y > 1) {
                FftsCrossCoreSync<PIPE_FIX, 0>(AICFLAGID); // 核间同步
                WaitFlagDev(AICFLAGID);
            }

            // aic sync aiv
            FftsCrossCoreSync<PIPE_FIX, 2>(AIC2AIVFLAGID);
            WaitFlagDev(AIV2AICFLAGID);

            // cube2与rowsum的混算
            mix_cube2_rowsum(0);
        }
    } else {
        int32_t cube2_index = 0;
        /************************************CUBE1*2******************************************************/
        for (int32_t i = 0; i < 2; i++) {
            if (i * F + core_group_index < B * N * L) {
                PhyAddrCube1<TYPE, WORKSPACE_TYPE> src[2];
                PhyAddrCube1<TYPE, WORKSPACE_TYPE> remain[2];
                unified_addrMapping(i, src, remain);
                cube1(src, remain);

                if (Y > 1) {
                    FftsCrossCoreSync<PIPE_FIX, 0>(AICFLAGID);
                    WaitFlagDev(AICFLAGID);
                }
                if (i == 0) {
                    FftsCrossCoreSync<PIPE_FIX, 2>(AIC2AIVFLAGID);
                }
            } else if (i == (Z - 1)) {
                if (Y > 1) {
                    FftsCrossCoreSync<PIPE_FIX, 0>(AICFLAGID);
                    WaitFlagDev(AICFLAGID);
                }
            }
        }
        /****************************************** CUBE2 + CUBE1 ***************************************/
        for (int32_t i = 2; i < Z; i++) {
            WaitFlagDev(AIV2AICFLAGID);
            FftsCrossCoreSync<PIPE_FIX, 2>(AIC2AIVFLAGID);

            // cube2与rowsum的混算
            mix_cube2_rowsum(cube2_index);
            cube2_index += 1;

            if (i * F + core_group_index < B * N * L) {
                PhyAddrCube1<TYPE, WORKSPACE_TYPE> src[2];
                PhyAddrCube1<TYPE, WORKSPACE_TYPE> remain[2];
                unified_addrMapping(i, src, remain);
                cube1(src, remain);

                // Wait Vec
                // cube2 load some data to l1

                // 多核处理一个完整行，需要全核同步
                if (Y > 1) {
                    FftsCrossCoreSync<PIPE_FIX, 0>(AICFLAGID);
                    WaitFlagDev(AICFLAGID);
                }
            } else {
                if (Y > 1) {
                    FftsCrossCoreSync<PIPE_FIX, 0>(AICFLAGID);
                    WaitFlagDev(AICFLAGID);
                }
            }
        }
        /************************************CUBE2*2******************************************************/
        for (int32_t i = 0; i < 2; i++) {
            if ((Z - 2 + i) * F + core_group_index < B * N * L) {
                WaitFlagDev(AIV2AICFLAGID);
                if (i == 0) {
                    FftsCrossCoreSync<PIPE_FIX, 2>(AIC2AIVFLAGID);
                }

                // cube2与rowsum的混算
                mix_cube2_rowsum(cube2_index);
                cube2_index += 1;
            }
        }
    }
    FftsCrossCoreSync<PIPE_FIX, 0>(AICFLAGID);
    WaitFlagDev(AICFLAGID);


    FftsCrossCoreSync<PIPE_FIX, 2>(AIC2AIVFLAGID);

    ClearFlag();
}

// --------------------------------------
// [+] Unified addressing function entrance
// @unified_addrMapping
// @unified_addrMapping_cube2
// @unified_addrMapping_rowsum
template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ __inline__ void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::unified_addrMapping(
    int32_t roundId, PhyAddrCube1<TYPE, WORKSPACE_TYPE> *src, PhyAddrCube1<TYPE, WORKSPACE_TYPE> *remain)
{
    if (this->qk_triangle == 1) {  // 倒三角
        addrMapping(roundId, src, remain);
    } else if (this->qk_triangle == 0) {  // 非倒三角
        addrMapping_nomask(roundId, src, remain);
    }
}

// --------------------------------------
// [+] No mask cases + qk不等长
// L = H
// column_per_core = H2 / Y
// column_remain = H2 % Y
// --------------------------------------
template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ __inline__ void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::addrMapping_nomask(
    int32_t roundId, PhyAddrCube1<TYPE, WORKSPACE_TYPE> *src, PhyAddrCube1<TYPE, WORKSPACE_TYPE> *remain)
{
    int32_t curCalcRow = roundId * F + core_group_index;  // 当前的row
    int32_t curCalcRowP = F * (roundId % 2) + core_group_index;
    int32_t b = curCalcRow / row_per_batch;               // 当前行所在的batch号
    int32_t n = (curCalcRow % row_per_batch) / L;         // 当前batch下的head号
    int32_t i_r = (curCalcRow % row_per_batch) % L;       // 当前head下的行号
    int32_t i_c =  local_block_index * column_per_core ;    // 当前core处理的起始列
    int32_t begin_remian = H2 - column_remain;           // MHA + 无mask

    int32_t q_left_index = i_r;           // MHA + 无mask
    int32_t q_right_index = L - 1 - i_r;  // MHA + 无mask
    int32_t switch_index = H2;            // MHA + 无mask

    int64_t bn_offset_1 = (b * N * H1 + n * H1) * BASE_BLOCK_SIZE;
    int32_t g = n / (N / G);
    int64_t bn_offset_GQA = (b * G * H2 + g * H2) * BASE_BLOCK_SIZE;
    int64_t out_offset = (curCalcRowP * H2) * BASE_BLOCK_SIZE;

    src[0].left = gm_a_cube1 + bn_offset_1 + q_left_index * BASE_BLOCK_SIZE;
    src[0].right = gm_b_cube1 + bn_offset_GQA + i_c * BASE_BLOCK_SIZE;
    src[0].out = gm_c_cube1 + out_offset + i_c * BASE_BLOCK_SIZE;
    src[0].k = column_per_core;

    if (column_remain != 0) {
        remain[0].left = gm_a_cube1 + bn_offset_1 + q_left_index * BASE_BLOCK_SIZE;
        remain[0].right = gm_b_cube1 + bn_offset_GQA + begin_remian * BASE_BLOCK_SIZE;
        remain[0].out = gm_c_cube1 + out_offset + begin_remian * BASE_BLOCK_SIZE;
        remain[0].k = column_remain;
    }
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ __inline__ void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::addrMapping(
    int32_t roundId, PhyAddrCube1<TYPE, WORKSPACE_TYPE> *src, PhyAddrCube1<TYPE, WORKSPACE_TYPE> *remain)
{
    int32_t curCalcRow = roundId * F + core_group_index;  // 当前的row
    int32_t curCalcRowP = F * (roundId % 2) + core_group_index;
    int32_t b = curCalcRow / row_per_batch;           // 当前行所在的batch号
    int32_t n = (curCalcRow % row_per_batch) / L;     // 当前batch下的head号
    int32_t i_r = (curCalcRow % row_per_batch) % L;   // 当前head下的行号
    int32_t i_c = local_block_index * column_per_core;  // 当前core处理的起始列
    int32_t begin_remain = this->sparseMode == 1 ? (W + 1 - column_remain) : (H1 + 1 - column_remain);  // update

    int32_t q_left_index = this->sparseMode == 1 ? (W / 2 + i_r) : (L + i_r);
    int32_t q_right_index = this->sparseMode == 1 ? (W / 2 - 1 - i_r) : (L - 1 - i_r);
    int32_t switch_index = q_left_index;

    // 设置sparse的标志位 以及 sparse场景非倒三角部分的偏移
    int32_t sparse_flag = false;
    if (this->sparseMode == 1) {
        sparse_flag = i_r > ((W - 1) / 2) ? true : false;
    }
    int32_t row_down = i_r + W / 2;
    int32_t col_down = i_r + i_c - W / 2;
    int32_t col_down_remain = i_r + begin_remain - W / 2;

    int64_t bn_offset = (b * N * H1 + n * H1) * BASE_BLOCK_SIZE;
    int32_t g = n / (N / G);
    int64_t bn_offset_GQA = (b * G * H1 + g * H1) * BASE_BLOCK_SIZE;
    int64_t q_left_offset = bn_offset + q_left_index * BASE_BLOCK_SIZE;
    int64_t q_right_offset = bn_offset + q_right_index * BASE_BLOCK_SIZE;
    // int64_t out_offset = (curCalcRow * (H + 1)) * BASE_BLOCK_SIZE;
    // workspace的大小(F * (H + 1) * BASE_BLOCK_SIZE) * (roundId % 2) + core_group_index * (H + 1) * BASE_BLOCK_SIZE
    // 反向workspace的大小  coreNum * len * BASE_BLOCK_SIZE,     curCoreIndex * len * BASE_BLOCK_SIZE
    int64_t out_offset = this->sparseMode == 1 ? ((curCalcRowP * (W + 1)) * BASE_BLOCK_SIZE)
                                               : ((curCalcRowP * (H1 + 1)) * BASE_BLOCK_SIZE);

    if (!sparse_flag && switch_index < i_c) {
        src[0].left = gm_a_cube1 + bn_offset + q_right_index * BASE_BLOCK_SIZE;
        src[0].right = gm_b_cube1 + bn_offset_GQA + (i_c - switch_index - 1) * BASE_BLOCK_SIZE;
        src[0].out = gm_c_cube1 + out_offset + i_c * BASE_BLOCK_SIZE;
        src[0].k = column_per_core;

        if (column_remain != 0) {
            remain[0].left = src[0].left;
            remain[0].right = gm_b_cube1 + bn_offset_GQA + (begin_remain - switch_index - 1) * BASE_BLOCK_SIZE;
            remain[0].out = gm_c_cube1 + out_offset + begin_remain * BASE_BLOCK_SIZE;
            remain[0].k = column_remain;
        }
    } else if (!sparse_flag && i_c <= switch_index && i_c + column_per_core > switch_index) {
        src[0].left = gm_a_cube1 + bn_offset + q_left_index * BASE_BLOCK_SIZE;
        src[0].right = gm_b_cube1 + bn_offset_GQA + i_c * BASE_BLOCK_SIZE;
        src[0].out = gm_c_cube1 + out_offset + i_c * BASE_BLOCK_SIZE;
        src[0].k = switch_index - i_c + 1;

        src[1].left = gm_a_cube1 + bn_offset + q_right_index * BASE_BLOCK_SIZE;
        src[1].right = gm_b_cube1 + bn_offset_GQA;
        src[1].out = src[0].out + src[0].k * BASE_BLOCK_SIZE;
        src[1].k = column_per_core - src[0].k;

        if (column_remain != 0) {
            remain[0].left = src[1].left;
            remain[0].right = gm_b_cube1 + bn_offset_GQA + (begin_remain - switch_index - 1) * BASE_BLOCK_SIZE;
            remain[0].out = gm_c_cube1 + out_offset + begin_remain * BASE_BLOCK_SIZE;
            remain[0].k = column_remain;
        }
    } else if (!sparse_flag && i_c <= switch_index && i_c + column_per_core <= switch_index) {
        src[0].left = gm_a_cube1 + bn_offset + q_left_index * BASE_BLOCK_SIZE;
        src[0].right = gm_b_cube1 + bn_offset_GQA + i_c * BASE_BLOCK_SIZE;
        src[0].out = gm_c_cube1 + out_offset + i_c * BASE_BLOCK_SIZE;
        src[0].k = column_per_core;

        if (column_remain != 0) {
            if (switch_index < begin_remain) {
                remain[0].left = gm_a_cube1 + bn_offset + q_right_index * BASE_BLOCK_SIZE;
                remain[0].right = gm_b_cube1 + bn_offset_GQA + (begin_remain - switch_index - 1) * BASE_BLOCK_SIZE;
                remain[0].out = gm_c_cube1 + out_offset + begin_remain * BASE_BLOCK_SIZE;
                remain[0].k = column_remain;
            } else {
                remain[0].left = gm_a_cube1 + bn_offset + q_left_index * BASE_BLOCK_SIZE;
                remain[0].right = gm_b_cube1 + bn_offset_GQA + begin_remain * BASE_BLOCK_SIZE;
                remain[0].out = gm_c_cube1 + out_offset + begin_remain * BASE_BLOCK_SIZE;
                remain[0].k = switch_index - begin_remain + 1;

                remain[1].left = gm_a_cube1 + bn_offset + q_right_index * BASE_BLOCK_SIZE;
                remain[1].right = gm_b_cube1 + bn_offset_GQA;
                remain[1].out = remain[0].out + remain[0].k * BASE_BLOCK_SIZE;
                remain[1].k = column_remain - remain[0].k;
            }
        }
    } else {
        src[0].left = gm_a_cube1 + bn_offset + row_down * BASE_BLOCK_SIZE;
        src[0].right = gm_b_cube1 + bn_offset_GQA + col_down * BASE_BLOCK_SIZE;
        src[0].out = gm_c_cube1 + out_offset + i_c * BASE_BLOCK_SIZE;
        src[0].k = column_per_core;

        if (column_remain != 0) {
            remain[0].left = src[0].left;
            remain[0].right = gm_b_cube1 + bn_offset_GQA + col_down_remain * BASE_BLOCK_SIZE;
            remain[0].out = gm_c_cube1 + out_offset + begin_remain * BASE_BLOCK_SIZE;
            remain[0].k = column_remain;
        }
    }
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ __inline__ void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::cube1(
    PhyAddrCube1<TYPE, WORKSPACE_TYPE> *src, PhyAddrCube1<TYPE, WORKSPACE_TYPE> *remain)
{
    int32_t switch_flag = 0;
    int32_t l0a_offset_remain = 0;

    auto l1_base_a_cube1_tensor = l1_a_ping_pong_flag_ ? l1_a_pong_tensor : l1_a_ping_tensor;
    WAIT_FLAG(MTE1, MTE2, l1_a_ping_pong_flag_);

    auto commonNd2NzParams = AscendC::Nd2NzParams(
        1,            // ndNum
        BASE_LEN,     // nValue
        BASE_LEN,     // dValue
        0,            // srcNdMatrixStride, unused
        BASE_LEN,     // srcDValue
        BASE_LEN,     // dstNzC0Stride
        1,            // dstNzNStride
        0             // dstNzMatrixStride, unused
    );
    for (int i = 0; i < 2; i++) {
        if (src[i].k == 0) {
            continue;
        }
        AscendC::DataCopy(
            l1_base_a_cube1_tensor[BASE_BLOCK_SIZE * i],
            this->gm_Q_tensor[(uint32_t(src[i].left - this->gm_a_cube1))],
            commonNd2NzParams
        );
        switch_flag++;
    }

    if (switch_flag == 2) {
        l0a_offset_remain = BASE_BLOCK_SIZE;
    } else if (src[0].k == 0 && src[1].k == 0) {
        for (int i = 0; i < 2; i++) {
            if (remain[i].k == 0) {
                continue;
            }
            AscendC::DataCopy(
                l1_base_a_cube1_tensor[BASE_BLOCK_SIZE * i],
                this->gm_Q_tensor[(uint32_t(remain[i].left - this->gm_a_cube1))],
                commonNd2NzParams
            );
            switch_flag++;
        }
    } else if (remain[0].k != 0 && src[0].left != remain[0].left) {
        l0a_offset_remain = BASE_BLOCK_SIZE;
        AscendC::DataCopy(
            l1_base_a_cube1_tensor[BASE_BLOCK_SIZE],
            this->gm_Q_tensor[(uint32_t(remain[0].left - this->gm_a_cube1))],
            commonNd2NzParams
        );
        switch_flag++;
    } else if (remain[1].k != 0) {
        AscendC::DataCopy(
            l1_base_a_cube1_tensor[BASE_BLOCK_SIZE],
            this->gm_Q_tensor[(uint32_t(remain[1].left - this->gm_a_cube1))],
            commonNd2NzParams
        );
        switch_flag++;
    }

    SET_FLAG(MTE2, MTE1, l1_a_ping_pong_flag_);
    WAIT_FLAG(MTE2, MTE1, l1_a_ping_pong_flag_);

    // 处理完整基本块
    base_block_mad(src[0], src[1]);
    // 处理尾块
    if (remain[0].k + remain[1].k != 0) {
        base_block_mad(remain[0], remain[1], l0a_offset_remain);
    }

    SET_FLAG(MTE1, MTE2, l1_a_ping_pong_flag_);
    l1_a_ping_pong_flag_ = 1 - l1_a_ping_pong_flag_;
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ __inline__ void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::base_block_mad(
    PhyAddrCube1<TYPE, WORKSPACE_TYPE> addr_1, PhyAddrCube1<TYPE, WORKSPACE_TYPE> addr_2,
     int32_t l0a_offset_remain)
{
    auto remain_offset = (l0a_offset_remain == -1) ? 0 : 128 / Y * 128 * local_block_index;

    // 判断循环的次数
    int32_t loopTimes = 1;
    if (addr_2.k != 0) {
        loopTimes = 2;
    }
    // todo:
    for (int32_t k = 0; k < loopTimes; k++) {

        auto l0_a_buf_tensor = l0_a_ping_pong_flag_ ? l0_a_pong_tensor : l0_a_ping_tensor;
        auto l1_base_a_cube1_tensor = l1_a_ping_pong_flag_ ? l1_a_pong_tensor : l1_a_ping_tensor;
        // 左矩阵加载进L0A
        WAIT_FLAG(M, MTE1, l0_a_ping_pong_flag_);
        auto commonLoadData2dParamsNoTranspose = AscendC::LoadData2dParams(
            0,                // startIndex
            REPEAT_TIME_8,    // repeatTimes
            SRC_STRIDE_8,     // srcStride
            0,                // sid
            0,                // dstGap
            false,            // ifTranspose
            0                 // addrMode
        );
        if (l0a_offset_remain == -1 || l0a_offset_remain == 0) {
            for (int32_t i = 0; i < SIZE_128 / BLOCK_SIZE; i++) {  // 按照Q的行16为单位循环搬入
                AscendC::LoadData(
                    l0_a_buf_tensor[i * SIZE_128 * BLOCK_SIZE],
                    l1_base_a_cube1_tensor[k * BASE_BLOCK + i * CUBE_MATRIX_SIZE],
                    commonLoadData2dParamsNoTranspose
                );
            }
        } else {
            for (int32_t i = 0; i < SIZE_128 / BLOCK_SIZE; i++) {  // 按照Q的行16为单位循环搬入
                AscendC::LoadData(
                    l0_a_buf_tensor[i * SIZE_128 * BLOCK_SIZE],
                    l1_base_a_cube1_tensor[BASE_BLOCK + i * CUBE_MATRIX_SIZE],
                    commonLoadData2dParamsNoTranspose
                );
            }
        }

        // 加载右矩阵
        auto addr_temp = k == 0 ? addr_1 : addr_2;
        auto gm_k = addr_temp.right;
        auto gm_c = addr_temp.out + remain_offset;
        for (int32_t j = 0; j < addr_temp.k; j++) {  // (H+1)/Y
            // 右矩阵：GM -> L1
            WAIT_FLAG(MTE1, MTE2, l1_b_ping_pong_flag_ + 2);

            auto l1_buf_b_tensor = l1_b_ping_pong_flag_ ? l1_b_pong_tensor : l1_b_ping_tensor;
            AscendC::DataCopy(
                l1_buf_b_tensor,
                this->gm_K_tensor[(uint32_t(gm_k - this->gm_b_cube1))],
                AscendC::Nd2NzParams(
                    1,         // ndNum
                    SIZE_128,  // nValue
                    SIZE_128,  // dValue
                    0,         // srcNdMatrixStride, unused
                    SIZE_128,  // srcDValue
                    SIZE_128,  // dstNzC0Stride
                    1,         // dstNzNStride
                    0          // dstNzMatrixStride, unused
                )
            );

            SET_FLAG(MTE2, MTE1, l1_b_ping_pong_flag_ + 2);
            WAIT_FLAG(MTE2, MTE1, l1_b_ping_pong_flag_ + 2);

            auto l0_b_buf_tensor = l0_b_ping_pong_flag_ ? l0_b_pong_tensor : l0_b_ping_tensor;
            auto l0_c_buf_tensor = l0_c_ping_pong_flag_ ? l0_c_pong_tensor : l0_c_ping_tensor;

            WAIT_FLAG(M, MTE1, l0_b_ping_pong_flag_ + 2);

            commonLoadData2dParamsNoTranspose.repeatTimes = REPEAT_TIME_64;
            commonLoadData2dParamsNoTranspose.srcStride = SRC_STRIDE_1;
            AscendC::LoadData(
                l0_b_buf_tensor,
                l1_buf_b_tensor,
                commonLoadData2dParamsNoTranspose
            );

            SET_FLAG(MTE1, MTE2, l1_b_ping_pong_flag_ + 2);
            SET_FLAG(MTE1, M, l0_b_ping_pong_flag_ + 2);
            WAIT_FLAG(MTE1, M, l0_b_ping_pong_flag_ + 2);

            auto l0_a_remain_offset = 0;
            auto m_mad = BASE_LEN;
            if (l0a_offset_remain != -1) {
                l0_a_remain_offset = remain_offset;
                m_mad = BASE_LEN / Y;
            }

            int unit_flag = 0b11;
            AscendC::Mmad(
                l0_c_buf_tensor,                     // dstLocal
                l0_a_buf_tensor[l0_a_remain_offset], // fmLocal
                l0_b_buf_tensor,                     // filterLocal
                AscendC::MmadParams(
                    m_mad,       // m
                    SIZE_128,    // n
                    SIZE_128,    // k
                    unit_flag,   // unitFlag
                    false,       // cmatrixSource
                    true         // cmatrixInitVal
                )
            );

            SET_FLAG(M, MTE1, l0_b_ping_pong_flag_ + 2);

            auto intriParams = AscendC::FixpipeParamsV220(
                SIZE_128,   // nSize
                m_mad,      // mSize
                m_mad,      // srcStride
                SIZE_128,   // dstStride
                false       // reluEn
            );
            intriParams.quantPre = QuantMode_t::NoQuant;
            intriParams.unitFlag = unit_flag;
            AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
                gm_c_cube1_tensor[gm_c - this->gm_c_cube1], // dstGlobal
                l0_c_buf_tensor,                            // srcLocal
                intriParams                                 // intriParams
            );

            l0_b_ping_pong_flag_ = 1 - l0_b_ping_pong_flag_;
            l0_c_ping_pong_flag_ = 1 - l0_c_ping_pong_flag_;
            l1_b_ping_pong_flag_ = 1 - l1_b_ping_pong_flag_;

            gm_k += BASE_BLOCK_SIZE;
            gm_c += BASE_BLOCK_SIZE;
        }

        SET_FLAG(M, MTE1, l0_a_ping_pong_flag_);
        l0_a_ping_pong_flag_ = 1 - l0_a_ping_pong_flag_;
    }
}


template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ inline void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::loadRowSumOnesblock()
{
    AscendC::DataCopy(
        l1_row_sum_1_buf_tensor,
        this->gm_ones_tensor[(uint32_t(gm_ones - this->gm_ones))],
        AscendC::Nd2NzParams(
            1,          // ndNum
            BASE_LEN,   // nValue
            SIZE_16,    // dValue
            0,          // srcNdMatrixStride, unused
            BASE_LEN,   // srcDValue
            BASE_LEN,   // dstNzC0Stride
            1,          // dstNzNStride
            0           // dstNzMatrixStride, unused
        )
    );

    SET_FLAG(MTE2, MTE1, EVENT_ID7);
    WAIT_FLAG(MTE2, MTE1, EVENT_ID7);
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ inline void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::addrMapping_cube2_rowsum(int32_t roundId,
    int64_t &src_length, int64_t &remain_length, PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *src,
    PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *remain)
{
    int32_t curCalcRow = roundId * F + core_group_index;  // 当前的row
    int32_t curCalcRowP = F * (roundId % 2) + core_group_index;
    int32_t b = curCalcRow / row_per_batch;           // 当前行所在的batch号
    int32_t n = (curCalcRow % row_per_batch) / L;     // 当前batch下的head号
    int32_t i_r = (curCalcRow % row_per_batch) % L;   // 当前head下的行号
    int32_t i_c = local_block_index * column_per_core;  // 当前core处理的起始列
    int32_t begin_remain =
        this->sparseMode == 1 ? (W + 1 - column_remain) : (H1 + 1 - column_remain);  // sparse or triangle

    int32_t q_left_index = this->sparseMode == 1 ? (W / 2 + i_r) : (L + i_r);           // sparse or triangle
    int32_t q_right_index = this->sparseMode == 1 ? (W / 2 - 1 - i_r) : (L - 1 - i_r);  // sparse or triangle
    int32_t switch_index = q_left_index;

    // 设置sparse的标志位 以及 sparse场景非倒三角部分的偏移
    int32_t sparse_flag = false;
    if (this->sparseMode == 1) {
        sparse_flag = i_r > ((W - 1) / 2) ? true : false;
    }
    int32_t row_down = i_r + W / 2;
    int32_t col_down = i_r + i_c - W / 2;
    int32_t col_down_remain = i_r + begin_remain - W / 2;

    int32_t g = n / nG;
    int64_t bn_offset = (b * N * H1 + n * H1) * BASE_BLOCK_SIZE;
    int64_t bn_offset_rowsum = (b * N * H1 + n * H1) * 128;
    int64_t bn_offset_GQA = (b * G * H1 + g * H1) * BASE_BLOCK_SIZE;
    int64_t q_left_offset = bn_offset + q_left_index * BASE_BLOCK_SIZE;
    int64_t q_right_offset = bn_offset + q_right_index * BASE_BLOCK_SIZE;
    // int64_t out_offset = (curCalcRow * (H + 1)) * BASE_BLOCK_SIZE;
    int64_t out_offset = this->sparseMode == 1 ? ((curCalcRowP * (W + 1)) * BASE_BLOCK_SIZE)
                                               : ((curCalcRowP * (H1 + 1)) * BASE_BLOCK_SIZE);  // sparse or triangle

    if (!sparse_flag && switch_index < i_c) {
        src[0].left = gm_c_cube1 + out_offset + i_c * BASE_BLOCK_SIZE;
        src[0].right = gm_b_cube2 + bn_offset_GQA + (i_c - switch_index - 1) * BASE_BLOCK_SIZE;
        src[0].out = gm_c_cube2 + bn_offset + q_right_index * BASE_BLOCK_SIZE;
        src[0].rowsumOut = rowsum_gm + bn_offset_rowsum + q_right_index * 128;
        src[0].k = column_per_core;
        src_length = 1;

        if (column_remain != 0) {
            remain[0].left =
                gm_c_cube1 + out_offset + begin_remain * BASE_BLOCK_SIZE + (128 / Y) * local_block_index * 128;
            remain[0].right = gm_b_cube2 + bn_offset_GQA + (begin_remain - switch_index - 1) * BASE_BLOCK_SIZE;
            remain[0].out = src[0].out + (128 / Y) * local_block_index * 128;
            remain[0].rowsumOut = src[0].rowsumOut + (128 / Y) * local_block_index;
            remain[0].k = column_remain;
            remain_length = 1;
        }
    } else if (!sparse_flag && i_c <= switch_index && i_c + column_per_core > switch_index + 1) {
        src[0].left = gm_c_cube1 + out_offset + i_c * BASE_BLOCK_SIZE;
        src[0].right = gm_b_cube2 + bn_offset_GQA + i_c * BASE_BLOCK_SIZE;
        src[0].out = gm_c_cube2 + bn_offset + q_left_index * BASE_BLOCK_SIZE;
        src[0].rowsumOut = rowsum_gm + bn_offset_rowsum + q_left_index * 128;
        src[0].k = switch_index - i_c + 1;

        src[1].left = gm_c_cube1 + out_offset + i_c * BASE_BLOCK_SIZE + src[0].k * BASE_BLOCK_SIZE;
        src[1].right = gm_b_cube2 + bn_offset_GQA;
        src[1].out = gm_c_cube2 + bn_offset + q_right_index * BASE_BLOCK_SIZE;
        src[1].rowsumOut = rowsum_gm + bn_offset_rowsum + q_right_index * 128;
        src[1].k = column_per_core - src[0].k;

        src_length = 2;

        if (column_remain != 0) {
            remain[0].left =
                gm_c_cube1 + out_offset + begin_remain * BASE_BLOCK_SIZE + (128 / Y) * local_block_index * 128;
            remain[0].right = gm_b_cube2 + bn_offset_GQA + (begin_remain - switch_index - 1) * BASE_BLOCK_SIZE;
            remain[0].out = src[1].out + (128 / Y) * local_block_index * 128;
            remain[0].rowsumOut = src[1].rowsumOut + (128 / Y) * local_block_index;
            remain[0].k = column_remain;
            remain_length = 1;
        }
    } else if (!sparse_flag && i_c <= switch_index && i_c + column_per_core <= switch_index + 1) {
        src[0].left = gm_c_cube1 + out_offset + i_c * BASE_BLOCK_SIZE;
        src[0].right = gm_b_cube2 + bn_offset_GQA + i_c * BASE_BLOCK_SIZE;
        src[0].out = gm_c_cube2 + bn_offset + q_left_index * BASE_BLOCK_SIZE;
        src[0].rowsumOut = rowsum_gm + bn_offset_rowsum + q_left_index * 128;
        src[0].k = column_per_core;
        src_length = 1;

        if (column_remain != 0) {
            if (switch_index < begin_remain) {
                remain[0].left =
                    gm_c_cube1 + out_offset + begin_remain * BASE_BLOCK_SIZE + (128 / Y) * local_block_index * 128;
                remain[0].right = gm_b_cube2 + bn_offset_GQA + (begin_remain - switch_index - 1) * BASE_BLOCK_SIZE;
                remain[0].out =
                    gm_c_cube2 + bn_offset + q_right_index * BASE_BLOCK_SIZE + (128 / Y) * local_block_index * 128;
                remain[0].rowsumOut =
                    rowsum_gm + bn_offset_rowsum + q_right_index * 128 + (128 / Y) * local_block_index;
                remain[0].k = column_remain;
                remain_length = 1;
            } else {
                remain[0].left =
                    gm_c_cube1 + out_offset + begin_remain * BASE_BLOCK_SIZE + (128 / Y) * local_block_index * 128;
                remain[0].right = gm_b_cube2 + bn_offset_GQA + begin_remain * BASE_BLOCK_SIZE;
                remain[0].out =
                    gm_c_cube2 + bn_offset + q_left_index * BASE_BLOCK_SIZE + (128 / Y) * local_block_index * 128;
                remain[0].rowsumOut =
                    rowsum_gm + bn_offset_rowsum + q_left_index * 128 + (128 / Y) * local_block_index;
                remain[0].k = switch_index - begin_remain + 1;

                remain[1].left = remain[0].left + remain[0].k * BASE_BLOCK_SIZE;
                remain[1].right = gm_b_cube2 + bn_offset_GQA;
                remain[1].out =
                    gm_c_cube2 + bn_offset + q_right_index * BASE_BLOCK_SIZE + (128 / Y) * local_block_index * 128;
                remain[1].rowsumOut =
                    rowsum_gm + bn_offset_rowsum + q_right_index * 128 + (128 / Y) * local_block_index;
                remain[1].k = column_remain - remain[0].k;

                remain_length = 2;
            }
        }
    } else {
        src[0].left = gm_c_cube1 + out_offset + i_c * BASE_BLOCK_SIZE;
        src[0].right = gm_b_cube2 + bn_offset_GQA + col_down * BASE_BLOCK_SIZE;
        src[0].out = gm_c_cube2 + bn_offset + row_down * BASE_BLOCK_SIZE;
        src[0].rowsumOut = rowsum_gm + bn_offset_rowsum + row_down * 128;
        src[0].k = column_per_core;
        src_length = 1;

        if (column_remain != 0) {
            remain[0].left =
                gm_c_cube1 + out_offset + begin_remain * BASE_BLOCK_SIZE + (128 / Y) * local_block_index * 128;
            remain[0].right = gm_b_cube2 + bn_offset_GQA + col_down_remain * BASE_BLOCK_SIZE;
            remain[0].out = src[0].out + (128 / Y) * local_block_index * 128;
            remain[0].rowsumOut = src[0].rowsumOut + (128 / Y) * local_block_index;
            remain[0].k = column_remain;
            remain_length = 1;
        }
    }
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ inline void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::addrMapping_cube2_rowsum_nomask(int32_t roundId,
    int64_t &src_length, int64_t &remain_length, PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *src,
    PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *remain)
{
    int32_t curCalcRow = roundId * F + core_group_index;  // 当前的row 0
    int32_t curCalcRowP = F * (roundId % 2) + core_group_index;
    int32_t b = curCalcRow / row_per_batch;               // 当前行所在的batch号
    int32_t n = (curCalcRow % row_per_batch) / L;         // 当前batch下的head号
    int32_t i_r = (curCalcRow % row_per_batch) % L;       // 当前head下的行号
    int32_t i_c =  local_block_index * column_per_core ;    // 当前core处理的起始列
    int32_t begin_remian = H2 - column_remain;

    int32_t q_left_index = i_r;
    int32_t q_right_index = L - 1 - i_r;
    int32_t switch_index = H2;

    int32_t g = n / nG;
    int64_t bn_offset_1 = (b * N * H1 + n * H1) * BASE_BLOCK_SIZE;
    int64_t bn_offset_rowsum = (b * N * H1 + n * H1) * 128;
    int64_t bn_offset_2 = (b * G * H2 + g * H2) * BASE_BLOCK_SIZE;  // GQA version of bn_offset_2
    int64_t out_offset = (curCalcRowP * H2) * BASE_BLOCK_SIZE;

    src[0].left = gm_c_cube1 + out_offset + i_c * BASE_BLOCK_SIZE;
    src[0].right = gm_b_cube2 + bn_offset_2 + i_c * BASE_BLOCK_SIZE;
    src[0].out = gm_c_cube2 + bn_offset_1 + q_left_index * BASE_BLOCK_SIZE;
    src[0].rowsumOut = rowsum_gm + bn_offset_rowsum + q_left_index * 128;
    src[0].k = column_per_core;
    src_length = 1;
    if (column_remain != 0) {
        remain[0].left = gm_c_cube1 + out_offset + begin_remian *
            BASE_BLOCK_SIZE + (128 / Y) * local_block_index * 128;
        remain[0].right = gm_b_cube2 + bn_offset_2 + begin_remian * BASE_BLOCK_SIZE;
        remain[0].out = gm_c_cube2 + bn_offset_1 + q_left_index *
            BASE_BLOCK_SIZE + (128 / Y) * local_block_index * 128;
        remain[0].rowsumOut = rowsum_gm + bn_offset_rowsum +
            q_left_index * 128 + (128 / Y) * local_block_index;
        remain[0].k = column_remain;
        remain_length = 1;
    }
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ inline void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::unified_addrMapping_cube2mix(int32_t roundId,
    int64_t &src_length, int64_t &remain_length, PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *src,
    PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *remain)
{
    if (this->qk_triangle == 1) {  // 倒三角
        addrMapping_cube2_rowsum(roundId, src_length, remain_length, src, remain);
    } else if (this->qk_triangle == 0) {  // 非倒三角
        addrMapping_cube2_rowsum_nomask(roundId, src_length, remain_length, src, remain);
    }
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ inline void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::cube2_rowsum_mix_only(
    const PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *cube_addr, int32_t cube_length, int32_t m_length,
    int32_t vcore_num_per_head) {
    int32_t l1_m = CUBE2_LENGTH_M;    // 128
    int32_t l1_k = CUBE2_LENGTH_K;
    int32_t l1_n = CUBE2_LENGTH_N;
    int32_t l0_k = BASE_BLOCK_LENGTH; // 128
    int32_t l0_m = BASE_BLOCK_LENGTH; // 128
    int32_t l0_n = BASE_BLOCK_LENGTH; // 128

    for (int32_t i = 0; i < cube_length; i++) {
        __gm__ TYPE *__restrict__ left_start_addr = (__gm__ TYPE *)(cube_addr[i].left);  // 高精度是需要将float转为TYPE
        __gm__ TYPE *__restrict__ right_start_addr = cube_addr[i].right;
        __gm__ float *__restrict__ result_gm = cube_addr[i].out;
        __gm__ float *__restrict__ rowsum_out_gm = cube_addr[i].rowsumOut;

        int32_t k_core_gm = cube_addr[i].k * l0_k;
        l1_k = k_core_gm < l1_k ? k_core_gm : l1_k;
        int32_t loop_l1_times = k_core_gm / l1_k;
        int32_t l0_k_block_num = l0_k / BLOCK_SIZE; // 128/16
        int32_t l0_m_block_num = m_length / BLOCK_SIZE; // 128/16
        int32_t l0_n_block_num = l0_n / BLOCK_SIZE; // 8

        for (int32_t l1_loop_index = 0; l1_loop_index < loop_l1_times; l1_loop_index++) {
            auto l1_buf_a_cube2_tensor = l1_a_ping_pong_flag_ ? l1_a_pong_tensor : l1_a_ping_tensor;
            auto l1_buf_b_cube2_tensor = l1_b_ping_pong_flag_ ? l1_b_pong_tensor : l1_b_ping_tensor;
            auto l0a_buf_tensor = l0_a_ping_pong_flag_ ? l0_a_pong_tensor: l0_a_ping_tensor;
            auto l0b_buf_tensor = l0_b_ping_pong_flag_ ? l0_b_pong_tensor: l0_b_ping_tensor;
            auto l0c_buf_tensor = l0_c_ping_pong_flag_ ? l0_c_pong_tensor: l0_c_ping_tensor;

            WAIT_FLAG(MTE1, MTE2, l1_a_ping_pong_flag_);

            AscendC::DataCopy(
                l1_buf_a_cube2_tensor,
                this->gm_c_cube1_tensor_type[(uint32_t(left_start_addr - (__gm__ TYPE *)this->gm_c_cube1)
                + l1_loop_index * l0_m * l0_k * 2)],
                AscendC::Nd2NzParams(
                    vcore_num_per_head,
                    m_length / vcore_num_per_head,
                    l0_k,
                    m_length * l0_k / vcore_num_per_head * 2,
                    l0_k,
                    m_length,
                    1,
                    m_length / vcore_num_per_head * BLOCK_SIZE
                )
            );

            SET_FLAG(MTE2, MTE1, l1_a_ping_pong_flag_);
            WAIT_FLAG(MTE2, MTE1, l1_a_ping_pong_flag_);
            WAIT_FLAG(M, MTE1, l0_a_ping_pong_flag_);

            for (int32_t l0a_load_idx = 0; l0a_load_idx < l0_k_block_num; l0a_load_idx++) {
                AscendC::LoadData(
                    l0a_buf_tensor[l0a_load_idx * CUBE_MATRIX_SIZE],
                    l1_buf_a_cube2_tensor[l0a_load_idx * l0_m_block_num * CUBE_MATRIX_SIZE],
                    AscendC::LoadData2dParams(
                        0,                  // startIndex
                        l0_m_block_num,     // repeatTimes
                        1,                  // srcStride
                        0,                  // sid
                        l0_k_block_num - 1, // dstGap
                        false,              // ifTranspose
                        0                   // addrMode
                    )
                );
            }

            SET_FLAG(MTE1, MTE2, l1_a_ping_pong_flag_);
            WAIT_FLAG(MTE1, MTE2, l1_b_ping_pong_flag_ + 2);

            AscendC::DataCopy(
                l1_buf_b_cube2_tensor,
                this->gm_V_tensor[(uint32_t(right_start_addr - this->gm_b_cube2) + l1_loop_index * l0_k * l0_n)],
                AscendC::Nd2NzParams(
                    1,            // ndNum
                    l0_k,         // nValue
                    l0_n,         // dValue
                    l0_k * l0_n,  // srcNdMatrixStride
                    l0_n,         // srcDValue
                    l0_k,         // dstNzC0Stride
                    1,            // dstNzNStride
                    l0_k * l0_n   // dstNzMatrixStride
                )
            );

            SET_FLAG(MTE2, MTE1, l1_b_ping_pong_flag_ + 2);
            WAIT_FLAG(MTE2, MTE1, l1_b_ping_pong_flag_ + 2);
            WAIT_FLAG(M, MTE1, l0_b_ping_pong_flag_ + 2);

            for (int32_t l0b_load_idx = 0; l0b_load_idx < l0_k_block_num; l0b_load_idx++) {
                AscendC::LoadData(
                    l0b_buf_tensor[l0b_load_idx * l0_n * BLOCK_SIZE],
                    l1_buf_b_cube2_tensor[l0b_load_idx * CUBE_MATRIX_SIZE],
                    AscendC::LoadData2dParams(
                        0,              // startIndex
                        l0_n_block_num, // repeatTimes
                        l0_k_block_num, // srcStride
                        0,              // sid
                        0,              // dstGap
                        true,           // ifTranspose
                        0               // addrMode
                    )
                );
            }

            SET_FLAG(MTE1, MTE2, l1_b_ping_pong_flag_ + 2);
            SET_FLAG(MTE1, M, l0_b_ping_pong_flag_ + 2);
            WAIT_FLAG(MTE1, M, l0_b_ping_pong_flag_ + 2);

            bool init_c = (l1_loop_index == 0);
            bool out_c = (l1_loop_index == loop_l1_times - 1);
            int unit_flag = 0b10;
            if (out_c) {
                unit_flag = 0b11;
            }

            AscendC::Mmad(
                l0c_buf_tensor,                  // dstLocal
                l0a_buf_tensor,                  // fmLocal
                l0b_buf_tensor,                  // filterLocal
                AscendC::MmadParams(
                    m_length,                    // m
                    l0_n,                        // n
                    l0_k,                        // k
                    unit_flag,                   // unitFlag
                    false,                       // cmatrixSource
                    (init_c == 1) ? true : false // cmatrixInitVal
                )
            );

            SET_FLAG(M, MTE1, l0_b_ping_pong_flag_ + 2);

            if (out_c) {
                AscendC::SetAtomicAdd<float>();
                AscendC::SetAtomicType<float>();
                auto intriParams = AscendC::FixpipeParamsV220(
                    l0_n,     // nSize
                    m_length, // mSize
                    m_length, // srcStride
                    l0_n,     // dstStride
                    false     // reluEn
                );
                intriParams.quantPre = QuantMode_t::NoQuant;
                intriParams.unitFlag = unit_flag;
                AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
                    gm_c_cube2_tensor[result_gm - this->gm_c_cube2], // dstGlobal
                    l0c_buf_tensor,                                  // srcLocal
                    intriParams                                      // intriParams
                );
                AscendC::SetAtomicNone();
            }

            l0_b_ping_pong_flag_ = 1 - l0_b_ping_pong_flag_;
            l0_c_ping_pong_flag_ = 1 - l0_c_ping_pong_flag_;
            l0b_buf_tensor = l0_b_ping_pong_flag_ ? l0_b_pong_tensor: l0_b_ping_tensor;
            l0c_buf_tensor = l0_c_ping_pong_flag_ ? l0_c_pong_tensor: l0_c_ping_tensor;

            // rowsum: 将全1矩阵从L1搬运到L0b
            WAIT_FLAG(M, MTE1, l0_b_ping_pong_flag_ + 2);

            SET_FLAG(M, MTE1, l0_b_ping_pong_flag_ + 2);
            SET_FLAG(M, MTE1, l0_a_ping_pong_flag_);

            l1_a_ping_pong_flag_ = 1 - l1_a_ping_pong_flag_;
            l1_b_ping_pong_flag_ = 1 - l1_b_ping_pong_flag_;
            l0_a_ping_pong_flag_ = 1 - l0_a_ping_pong_flag_;
            l0_b_ping_pong_flag_ = 1 - l0_b_ping_pong_flag_;
            l0_c_ping_pong_flag_ = 1 - l0_c_ping_pong_flag_;
        }
    }
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ inline void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::mix_cube2_rowsum(int32_t cube2_roundId)
{
    // 混算寻址：
    PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> src[2];
    PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> remain[2];
    int64_t src_length = 0;
    int64_t remain_length = 0;
    unified_addrMapping_cube2mix(cube2_roundId, src_length, remain_length, src, remain);  // cube2和rowsum的寻址

    // 开始混算：
    cube2_rowsum_mix_only(src, src_length, BASE_BLOCK_LENGTH, this->Y * 2);
    if (remain_length > 0) {
        cube2_rowsum_mix_only(remain, remain_length, BASE_BLOCK_LENGTH / this->Y, 2);
    }
}

#endif

#endif  // __CUBEFORWARD_H__
