/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __CUBEFORWARD_192_H__
#define __CUBEFORWARD_192_H__

#define USE_ASCENDC
#include "ppmatmul_const.h"
#include "AddressMappingForward.h"
#include "kernels/utils/kernel/common.h"
#include "kernels/utils/kernel/utils.h"
#include "kernels/utils/kernel/iterator.h"
#include "kernels/utils/kernel/iterator.h"

using namespace AscendC;

#ifdef __DAV_C220_CUBE__

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE> class CubeForward_192 {
public:
    __aicore__ inline CubeForward_192(){};
    __aicore__ inline void Init(__gm__ uint8_t *__restrict__ a_cube1, __gm__ uint8_t *__restrict__ b_cube1,
                                __gm__ uint8_t *__restrict__ b_cube2, __gm__ uint8_t *__restrict__ c_cube1,
                                __gm__ float *__restrict__ c_cube2, __gm__ uint8_t *__restrict__ ones_rowsum,
                                __gm__ float *__restrict__ gm_rowsum, int32_t Y, int32_t F, int32_t B, int32_t N,
                                int32_t S1, int32_t S2, int32_t D, int32_t nG, int32_t qk_triangle, int32_t sparseMode,
                                int32_t windowLength);
    __aicore__ inline void PresetFlag();
    __aicore__ inline void ClearFlag();
    __aicore__ inline void SetHighPrecision(bool isHighPrecision)
    {
        this->isHighPrecision = isHighPrecision;
    };
    __aicore__ inline void Run();

    // headDim = 192的cube1模块
    __aicore__ __inline__ void
    cube1_headDim192(Address_192::PHY_ADDR_FORWARD_CUBE1<TYPE, TYPE, WORKSPACE_TYPE> *src,
                     Address_192::PHY_ADDR_FORWARD_CUBE1<TYPE, TYPE, WORKSPACE_TYPE> *remain);
    __aicore__ __inline__ void
    base_block_mad_headDim192(Address_192::PHY_ADDR_FORWARD_CUBE1<TYPE, TYPE, WORKSPACE_TYPE> addr_1,
                              Address_192::PHY_ADDR_FORWARD_CUBE1<TYPE, TYPE, WORKSPACE_TYPE> addr_2,
                              int32_t l0a_offset_remain = -1);

protected:
    __aicore__ __inline__ void loadRowSumOnesblock(); // 加载rowsum右全1矩阵
    __aicore__ __inline__ void addrMapping_cube2_rowsum_192(
        int32_t roundId, int64_t &src_length, int64_t &remain_length, PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *src,
        PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *remain); // 混算中cube2、rowsum寻址（倒三角+sparse）
    __aicore__ __inline__ void addrMapping_cube2_rowsum_nomask_192(
        int32_t roundId, int64_t &src_length, int64_t &remain_length, PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *src,
        PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *remain); // 混算中cube2、rowsum寻址（nomask+不等长）
    __aicore__ __inline__ void
    unified_addrMapping_cube2mix_192(int32_t roundId, int64_t &src_length, int64_t &remain_length,
                                     PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *src,
                                     PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *remain); // cube2和rowsum混算的寻址
    __aicore__ __inline__ void cube2_rowsum_mix_only_192(const PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *cube_addr,
                                                         int32_t cube_length, int32_t m_length,
                                                         int32_t vcore_num_per_head); // cube2和row_sum混算的基本单元
    __aicore__ __inline__ void mix_cube2_rowsum_192(int32_t cube2_roundId);           // cube2、rowsum的混算
protected:
    Address_192::AddressMappingForward<TYPE> address; // 注入前向寻址模块

    AsdopsBuffer<ArchType::ASCEND_V220> asdopsBuf;

    __gm__ TYPE *__restrict__ gm_a_cube1;
    __gm__ TYPE *__restrict__ gm_b_cube1;
    __gm__ TYPE *__restrict__ gm_b_cube2;
    __gm__ WORKSPACE_TYPE *__restrict__ gm_c_cube1;
    __gm__ float *__restrict__ gm_c_cube2;
    __gm__ TYPE *__restrict__ gm_ones;
    __gm__ float *__restrict__ rowsum_gm;

    // init gm input tensor
    GlobalTensor<TYPE> gm_Q_tensor;
    GlobalTensor<TYPE> gm_K_tensor;
    GlobalTensor<TYPE> gm_V_tensor;

    // init gm output tensor
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

    uint32_t l1_a_ping_pong_flag_ = 0;
    uint32_t l1_b_ping_pong_flag_ = 0;

    uint32_t l0_a_ping_pong_flag_ = 0;
    uint32_t l0_b_ping_pong_flag_ = 0;
    uint32_t l0_c_ping_pong_flag_ = 0;

    LocalTensor<TYPE> l0_a_ping_tensor;
    LocalTensor<TYPE> l0_a_pong_tensor;
    LocalTensor<TYPE> l0_b_ping_tensor;
    LocalTensor<TYPE> l0_b_pong_tensor;
    LocalTensor<float> l0_c_ping_tensor;
    LocalTensor<float> l0_c_pong_tensor;

    // Y个core处理一个完整行，所有core分成 F 组， block_dim = F * Y
    int32_t Y;
    int32_t F;

    // Q K V shape : [B,N,S,D] 其中 D 固定为128
    int32_t B;
    int32_t N;
    // int32_t S; //256 - 128k (256的倍数) 方阵的行
    int32_t S1; // 256 - 128k (256的倍数) 方阵的行
    int32_t S2; // 256 - 128k (256的倍数) 方阵的行
    int32_t D;
    int32_t nG;
    int32_t G;            // GQA 场景  k v shape [B,G,S,D]
    int32_t qk_triangle;  // GQA 场景  k v shape [B,G,S,D]
    int32_t sparseMode;   // sparseMode: 0:dense, 1:sparse
    int32_t windowLength; // sparse场景下，滑动窗口的长度

    // int32_t H;
    int32_t H1;
    int32_t H2;
    int32_t L; // 负责均衡
    int32_t W;

    int32_t curCoreIndex;
    int32_t local_block_index; // 组内 core id [0, Y-1], 组内第几个 0
    int32_t core_group_index;  // [0, F-1], 第几组

    int32_t row_per_batch;
    int32_t column_per_core;
    int32_t column_remain;

    bool isHighPrecision = true;
};


template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ inline void CubeForward_192<TYPE, IF_BF16, WORKSPACE_TYPE>::PresetFlag()
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
__aicore__ inline void CubeForward_192<TYPE, IF_BF16, WORKSPACE_TYPE>::ClearFlag()
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
__aicore__ inline void CubeForward_192<TYPE, IF_BF16, WORKSPACE_TYPE>::loadRowSumOnesblock()
{
    AscendC::DataCopy(l1_row_sum_1_buf_tensor, this->gm_ones_tensor[(uint32_t(gm_ones - this->gm_ones))],
                      AscendC::Nd2NzParams(1,   // ndNum
                                           128, // nValue
                                           16,  // dValue
                                           0,   // srcNdMatrixStride, unused
                                           128, // srcDValue
                                           128, // dstNzC0Stride
                                           1,   // dstNzNStride
                                           0    // dstNzMatrixStride, unused
                                           ));
    SET_FLAG(MTE2, MTE1, EVENT_ID7);
    WAIT_FLAG(MTE2, MTE1, EVENT_ID7);
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ inline void CubeForward_192<TYPE, IF_BF16, WORKSPACE_TYPE>::Init(
    __gm__ uint8_t *__restrict__ a_cube1, __gm__ uint8_t *__restrict__ b_cube1, __gm__ uint8_t *__restrict__ b_cube2,
    __gm__ uint8_t *__restrict__ c_cube1, __gm__ float *__restrict__ c_cube2, __gm__ uint8_t *__restrict__ ones_rowsum,
    __gm__ float *__restrict__ gm_rowsum, int32_t y_cube_num_per_line, int32_t group_num, int32_t batch,
    int32_t n_batch,
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
        this->H1 = this->S1 / 128;
        this->H2 = this->S2 / 128;
        this->L = this->H1; // 负责均衡
        this->column_per_core = this->H2 / this->Y;
        this->column_remain = this->H2 % this->Y;
    }
    // 有mask
    else {
        this->H1 = this->S1 / 128;
        this->H2 = this->S2 / 128;
        this->L = this->H1 / 2; // 负责均衡
        this->column_per_core = (this->H2 + 1) / this->Y;
        this->column_remain = (this->H2 + 1) % this->Y;

        // sparse场景：行数、列数以及尾块需要重新设置 update
        if (this->sparseMode == 1) {
            this->H1 = this->S1 / 128;
            this->H2 = this->S2 / 128;
            this->W = this->windowLength / 128;
            this->L = this->H1 - this->W / 2;
            this->column_per_core = (this->W + 1) / this->Y;
            this->column_remain = (this->W + 1) % this->Y;
        }
    }

    this->curCoreIndex = get_block_idx();
    this->core_group_index = this->curCoreIndex / this->Y;
    this->local_block_index = this->curCoreIndex % this->Y;

    this->row_per_batch = this->N * this->L;

    // 寻址模块的初始化
    address.init(this->B, this->N, this->G, this->S1, this->S2, this->D, this->qk_triangle);

    // 设置寻址模块核组信息
    address.set_core_info(this->Y * this->F, this->F, this->Y, this->curCoreIndex, this->core_group_index,
                          this->local_block_index);

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
    l1_base_b_cube1_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(128 * 1024);
    l1_base_a_cube2_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(256 * 1024);
    l1_base_b_cube2_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(384 * 1024);

    l1_a_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(0);
    l1_a_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(128 * 1024);
    l1_b_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(256 * 1024);
    l1_b_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(384 * 1024);

    l1_row_sum_1_buf_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(504 * 1024);

    // init L0A/L0B/L0C tensor
    l0_a_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0A, TYPE>(0);
    l0_a_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0A, TYPE>(32 * 1024);
    l0_b_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0B, TYPE>(0);
    l0_b_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0B, TYPE>(32 * 1024);
    l0_c_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(0);
    l0_c_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(64 * 1024);
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ inline void CubeForward_192<TYPE, IF_BF16, WORKSPACE_TYPE>::Run()
{
    SetPadding(0);
    uint64_t config = 0x1;
    AscendC::SetNdParaImpl(config);

    // 等待vector workspace的初始化
    WaitFlagDev(AIV2AICFLAGID);
    this->loadRowSumOnesblock(); // 先加载全1矩阵，使其常驻L1

    this->PresetFlag();
    // 计算每个核的总轮次 Z，  注意：存在余数需要计算当前核释放参与尾行的计算
    int total_rounds = address.get_total_round();

    if (total_rounds == 1) {
        if (!address.is_running(0)) {
            if (this->Y > 1) {
                FftsCrossCoreSync<PIPE_FIX, 0>(AICFLAGID);
                WaitFlagDev(AICFLAGID);
            }
        } else {
            Address_192::PHY_ADDR_FORWARD_CUBE1<TYPE, TYPE, WORKSPACE_TYPE> src[2];
            Address_192::PHY_ADDR_FORWARD_CUBE1<TYPE, TYPE, WORKSPACE_TYPE> remain[2];
            address.addrMapping_cube1(this->gm_a_cube1, this->gm_b_cube1, this->gm_c_cube1, 0, src, remain);
            cube1_headDim192(src, remain);

            if (this->Y > 1) {
                FftsCrossCoreSync<PIPE_FIX, 0>(AICFLAGID);
                WaitFlagDev(AICFLAGID);
            }

            // aic sync aiv
            FftsCrossCoreSync<PIPE_FIX, 2>(AIC2AIVFLAGID);
            WaitFlagDev(AIV2AICFLAGID);

            // cube2与rowsum的混算
            mix_cube2_rowsum_192(0);
        }
    } else {
        int32_t cube2_index = 0;
        /************************************CUBE1*2******************************************************/
        for (int32_t i = 0; i < 2; i++) {
            if (address.is_running(i)) {
                Address_192::PHY_ADDR_FORWARD_CUBE1<TYPE, TYPE, WORKSPACE_TYPE> src[2];
                Address_192::PHY_ADDR_FORWARD_CUBE1<TYPE, TYPE, WORKSPACE_TYPE> remain[2];
                address.addrMapping_cube1(this->gm_a_cube1, this->gm_b_cube1, this->gm_c_cube1, i, src, remain);
                cube1_headDim192(src, remain);

                if (this->Y > 1) {
                    FftsCrossCoreSync<PIPE_FIX, 0>(AICFLAGID);
                    WaitFlagDev(AICFLAGID);
                }
                if (i == 0) {
                    // aic sync aiv
                    FftsCrossCoreSync<PIPE_FIX, 2>(AIC2AIVFLAGID);
                }
            } else if (i == (total_rounds - 1)) {
                if (this->Y > 1) {
                    FftsCrossCoreSync<PIPE_FIX, 0>(AICFLAGID);
                    WaitFlagDev(AICFLAGID);
                }
            }
        }
        /****************************************** CUBE2 + CUBE1 ***************************************/
        for (int32_t i = 2; i < total_rounds; i++) {
            WaitFlagDev(AIV2AICFLAGID);
            FftsCrossCoreSync<PIPE_FIX, 2>(AIC2AIVFLAGID);

            // cube2与rowsum的混算
            this->mix_cube2_rowsum_192(cube2_index);
            cube2_index += 1;

            if (address.is_running(i)) {
                Address_192::PHY_ADDR_FORWARD_CUBE1<TYPE, TYPE, WORKSPACE_TYPE> src[2];
                Address_192::PHY_ADDR_FORWARD_CUBE1<TYPE, TYPE, WORKSPACE_TYPE> remain[2];
                address.addrMapping_cube1(this->gm_a_cube1, this->gm_b_cube1, this->gm_c_cube1, i, src, remain);
                cube1_headDim192(src, remain);

                // 多核处理一个完整行，需要全核同步
                if (this->Y > 1) {
                    FftsCrossCoreSync<PIPE_FIX, 0>(AICFLAGID);
                    WaitFlagDev(AICFLAGID);
                }
            } else {
                if (this->Y > 1) {
                    FftsCrossCoreSync<PIPE_FIX, 0>(AICFLAGID);
                    WaitFlagDev(AICFLAGID);
                }
            }
        }
        /************************************CUBE2*2******************************************************/
        for (int32_t i = 0; i < 2; i++) {
            if (address.is_running(total_rounds - 2 + i)) {
                WaitFlagDev(AIV2AICFLAGID);
                if (i == 0) {
                    FftsCrossCoreSync<PIPE_FIX, 2>(AIC2AIVFLAGID);
                }

                // cube2与rowsum的混算
                this->mix_cube2_rowsum_192(cube2_index);
                cube2_index += 1;
            }
        }
    }
    FftsCrossCoreSync<PIPE_FIX, 0>(AICFLAGID);
    WaitFlagDev(AICFLAGID);

    FftsCrossCoreSync<PIPE_FIX, 2>(AIC2AIVFLAGID);

    this->ClearFlag();
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ __inline__ void CubeForward_192<TYPE, IF_BF16, WORKSPACE_TYPE>::base_block_mad_headDim192(
    Address_192::PHY_ADDR_FORWARD_CUBE1<TYPE, TYPE, WORKSPACE_TYPE> addr_1,
    Address_192::PHY_ADDR_FORWARD_CUBE1<TYPE, TYPE, WORKSPACE_TYPE> addr_2, int32_t l0a_offset_remain)
{
    // 整体思路：将128*192左矩阵（A）切为128*96两块(A1,A2)，将192*128右矩阵(B)切为96*128两块（B1,B2）,
    // 则C = A * B = A1 * B1 + A2 * B2, 通过两次循环分别计算A1 * B1 和 A2 * B2
    int32_t headDim = 192;
    int32_t head_dim_half = 192 / 2;
    auto gc_remain_offset = (l0a_offset_remain == -1) ? 0 : 128 / this->Y * 128 * this->local_block_index;

    // 判断循环的次数
    int32_t loopTimes = 1;
    if (addr_2.k != 0) {
        loopTimes = 2;
    }

    for (int32_t k = 0; k < loopTimes; k++) {
        // 加载右矩阵
        auto addr_temp = k == 0 ? addr_1 : addr_2;
        auto gm_k = addr_temp.right;
        auto gm_c = addr_temp.out + gc_remain_offset;
        for (int32_t j = 0; j < addr_temp.k; j++) {
            // 右矩阵：GM -> L1
            WAIT_FLAG(MTE1, MTE2, this->l1_b_ping_pong_flag_ + 2);
            auto l1_buf_b_tensor = this->l1_b_ping_pong_flag_ ? this->l1_b_pong_tensor : this->l1_b_ping_tensor;

            AscendC::DataCopy(l1_buf_b_tensor, this->gm_K_tensor[(uint32_t(gm_k - this->gm_b_cube1))],
                              AscendC::Nd2NzParams(1,   // ndNum
                                                   128, // nValue
                                                   256, // dValue
                                                   0,   // srcNdMatrixStride, unused
                                                   256, // srcDValue
                                                   128, // dstNzC0Stride
                                                   1,   // dstNzNStride
                                                   0    // dstNzMatrixStride, unused
                                                   ));
            SET_FLAG(MTE2, MTE1, this->l1_b_ping_pong_flag_ + 2);
            WAIT_FLAG(MTE2, MTE1, this->l1_b_ping_pong_flag_ + 2);

            auto l0_c_buf_tensor = this->l0_c_ping_tensor;
            for (int32_t m = 0; m < 2; m++) {
                auto l0_a_buf_tensor = this->l0_a_ping_pong_flag_ ? this->l0_a_pong_tensor : this->l0_a_ping_tensor;
                auto l1_base_a_cube1_tensor =
                    this->l1_a_ping_pong_flag_ ? this->l1_a_pong_tensor : this->l1_a_ping_tensor;
                // 左矩阵加载进L0A
                WAIT_FLAG(M, MTE1, this->l0_a_ping_pong_flag_);
                // 偏移量的设置
                auto l1a_offset = m == 0 ? 0 : 128 * 128;
                auto l0a_offset = m == 0 ? (128 * BLOCK_SIZE) : (64 * BLOCK_SIZE);
                auto l0a_repeat_times = m == 0 ? 8 : 4;
                if (l0a_offset_remain == -1 || l0a_offset_remain == 0) {
                    for (int32_t i = 0; i < 128 / BLOCK_SIZE; i++) {
                        AscendC::LoadData(l0_a_buf_tensor[i * l0a_offset],
                                          l1_base_a_cube1_tensor[k * 128 * headDim + l1a_offset + i * CUBE_MATRIX_SIZE],
                                          AscendC::LoadData2dParams(0,                // startIndex
                                                                    l0a_repeat_times, // repeatTimes
                                                                    8,                // srcStride
                                                                    0,                // sid
                                                                    0,                // dstGap
                                                                    false,            // ifTranspose
                                                                    0                 // addrMode
                                                                    ));
                    }
                } else {
                    for (int32_t i = 0; i < 128 / BLOCK_SIZE; i++) {
                        AscendC::LoadData(l0_a_buf_tensor[i * l0a_offset],
                                          l1_base_a_cube1_tensor[128 * headDim + l1a_offset + i * CUBE_MATRIX_SIZE],
                                          AscendC::LoadData2dParams(0,                // startIndex
                                                                    l0a_repeat_times, // repeatTimes
                                                                    8,                // srcStride
                                                                    0,                // sid
                                                                    0,                // dstGap
                                                                    false,            // ifTranspose
                                                                    0                 // addrMode
                                                                    ));
                    }
                }

                // 将K基本块从L1搬到L0B
                auto l0_b_buf_tensor = this->l0_b_ping_pong_flag_ ? this->l0_b_pong_tensor : this->l0_b_ping_tensor;
                auto l1b_offset = m == 0 ? 0 : 128 * 128;
                auto l0b_repeat_times = m == 0 ? 64 : 32;
                WAIT_FLAG(M, MTE1, this->l0_b_ping_pong_flag_ + 2);
                AscendC::LoadData(l0_b_buf_tensor, l1_buf_b_tensor[l1b_offset],
                                  AscendC::LoadData2dParams(0,                // startIndex
                                                            l0b_repeat_times, // repeatTimes
                                                            1,                // srcStride
                                                            0,                // sid
                                                            0,                // dstGap
                                                            false,            // ifTranspose
                                                            0                 // addrMode
                                                            ));

                if (m == 1) {
                    SET_FLAG(MTE1, MTE2, this->l1_b_ping_pong_flag_ + 2);
                }

                SET_FLAG(MTE1, M, this->l0_b_ping_pong_flag_ + 2);
                WAIT_FLAG(MTE1, M, this->l0_b_ping_pong_flag_ + 2);

                auto l0_a_remain_offset = 0;
                auto remain_offset = 0;
                if (l0a_offset_remain != -1) {
                    remain_offset = m == 0 ? (128 / this->Y * 128 * this->local_block_index) :
                                             (128 / this->Y * 64 * this->local_block_index);
                }
                auto m_mad = 128;
                if (l0a_offset_remain != -1) {
                    l0_a_remain_offset = remain_offset;
                    m_mad = 128 / this->Y;
                }
                int unit_flag = 0b10; // 搬出时等于0b11， 累加时0b10
                if (m == 1) {
                    unit_flag = 0b11;
                }
                bool init_c = (m == 0);
                auto d_value = m == 0 ? 128 : 64;
                AscendC::Mmad(l0_c_buf_tensor,                                 // dstLocal
                              l0_a_buf_tensor[l0_a_remain_offset],             // fmLocal
                              l0_b_buf_tensor,                                 // filterLocal
                              AscendC::MmadParams(m_mad,                       // m
                                                  128,                         // n
                                                  d_value,                     // k
                                                  unit_flag,                   // unitFlag
                                                  false,                       // cmatrixSource
                                                  (init_c == 1) ? true : false // cmatrixInitVal
                                                  ));
                SET_FLAG(M, MTE1, this->l0_b_ping_pong_flag_ + 2);

                this->l0_b_ping_pong_flag_ = 1 - this->l0_b_ping_pong_flag_;
                this->l0_c_ping_pong_flag_ = 1 - this->l0_c_ping_pong_flag_;

                SET_FLAG(M, MTE1, this->l0_a_ping_pong_flag_);
                this->l0_a_ping_pong_flag_ = 1 - this->l0_a_ping_pong_flag_;
            }
            int unit_flag = 0b11;
            auto m_mad = 128;
            if (l0a_offset_remain != -1) {
                m_mad = 128 / this->Y;
            }
            auto intriParams = AscendC::FixpipeParamsV220(128,   // nSize
                                                          m_mad, // mSize
                                                          m_mad, // srcStride
                                                          128,   // dstStride
                                                          false  // reluEn
            );
            intriParams.quantPre = QuantMode_t::NoQuant;
            intriParams.unitFlag = unit_flag;
            AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
                gm_c_cube1_tensor[gm_c - this->gm_c_cube1], // dstGlobal
                l0_c_buf_tensor,                            // srcLocal
                intriParams                                 // intriParams
            );
            this->l1_b_ping_pong_flag_ = 1 - this->l1_b_ping_pong_flag_;

            gm_k += BASE_BLOCK_SIZE_256;
            gm_c += BASE_BLOCK_SIZE;
        }
    }
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ __inline__ void CubeForward_192<TYPE, IF_BF16, WORKSPACE_TYPE>::cube1_headDim192(
    Address_192::PHY_ADDR_FORWARD_CUBE1<TYPE, TYPE, WORKSPACE_TYPE> *src,
    Address_192::PHY_ADDR_FORWARD_CUBE1<TYPE, TYPE, WORKSPACE_TYPE> *remain)
{
    int32_t switch_flag = 0;
    int32_t l0a_offset_remain = 0;
    int32_t loop_times_Q = 2; // 循环两次搬运Q基本块（非mask场景1块，mask场景liang）
    int32_t headDim = 192;

    auto l1_base_a_cube1_tensor = this->l1_a_ping_pong_flag_ ? this->l1_a_pong_tensor : this->l1_a_ping_tensor;

    WAIT_FLAG(MTE1, MTE2, this->l1_a_ping_pong_flag_);
    for (int i = 0; i < loop_times_Q; i++) {
        if (src[i].k == 0) {
            continue;
        }
        AscendC::DataCopy(l1_base_a_cube1_tensor[BASE_BLOCK_SIZE_192 * i],
                          this->gm_Q_tensor[(uint32_t(src[i].left - this->gm_a_cube1))],
                          AscendC::Nd2NzParams(1,       // ndNum
                                               128,     // nValue
                                               headDim, // dValue
                                               0,       // srcNdMatrixStride, unused
                                               headDim, // srcDValue
                                               128,     // dstNzC0Stride
                                               1,       // dstNzNStride
                                               0        // dstNzMatrixStride, unused
                                               ));
        switch_flag++;
    }

    if (switch_flag == 2) {
        l0a_offset_remain = BASE_BLOCK_SIZE_192;
    } else if (src[0].k == 0 && src[1].k == 0) {
        for (int i = 0; i < 2; i++) {
            if (remain[i].k == 0) {
                continue;
            }
            AscendC::DataCopy(l1_base_a_cube1_tensor[BASE_BLOCK_SIZE_192 * i],
                              this->gm_Q_tensor[(uint32_t(remain[i].left - this->gm_a_cube1))],
                              AscendC::Nd2NzParams(1,       // ndNum
                                                   128,     // nValue
                                                   headDim, // dValue
                                                   0,       // srcNdMatrixStride, unused
                                                   headDim, // srcDValue
                                                   128,     // dstNzC0Stride
                                                   1,       // dstNzNStride
                                                   0        // dstNzMatrixStride, unused
                                                   ));
            switch_flag++;
        }
    } else if (remain[0].k != 0 && src[0].left != remain[0].left) {
        l0a_offset_remain = BASE_BLOCK_SIZE_192;
        AscendC::DataCopy(l1_base_a_cube1_tensor[BASE_BLOCK_SIZE_192],
                          this->gm_Q_tensor[(uint32_t(remain[0].left - this->gm_a_cube1))],
                          AscendC::Nd2NzParams(1,       // ndNum
                                               128,     // nValue
                                               headDim, // dValue
                                               0,       // srcNdMatrixStride, unused
                                               headDim, // srcDValue
                                               128,     // dstNzC0Stride
                                               1,       // dstNzNStride
                                               0        // dstNzMatrixStride, unused
                                               ));
        switch_flag++;
    } else if (remain[1].k != 0) {
        AscendC::DataCopy(l1_base_a_cube1_tensor[BASE_BLOCK_SIZE_192],
                          this->gm_Q_tensor[(uint32_t(remain[1].left - this->gm_a_cube1))],
                          AscendC::Nd2NzParams(1,       // ndNum
                                               128,     // nValue
                                               headDim, // dValue
                                               0,       // srcNdMatrixStride, unused
                                               headDim, // srcDValue
                                               128,     // dstNzC0Stride
                                               1,       // dstNzNStride
                                               0        // dstNzMatrixStride, unused
                                               ));
        switch_flag++;
    }

    SET_FLAG(MTE2, MTE1, this->l1_a_ping_pong_flag_);
    WAIT_FLAG(MTE2, MTE1, this->l1_a_ping_pong_flag_);

    // 处理完整基本块
    base_block_mad_headDim192(src[0], src[1]);
    // 处理尾块
    if (remain[0].k + remain[1].k != 0) {
        base_block_mad_headDim192(remain[0], remain[1], l0a_offset_remain);
    }

    SET_FLAG(MTE1, MTE2, this->l1_a_ping_pong_flag_);
    this->l1_a_ping_pong_flag_ = 1 - this->l1_a_ping_pong_flag_;
}


template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ inline void CubeForward_192<TYPE, IF_BF16, WORKSPACE_TYPE>::addrMapping_cube2_rowsum_192(
    int32_t roundId, int64_t &src_length, int64_t &remain_length, PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *src,
    PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *remain)
{
    int32_t curCalcRow = roundId * this->F + this->core_group_index; // 当前的row
    int32_t curCalcRowP = this->F * (roundId % 2) + this->core_group_index;
    int32_t b = curCalcRow / this->row_per_batch;                  // 当前行所在的batch号
    int32_t n = (curCalcRow % this->row_per_batch) / this->L;      // 当前batch下的head号
    int32_t i_r = (curCalcRow % this->row_per_batch) % this->L;    // 当前head下的行号
    int32_t i_c = this->local_block_index * this->column_per_core; // 当前core处理的起始列
    int32_t begin_remain = this->sparseMode == 1 ? (this->W + 1 - this->column_remain) :
                                                   (this->H1) + 1 - this->column_remain; // sparse or triangle

    int32_t q_left_index = this->sparseMode == 1 ? (this->W / 2 + i_r) : (this->L + i_r);          // sparse or triangle
    int32_t q_right_index = this->sparseMode == 1 ? (this->W / 2 - 1 - i_r) : (this->L - 1 - i_r); // sparse or triangle
    int32_t switch_index = q_left_index;

    // 设置sparse的标志位 以及 sparse场景非倒三角部分的偏移
    int32_t sparse_flag = false;
    if (this->sparseMode == 1) {
        sparse_flag = i_r > ((this->W - 1) / 2) ? true : false;
    }
    int32_t row_down = i_r + this->W / 2;
    int32_t col_down = i_r + i_c - this->W / 2;
    int32_t col_down_remain = i_r + begin_remain - this->W / 2;

    int32_t g = n / this->nG;
    int64_t bn_offset = (b * this->N * this->H1 + n * this->H1) * BASE_BLOCK_SIZE;
    int64_t bn_offset_rowsum = (b * this->N * this->H1 + n * this->H1) * 128;
    int64_t bn_offset_GQA = (b * this->G * this->H1 + g * this->H1) * BASE_BLOCK_SIZE;
    int64_t q_left_offset = bn_offset + q_left_index * BASE_BLOCK_SIZE;
    int64_t q_right_offset = bn_offset + q_right_index * BASE_BLOCK_SIZE;
    int64_t out_offset = this->sparseMode == 1 ? ((curCalcRowP * (this->W + 1)) * BASE_BLOCK_SIZE) // sparse or triangle
                                                 :
                                                 ((curCalcRowP * (this->H1 + 1)) * BASE_BLOCK_SIZE);

    if (!sparse_flag && switch_index < i_c) {
        src[0].left = this->gm_c_cube1 + out_offset + i_c * BASE_BLOCK_SIZE;
        src[0].right = this->gm_b_cube2 + bn_offset_GQA + (i_c - switch_index - 1) * BASE_BLOCK_SIZE;
        src[0].out = this->gm_c_cube2 + bn_offset + q_right_index * BASE_BLOCK_SIZE;
        src[0].rowsumOut = this->rowsum_gm + bn_offset_rowsum + q_right_index * 128;
        src[0].k = this->column_per_core;
        src_length = 1;

        if (this->column_remain != 0) {
            remain[0].left = this->gm_c_cube1 + out_offset + begin_remain * BASE_BLOCK_SIZE +
                             (128 / this->Y) * this->local_block_index * 128;
            remain[0].right = this->gm_b_cube2 + bn_offset_GQA + (begin_remain - switch_index - 1) * BASE_BLOCK_SIZE;
            remain[0].out = src[0].out + (128 / this->Y) * this->local_block_index * 128;
            remain[0].rowsumOut = src[0].rowsumOut + (128 / this->Y) * this->local_block_index;
            remain[0].k = this->column_remain;
            remain_length = 1;
        }
    } else if (!sparse_flag && i_c <= switch_index && i_c + this->column_per_core > switch_index + 1) {
        src[0].left = this->gm_c_cube1 + out_offset + i_c * BASE_BLOCK_SIZE;
        src[0].right = this->gm_b_cube2 + bn_offset_GQA + i_c * BASE_BLOCK_SIZE;
        src[0].out = this->gm_c_cube2 + bn_offset + q_left_index * BASE_BLOCK_SIZE;
        src[0].rowsumOut = this->rowsum_gm + bn_offset_rowsum + q_left_index * 128;
        src[0].k = switch_index - i_c + 1;

        src[1].left = this->gm_c_cube1 + out_offset + i_c * BASE_BLOCK_SIZE + src[0].k * BASE_BLOCK_SIZE;
        src[1].right = this->gm_b_cube2 + bn_offset_GQA;
        src[1].out = this->gm_c_cube2 + bn_offset + q_right_index * BASE_BLOCK_SIZE;
        src[1].rowsumOut = this->rowsum_gm + bn_offset_rowsum + q_right_index * 128;
        src[1].k = this->column_per_core - src[0].k;

        src_length = 2;

        if (this->column_remain != 0) {
            remain[0].left = this->gm_c_cube1 + out_offset + begin_remain * BASE_BLOCK_SIZE +
                             (128 / this->Y) * this->local_block_index * 128;
            remain[0].right = this->gm_b_cube2 + bn_offset_GQA + (begin_remain - switch_index - 1) * BASE_BLOCK_SIZE;
            remain[0].out = src[1].out + (128 / this->Y) * this->local_block_index * 128;
            remain[0].rowsumOut = src[1].rowsumOut + (128 / this->Y) * this->local_block_index;
            remain[0].k = this->column_remain;
            remain_length = 1;
        }
    } else if (!sparse_flag && i_c <= switch_index && i_c + this->column_per_core <= switch_index + 1) {
        src[0].left = this->gm_c_cube1 + out_offset + i_c * BASE_BLOCK_SIZE;
        src[0].right = this->gm_b_cube2 + bn_offset_GQA + i_c * BASE_BLOCK_SIZE;
        src[0].out = this->gm_c_cube2 + bn_offset + q_left_index * BASE_BLOCK_SIZE;
        src[0].rowsumOut = this->rowsum_gm + bn_offset_rowsum + q_left_index * 128;
        src[0].k = this->column_per_core;
        src_length = 1;

        if (this->column_remain != 0) {
            if (switch_index < begin_remain) {
                remain[0].left = this->gm_c_cube1 + out_offset + begin_remain * BASE_BLOCK_SIZE +
                                 (128 / this->Y) * this->local_block_index * 128;
                remain[0].right =
                    this->gm_b_cube2 + bn_offset_GQA + (begin_remain - switch_index - 1) * BASE_BLOCK_SIZE;
                remain[0].out = this->gm_c_cube2 + bn_offset + q_right_index * BASE_BLOCK_SIZE +
                                (128 / this->Y) * this->local_block_index * 128;
                remain[0].rowsumOut = this->rowsum_gm + bn_offset_rowsum + q_right_index * 128 +
                                      (128 / this->Y) * this->local_block_index;
                remain[0].k = this->column_remain;
                remain_length = 1;
            } else {
                remain[0].left = this->gm_c_cube1 + out_offset + begin_remain * BASE_BLOCK_SIZE +
                                 (128 / this->Y) * this->local_block_index * 128;
                remain[0].right = this->gm_b_cube2 + bn_offset_GQA + begin_remain * BASE_BLOCK_SIZE;
                remain[0].out = this->gm_c_cube2 + bn_offset + q_left_index * BASE_BLOCK_SIZE +
                                (128 / this->Y) * this->local_block_index * 128;
                remain[0].rowsumOut =
                    this->rowsum_gm + bn_offset_rowsum + q_left_index * 128 + (128 / this->Y) * this->local_block_index;
                remain[0].k = switch_index - begin_remain + 1;

                remain[1].left = remain[0].left + remain[0].k * BASE_BLOCK_SIZE;
                remain[1].right = this->gm_b_cube2 + bn_offset_GQA;
                remain[1].out = this->gm_c_cube2 + bn_offset + q_right_index * BASE_BLOCK_SIZE +
                                (128 / this->Y) * this->local_block_index * 128;
                remain[1].rowsumOut = this->rowsum_gm + bn_offset_rowsum + q_right_index * 128 +
                                      (128 / this->Y) * this->local_block_index;
                remain[1].k = this->column_remain - remain[0].k;

                remain_length = 2;
            }
        }
    } else {
        src[0].left = this->gm_c_cube1 + out_offset + i_c * BASE_BLOCK_SIZE;
        src[0].right = this->gm_b_cube2 + bn_offset_GQA + col_down * BASE_BLOCK_SIZE;
        src[0].out = this->gm_c_cube2 + bn_offset + row_down * BASE_BLOCK_SIZE;
        src[0].rowsumOut = this->rowsum_gm + bn_offset_rowsum + row_down * 128;
        src[0].k = this->column_per_core;
        src_length = 1;

        if (this->column_remain != 0) {
            remain[0].left = this->gm_c_cube1 + out_offset + begin_remain * BASE_BLOCK_SIZE +
                             (128 / this->Y) * this->local_block_index * 128;
            remain[0].right = this->gm_b_cube2 + bn_offset_GQA + col_down_remain * BASE_BLOCK_SIZE;
            remain[0].out = src[0].out + (128 / this->Y) * this->local_block_index * 128;
            remain[0].rowsumOut = src[0].rowsumOut + (128 / this->Y) * this->local_block_index;
            remain[0].k = this->column_remain;
            remain_length = 1;
        }
    }
}


template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ inline void CubeForward_192<TYPE, IF_BF16, WORKSPACE_TYPE>::addrMapping_cube2_rowsum_nomask_192(
    int32_t roundId, int64_t &src_length, int64_t &remain_length, PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *src,
    PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *remain)
{
    int32_t curCalcRow = roundId * this->F + this->core_group_index; // 当前的row 0
    int32_t curCalcRowP = this->F * (roundId % 2) + this->core_group_index;
    int32_t b = curCalcRow / this->row_per_batch;                  // 当前行所在的batch号
    int32_t n = (curCalcRow % this->row_per_batch) / this->L;      // 当前batch下的head号
    int32_t i_r = (curCalcRow % this->row_per_batch) % this->L;    // 当前head下的行号
    int32_t i_c = this->local_block_index * this->column_per_core; // 当前core处理的起始列
    int32_t begin_remian = this->H2 - this->column_remain;

    int32_t q_left_index = i_r;
    int32_t q_right_index = this->L - 1 - i_r;
    int32_t switch_index = this->H2;

    int32_t g = n / this->nG;
    int64_t bn_offset_1 = (b * this->N * this->H1 + n * this->H1) * BASE_BLOCK_SIZE;
    int64_t bn_offset_rowsum = (b * this->N * this->H1 + n * this->H1) * 128;
    int64_t bn_offset_2 = (b * this->G * this->H2 + g * this->H2) * BASE_BLOCK_SIZE; // GQA version of bn_offset_2
    int64_t out_offset = (curCalcRowP * this->H2) * BASE_BLOCK_SIZE;

    src[0].left = this->gm_c_cube1 + out_offset + i_c * BASE_BLOCK_SIZE;
    src[0].right = this->gm_b_cube2 + bn_offset_2 + i_c * BASE_BLOCK_SIZE;
    src[0].out = this->gm_c_cube2 + bn_offset_1 + q_left_index * BASE_BLOCK_SIZE;
    src[0].rowsumOut = this->rowsum_gm + bn_offset_rowsum + q_left_index * 128;
    src[0].k = this->column_per_core;
    src_length = 1;
    if (this->column_remain != 0) {
        remain[0].left = this->gm_c_cube1 + out_offset + begin_remian * BASE_BLOCK_SIZE +
                         (128 / this->Y) * this->local_block_index * 128;
        remain[0].right = this->gm_b_cube2 + bn_offset_2 + begin_remian * BASE_BLOCK_SIZE;
        remain[0].out = this->gm_c_cube2 + bn_offset_1 + q_left_index * BASE_BLOCK_SIZE +
                        (128 / this->Y) * this->local_block_index * 128;
        remain[0].rowsumOut =
            this->rowsum_gm + bn_offset_rowsum + q_left_index * 128 + (128 / this->Y) * this->local_block_index;
        remain[0].k = this->column_remain;
        remain_length = 1;
    }
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ inline void CubeForward_192<TYPE, IF_BF16, WORKSPACE_TYPE>::unified_addrMapping_cube2mix_192(
    int32_t roundId, int64_t &src_length, int64_t &remain_length, PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *src,
    PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *remain)
{
    if (this->qk_triangle == 1) { // 倒三角
        addrMapping_cube2_rowsum_192(roundId, src_length, remain_length, src, remain);
    } else if (this->qk_triangle == 0) { // 非倒三角
        addrMapping_cube2_rowsum_nomask_192(roundId, src_length, remain_length, src, remain);
    }
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ inline void CubeForward_192<TYPE, IF_BF16, WORKSPACE_TYPE>::mix_cube2_rowsum_192(int32_t cube2_roundId)
{
    // 混算寻址：
    PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> src[2];
    PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> remain[2];
    int64_t src_length = 0;
    int64_t remain_length = 0;
    unified_addrMapping_cube2mix_192(cube2_roundId, src_length, remain_length, src, remain); // cube2和rowsum的寻址

    // 开始混算：
    cube2_rowsum_mix_only_192(src, src_length, BASE_BLOCK_LENGTH, this->Y * 2);
    if (remain_length > 0) {
        cube2_rowsum_mix_only_192(remain, remain_length, BASE_BLOCK_LENGTH / this->Y, 2);
    }
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ inline void CubeForward_192<TYPE, IF_BF16, WORKSPACE_TYPE>::cube2_rowsum_mix_only_192(
    const PhyAddrCube2Rowsum<TYPE, WORKSPACE_TYPE> *cube_addr, int32_t cube_length, int32_t m_length,
    int32_t vcore_num_per_head)
{
    int32_t l1_m = CUBE2_LENGTH_M; // 128
    int32_t l1_k = CUBE2_LENGTH_K;
    int32_t l1_n = CUBE2_LENGTH_N;
    int32_t l0_k = BASE_BLOCK_LENGTH; // 128
    int32_t l0_m = BASE_BLOCK_LENGTH; // 128
    int32_t l0_n = BASE_BLOCK_LENGTH; // 128

    for (int32_t i = 0; i < cube_length; i++) {
        __gm__ TYPE *__restrict__ left_start_addr = (__gm__ TYPE *)(cube_addr[i].left); // 高精度是需要将float转为TYPE
        __gm__ TYPE *__restrict__ right_start_addr = cube_addr[i].right;
        __gm__ float *__restrict__ result_gm = cube_addr[i].out;
        __gm__ float *__restrict__ rowsum_out_gm = cube_addr[i].rowsumOut;

        int32_t k_core_gm = cube_addr[i].k * l0_k;
        l1_k = k_core_gm < l1_k ? k_core_gm : l1_k;
        int32_t loop_l1_times = k_core_gm / l1_k;
        int32_t l0_k_block_num = l0_k / BLOCK_SIZE;     // 128/16
        int32_t l0_m_block_num = m_length / BLOCK_SIZE; // 128/16
        int32_t l0_n_block_num = l0_n / BLOCK_SIZE;     // 8

        for (int32_t l1_loop_index = 0; l1_loop_index < loop_l1_times; l1_loop_index++) {
            auto l1_buf_a_cube2_tensor = l1_a_ping_pong_flag_ ? l1_a_pong_tensor : l1_a_ping_tensor;
            auto l1_buf_b_cube2_tensor = l1_b_ping_pong_flag_ ? l1_b_pong_tensor : l1_b_ping_tensor;
            auto l0a_buf_tensor = l0_a_ping_pong_flag_ ? l0_a_pong_tensor : l0_a_ping_tensor;
            auto l0b_buf_tensor = l0_b_ping_pong_flag_ ? l0_b_pong_tensor : l0_b_ping_tensor;
            auto l0c_buf_tensor = l0_c_ping_pong_flag_ ? l0_c_pong_tensor : l0_c_ping_tensor;
            WAIT_FLAG(MTE1, MTE2, l1_a_ping_pong_flag_);

            AscendC::DataCopy(
                l1_buf_a_cube2_tensor,
                this->gm_c_cube1_tensor_type[(uint32_t(left_start_addr - (__gm__ TYPE *)this->gm_c_cube1) +
                                              l1_loop_index * l0_m * l0_k * 2)],
                AscendC::Nd2NzParams(vcore_num_per_head, m_length / vcore_num_per_head, l0_k,
                                     m_length * l0_k / vcore_num_per_head * 2, l0_k, m_length, 1,
                                     m_length / vcore_num_per_head * BLOCK_SIZE));

            SET_FLAG(MTE2, MTE1, l1_a_ping_pong_flag_);
            WAIT_FLAG(MTE2, MTE1, l1_a_ping_pong_flag_);
            WAIT_FLAG(M, MTE1, l0_a_ping_pong_flag_);

            for (int32_t l0a_load_idx = 0; l0a_load_idx < l0_k_block_num; l0a_load_idx++) {
                AscendC::LoadData(l0a_buf_tensor[l0a_load_idx * CUBE_MATRIX_SIZE],
                                  l1_buf_a_cube2_tensor[l0a_load_idx * l0_m_block_num * CUBE_MATRIX_SIZE],
                                  AscendC::LoadData2dParams(0,                  // startIndex
                                                            l0_m_block_num,     // repeatTimes
                                                            1,                  // srcStride
                                                            0,                  // sid
                                                            l0_k_block_num - 1, // dstGap
                                                            false,              // ifTranspose
                                                            0                   // addrMode
                                                            ));
            }

            SET_FLAG(MTE1, MTE2, l1_a_ping_pong_flag_);
            WAIT_FLAG(MTE1, MTE2, l1_b_ping_pong_flag_ + 2);

            AscendC::DataCopy(
                l1_buf_b_cube2_tensor,
                this->gm_V_tensor[(uint32_t(right_start_addr - this->gm_b_cube2) + l1_loop_index * l0_k * l0_n)],
                AscendC::Nd2NzParams(1,           // ndNum
                                     l0_k,        // nValue
                                     l0_n,        // dValue
                                     l0_k * l0_n, // srcNdMatrixStride
                                     l0_n,        // srcDValue
                                     l0_k,        // dstNzC0Stride
                                     1,           // dstNzNStride
                                     l0_k * l0_n  // dstNzMatrixStride
                                     ));

            SET_FLAG(MTE2, MTE1, l1_b_ping_pong_flag_ + 2);
            WAIT_FLAG(MTE2, MTE1, l1_b_ping_pong_flag_ + 2);
            WAIT_FLAG(M, MTE1, l0_b_ping_pong_flag_ + 2);

            for (int32_t l0b_load_idx = 0; l0b_load_idx < l0_k_block_num; l0b_load_idx++) {
                AscendC::LoadData(l0b_buf_tensor[l0b_load_idx * l0_n * BLOCK_SIZE],
                                  l1_buf_b_cube2_tensor[l0b_load_idx * CUBE_MATRIX_SIZE],
                                  AscendC::LoadData2dParams(0,              // startIndex
                                                            l0_n_block_num, // repeatTimes
                                                            l0_k_block_num, // srcStride
                                                            0,              // sid
                                                            0,              // dstGap
                                                            true,           // ifTranspose
                                                            0               // addrMode
                                                            ));
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
            AscendC::Mmad(l0c_buf_tensor,                                  // dstLocal
                          l0a_buf_tensor,                                  // fmLocal
                          l0b_buf_tensor,                                  // filterLocal
                          AscendC::MmadParams(m_length,                    // m
                                              l0_n,                        // n
                                              l0_k,                        // k
                                              unit_flag,                   // unitFlag
                                              false,                       // cmatrixSource
                                              (init_c == 1) ? true : false // cmatrixInitVal
                                              ));

            SET_FLAG(M, MTE1, l0_b_ping_pong_flag_ + 2);

            if (out_c) {
                AscendC::SetAtomicAdd<float>();
                AscendC::SetAtomicType<float>();
                auto intriParams = AscendC::FixpipeParamsV220(l0_n,     // nSize
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
            l0b_buf_tensor = l0_b_ping_pong_flag_ ? l0_b_pong_tensor : l0_b_ping_tensor;
            l0c_buf_tensor = l0_c_ping_pong_flag_ ? l0_c_pong_tensor : l0_c_ping_tensor;
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

#endif

#endif