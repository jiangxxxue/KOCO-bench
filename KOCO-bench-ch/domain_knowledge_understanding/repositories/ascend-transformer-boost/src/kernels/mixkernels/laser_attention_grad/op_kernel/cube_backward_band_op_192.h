/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef __CUBEBACKWARD_BAND_OP_192_H__
#define __CUBEBACKWARD_BAND_OP_192_H__
#ifdef __DAV_C220_CUBE__

#include "address_mapping_cube.h"
#include "ppmatmul_const_grad.h"
#include "kernel_operator.h"
#include "cube_backward_band_op.h"

namespace CUBE_BACK_BAND_OP_192 {

constexpr int32_t src_stride_K = 256;
constexpr int32_t src_stride_Q = 192;
constexpr int32_t SIZE_16 = 16;
constexpr int32_t SIZE_32 = 32;
constexpr int32_t SIZE_256 = 256;
constexpr int32_t SIZE_128 = 128;
constexpr int32_t SIZE_192 = 192;

template <typename TYPE, bool IF_BF16>
class CubeBackwardBandOp192 : public CUBE_BACK_BAND_OP::CubeBackwardBandOp<TYPE, IF_BF16> {
public:
    __aicore__ inline CubeBackwardBandOp192() {}

    __aicore__ __inline__ void Run_op();

private:
    template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
    __aicore__ inline __attribute__((always_inline)) void cube1_matmul_op_192(
        const Address::PhyAddrBackwardCube1<T_LEFT, T_RIGHT, T_OUTPUT> *src, int64_t src_len);

    __aicore__ __inline__ void cube1_base_matmul_headDim192(
        LocalTensor<TYPE> *l1_a_tensor, LocalTensor<TYPE> *l1_b_tensor,
        __gm__ float *gm_out, int32_t ky, int32_t out_put_matrix_line_strid, bool upper_right_flag);

    template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
    __aicore__ __inline__ void cube2_matmul_op_192_base(
        const Address::PhyAddrBackwardCube2<T_LEFT, T_RIGHT, T_OUTPUT> *src,
        int64_t src_len, int64_t vcore_num_per_head, uint64_t n_offset, uint64_t n_step, uint64_t n_size);

    template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
    __aicore__ __inline__ void cube2_matmul_op_192_left(
        const Address::PhyAddrBackwardCube2<T_LEFT, T_RIGHT, T_OUTPUT> *src,
        int64_t src_len, int64_t vcore_num_per_head);

    template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
    __aicore__ __inline__ void cube3_matmul_op_192_left(
        const Address::PhyAddrBackwardCube3<T_LEFT, T_RIGHT, T_OUTPUT> *src,
        int64_t src_len, int64_t vcore_num_per_head);
    
    __aicore__ __inline__ void Mat_mix_cube2_cube3_op_192_right(int64_t roundId);
    template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
    
    __aicore__ inline __attribute__((always_inline)) void cube2_cube3_mix_op_192_right(
        const Address::PhyAddrBackwardCube2<T_LEFT, T_RIGHT, T_OUTPUT> *src_cube2, int64_t src_length,
        const Address::PhyAddrBackwardCube3<T_LEFT, T_RIGHT, T_OUTPUT> *src_cube3, int64_t vcore_num_per_head);

    __aicore__ __inline__ void cube2_base_matmul(LocalTensor<TYPE> *l1_a_buf_tensor, LocalTensor<TYPE> &l1_b_buf_tensor,
        __gm__ float *result_gm, int64_t Kx_start, int64_t Kx_end, int64_t Ky_start, int64_t Ky_end,
        int64_t Ky, int64_t Kx, uint64_t n_offset, uint64_t n_step, uint64_t n_size, bool upperRight, bool is_skip);

    __aicore__ __inline__ void cube3_base_matmul(LocalTensor<TYPE> *l1_a_buf_tensor, LocalTensor<TYPE> *l1_b_buf_tensor,
        __gm__ float *result_gm, int64_t Kx_start, int64_t Kx_end, int64_t Ky_start, int64_t Ky_end,
        int64_t Ky, int64_t Kx, uint64_t n_offset, uint64_t n_step, uint64_t n_size, bool is_skip, bool is_skip_up);

private:
    AscendC::LoadData2dParams commonLoadData2dParamsTranspose {
        0,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        0,
        0,
        true,
        0
    };

    AscendC::LoadData2dParams commonLoadData2dParamsNoTranspose {
        0,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        0,
        0,
        false,
        0
    };

    AscendC::Nd2NzParams commonNd2NzParams {
        1,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        0,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        1,
        0
    };
};


template <typename TYPE, bool IF_BF16>
template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
__aicore__ inline __attribute__((always_inline)) void CubeBackwardBandOp192<TYPE, IF_BF16>::cube1_matmul_op_192(
    const Address::PhyAddrBackwardCube1<T_LEFT, T_RIGHT, T_OUTPUT> *src, int64_t src_len)
{
    for (int32_t idx = 0; idx < src_len; ++idx) {
        auto kx = src[idx].kx;
        auto ky = src[idx].ky;

        auto lineStride = src[idx].lineStride;
        int32_t m_loop = 1;
        int32_t n_loop = kx;
        auto l1_m_size_ = ky * 128;

        auto l1_k_size_ = 192;
        auto l1_n_size_ = 192;
        auto k_ = 128;
        auto n_ = 128;

        auto gm_a = src[idx].left;
        auto gm_b = src[idx].right;
        auto gm_c = src[idx].out;
        bool upperRight = src[idx].upperRight;
        bool lowerLeft = src[idx].lowerLeft;
        LocalTensor<TYPE> *l1_a_tensor = this->ping_pong_flag_l1_a_ ? &this->l1_a_pong_tensor : &this->l1_a_ping_tensor;

        WAIT_FLAG(MTE1, MTE2, this->ping_pong_flag_l1_a_ + 2);
        this->temp_tensor_bf16.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(gm_a));
        commonNd2NzParams.nValue = l1_m_size_;
        commonNd2NzParams.dValue = l1_k_size_;
        commonNd2NzParams.srcDValue = l1_k_size_;
        commonNd2NzParams.dstNzC0Stride = SIZE_128 * 2;
        AscendC::DataCopy(
            *l1_a_tensor,
            this->temp_tensor_bf16,
            commonNd2NzParams
        );
        SET_FLAG(MTE2, MTE1, this->ping_pong_flag_l1_a_ + 2);
        WAIT_FLAG(MTE2, MTE1, this->ping_pong_flag_l1_a_ + 2);

        commonNd2NzParams.nValue = SIZE_128;
        commonNd2NzParams.dValue = src_stride_K;
        commonNd2NzParams.srcDValue = src_stride_K;
        commonNd2NzParams.dstNzC0Stride = SIZE_128;
        
        for (int n_index = 0; n_index < n_loop; n_index++) {
            bool upper_right_flag = (upperRight && n_index == n_loop - 1);
            LocalTensor<TYPE>* l1_b_tensor = this->ping_pong_flag_l1_b_ ?
                    &this->l1_b_pong_tensor : &this->l1_b_ping_tensor;

            WAIT_FLAG(MTE1, MTE2, this->ping_pong_flag_l1_b_);
            this->temp_tensor_bf16.SetGlobalBuffer(
                reinterpret_cast<__gm__ TYPE *>(gm_b + n_index * src_stride_K * 128));
            AscendC::DataCopy(
                *l1_b_tensor,
                this->temp_tensor_bf16,
                commonNd2NzParams
            );

            SET_FLAG(MTE2, MTE1, this->ping_pong_flag_l1_b_);
            WAIT_FLAG(MTE2, MTE1, this->ping_pong_flag_l1_b_);

            cube1_base_matmul_headDim192(l1_a_tensor, l1_b_tensor, gm_c, ky, lineStride, upper_right_flag);

            SET_FLAG(MTE1, MTE2, this->ping_pong_flag_l1_b_);
            this->ping_pong_flag_l1_b_ = 1 - this->ping_pong_flag_l1_b_;

            gm_c += SIZE_128 * SIZE_128;
        }
        SET_FLAG(MTE1, MTE2, this->ping_pong_flag_l1_a_ + 2);
        this->ping_pong_flag_l1_a_ = 1 - this->ping_pong_flag_l1_a_;
    }
}


template <typename TYPE, bool IF_BF16>
__aicore__ __inline__ void CubeBackwardBandOp192<TYPE, IF_BF16>::cube1_base_matmul_headDim192(
    LocalTensor<TYPE> *l1_a_tensor, LocalTensor<TYPE> *l1_b_tensor,
    __gm__ float *gm_out, int32_t ky, int32_t out_put_matrix_line_strid, bool upper_right_flag)
{
    auto l1_n_size_ = 192;
    auto l1_m_size_ = ky * 128;
    auto k0_ = 192;
    auto n0_ = 128;
    auto m0_ = 128;
    auto n_ = 128;
    int32_t head_dim = 192;
    int32_t head_dim_half = 192 / 2;

    for (int idx = 0; idx < 2; idx += 1) {
        int m_offset = idx * 128;
        bool l0_skip_flag = (upper_right_flag && m_offset == 0);
        LocalTensor<float>* l0_c_buf_tensor = this->ping_pong_flag_l0_c_ ?
                &this->l0_c_pong_tensor : &this->l0_c_ping_tensor;

        for (int32_t m = 0; m < 2; m++) {
            LocalTensor<TYPE>* l0_a_buf_tensor = this->ping_pong_flag_l0_a_ ?
                    &this->l0_a_pong_tensor : &this->l0_a_ping_tensor;

            WAIT_FLAG(M, MTE1, this->ping_pong_flag_l0_a_);
            if (!l0_skip_flag) {
                int32_t l0a_oneline_offset = 128;
                if (m == 1) {
                    l0a_oneline_offset = head_dim - 128;
                }
                int32_t repeat_times = l0a_oneline_offset / SIZE_16;
                commonLoadData2dParamsNoTranspose.repeatTimes = l0a_oneline_offset / SIZE_16;
                commonLoadData2dParamsNoTranspose.srcStride = 16;
                for (int32_t i = 0; i < m0_ / SIZE_16; i++) {
                    AscendC::LoadData(
                        (*l0_a_buf_tensor)[i * l0a_oneline_offset * SIZE_16],
                        (*l1_a_tensor)[idx * 128 * 16 + m * 256 * 128 + i * SIZE_256],
                        commonLoadData2dParamsNoTranspose
                    );
                }
            }

            LocalTensor<TYPE>* l0_b_buf_tensor = this->ping_pong_flag_l0_b_ ?
                    &this->l0_b_pong_tensor : &this->l0_b_ping_tensor;

            WAIT_FLAG(M, MTE1, this->ping_pong_flag_l0_b_ + 2);
            if (!l0_skip_flag) {
                int32_t l0b_col_num = 128;
                if (m == 1) {
                    l0b_col_num = head_dim - 128;
                }
                commonLoadData2dParamsNoTranspose.repeatTimes = l0b_col_num * n0_ / SIZE_256;
                commonLoadData2dParamsNoTranspose.srcStride = 1;
                AscendC::LoadData(
                    *l0_b_buf_tensor,
                    (*l1_b_tensor)[m * 128 * 128],
                    commonLoadData2dParamsNoTranspose
                );
            }
            SET_FLAG(MTE1, M, this->ping_pong_flag_l0_b_ + 2);
            WAIT_FLAG(MTE1, M, this->ping_pong_flag_l0_b_ + 2);

            // 搬出等于0b11，累加等于0b10
            int unit_flag = 0b10;
            if (m == 1) {
                unit_flag = 0b11;
            }
            bool init_c = (m == 0);
            int32_t l0c_col_num = 128;
            if (m == 1) {
                l0c_col_num = head_dim - 128;
            }
            if (!l0_skip_flag) {
                AscendC::Mmad(
                    *l0_c_buf_tensor,
                    *l0_a_buf_tensor,
                    *l0_b_buf_tensor,
                    AscendC::MmadParams(
                        m0_,
                        n0_,
                        l0c_col_num,
                        unit_flag,
                        false,
                        init_c
                    )
                );
            }
            SET_FLAG(M, MTE1, this->ping_pong_flag_l0_b_ + 2);
            SET_FLAG(M, MTE1, this->ping_pong_flag_l0_a_);
            this->ping_pong_flag_l0_b_ = 1 - this->ping_pong_flag_l0_b_;
            this->ping_pong_flag_l0_a_ = 1 - this->ping_pong_flag_l0_a_;
        }

        auto out_offset = (m_offset / 128) * out_put_matrix_line_strid;
        
        if (!l0_skip_flag) {
            auto intriParams = AscendC::FixpipeParamsV220(
                n0_,
                m0_,
                m0_,
                n_,
                false
            );
            intriParams.quantPre = QuantMode_t::NoQuant;
            intriParams.unitFlag = 3;

            this->temp_tensor_fp32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_out + out_offset));
            AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
                this->temp_tensor_fp32,
                *l0_c_buf_tensor,
                intriParams
            );
        }
        this->ping_pong_flag_l0_c_ = 1 - this->ping_pong_flag_l0_c_;
    }
}


template <typename TYPE, bool IF_BF16>
__aicore__ __inline__ void CubeBackwardBandOp192<TYPE, IF_BF16>::cube2_base_matmul(
    LocalTensor<TYPE> *l1_a_buf_tensor, LocalTensor<TYPE> &l1_b_buf_tensor, __gm__ float *result_gm,
    int64_t Kx_start, int64_t Kx_end, int64_t Ky_start, int64_t Ky_end, int64_t Ky, int64_t Kx,
    uint64_t n_offset, uint64_t n_step, uint64_t n_size, bool upperRight, bool is_skip)
{
    n_offset = 128;
    n_step = 64;
    n_size = 192;
    int64_t l1_m = CUBE2_LENGTH_M;               // 128
    int64_t l1_k = CUBE2_LENGTH_K;               // 128
    int64_t l1_n = n_step;                       // 128
    int64_t l0_k = BASE_BLOCK_LENGTH;            // 128
    int64_t l0_m = BASE_BLOCK_LENGTH;            // 128
    int64_t l0_n = BASE_BLOCK_LENGTH;            // 128
    int64_t l0_k_block_num = l0_k / BLOCK_SIZE;  // 16
    int64_t l0_n_block_num = l0_n / BLOCK_SIZE;  // 16

    int64_t SIZE_256 = 256;
    int64_t SIZE_128 = 128;
    int64_t SIZE_16 = 16;
    auto n_ = n_size;
    auto n0_ = n_step;
    auto m0_ = 128;
    auto k0_ = 128;

    for (int32_t i = Kx_start; i < Kx_end; i++) {
        bool l1_skip_flag = (upperRight && i == Kx - 1);
        bool last_k = (i >= Kx - 1);
        bool l0_c_init_flag = (i == 0);
        LocalTensor<TYPE>* l0_b_buf_tensor = this->ping_pong_flag_l0_b_ ?
            &this->l0_b_pong_tensor : &this->l0_b_ping_tensor;
        WAIT_FLAG(M, MTE1, this->ping_pong_flag_l0_b_ + 2);
        if (!is_skip) {
            commonLoadData2dParamsTranspose.repeatTimes = n0_ / SIZE_16;
            commonLoadData2dParamsTranspose.srcStride = l1_k / SIZE_16;
            for (int nn = 0; nn < k0_ / SIZE_16; nn++) {
                AscendC::LoadData(
                    (*l0_b_buf_tensor)[nn * n0_ * SIZE_16],
                    l1_b_buf_tensor[nn * SIZE_256],
                    commonLoadData2dParamsTranspose
                );
            }
        }
        SET_FLAG(MTE1, M, this->ping_pong_flag_l0_b_ + 2);
        WAIT_FLAG(MTE1, M, this->ping_pong_flag_l0_b_ + 2);

        for (int32_t j = Ky_start; j < Ky_end; j++) {
            bool l0_skip_flag = (l1_skip_flag && j == 0);
            bool last_k = (i == Kx - 1 && j != 0) || (i == Kx - 1 && j == 0 && !upperRight) ||
                            (upperRight && i == Kx - 2 && j == 0);
            LocalTensor<TYPE>* l0_a_buf_tensor = this->ping_pong_flag_l0_a_ ?
                &this->l0_a_pong_tensor : &this->l0_a_ping_tensor;
            LocalTensor<float> l0_c_buf_tensor = (j == 0) ? this->l0_c_pong_tensor : this->l0_c_pong_tensor[128 * 64];

            WAIT_FLAG(M, MTE1, this->ping_pong_flag_l0_a_);
            if (!l0_skip_flag) {
                commonLoadData2dParamsNoTranspose.repeatTimes = k0_ / SIZE_16;
                commonLoadData2dParamsNoTranspose.srcStride = l1_m / SIZE_16;
                for (int32_t jj = 0; jj < m0_ / SIZE_16; jj++) {
                    AscendC::LoadData(
                        (*l0_a_buf_tensor)[jj * k0_ * SIZE_16],
                        (*l1_a_buf_tensor)[jj * SIZE_256],
                        commonLoadData2dParamsNoTranspose
                    );
                }
            }
            SET_FLAG(MTE1, M, this->ping_pong_flag_l0_a_);
            WAIT_FLAG(MTE1, M, this->ping_pong_flag_l0_a_);

            if (!l0_skip_flag) {
                AscendC::Mmad(
                    l0_c_buf_tensor,
                    *l0_a_buf_tensor,
                    *l0_b_buf_tensor,
                    AscendC::MmadParams(
                        m0_,
                        n0_,
                        k0_,
                        last_k ? 3 : 2,
                        false,
                        l0_c_init_flag
                    )
                );
            }

            if (last_k) {
                AscendC::SetAtomicType<float>();
                auto intriParams = AscendC::FixpipeParamsV220(
                    n0_,
                    m0_,
                    m0_,
                    n_,
                    false
                );
                intriParams.quantPre = QuantMode_t::NoQuant;
                intriParams.unitFlag = 3;
                this->temp_tensor_fp32.SetGlobalBuffer(
                    reinterpret_cast<__gm__ float *>(result_gm + j * SIZE_128 * n_ + n_offset));
                AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
                    this->temp_tensor_fp32,
                    l0_c_buf_tensor,
                    intriParams
                );
                AscendC::SetAtomicNone();
            }

            SET_FLAG(M, MTE1, this->ping_pong_flag_l0_a_);
            this->ping_pong_flag_l0_a_ = 1 - this->ping_pong_flag_l0_a_;
        }
        SET_FLAG(M, MTE1, this->ping_pong_flag_l0_b_ + 2);
        this->ping_pong_flag_l0_b_ = 1 - this->ping_pong_flag_l0_b_;
    }
}


template <typename TYPE, bool IF_BF16>
__aicore__ __inline__ void CubeBackwardBandOp192<TYPE, IF_BF16>::cube3_base_matmul(
    LocalTensor<TYPE> *l1_a_buf_tensor, LocalTensor<TYPE> *l1_b_buf_tensor,
    __gm__ float *result_gm, int64_t Kx_start, int64_t Kx_end, int64_t Ky_start, int64_t Ky_end, int64_t Ky,
    int64_t Kx, uint64_t n_offset, uint64_t n_step, uint64_t n_size, bool is_skip,
    bool is_skip_up)
{
    auto base_len = 128;
    auto k_stride = 64;
    int64_t out_matrix_cols = src_stride_K;

    auto n = 192;
    auto n1 = 64;
    auto n0 = 64;
    for (int64_t k_idx = Kx_start; k_idx < Kx_end; k_idx++) {
        for (int64_t m_idx = Ky_start; m_idx < Ky_end; m_idx ++) {
            LocalTensor<TYPE>* l0a_buf_tensor = this->ping_pong_flag_l0_a_ ?
                    &this->l0_a_pong_tensor : &this->l0_a_ping_tensor;
            LocalTensor<TYPE>* l0b_buf_tensor = this->ping_pong_flag_l0_b_ ?
                    &this->l0_b_pong_tensor : &this->l0_b_ping_tensor;

            WAIT_FLAG(M, MTE1, this->ping_pong_flag_l0_b_ + 2);
            if (!is_skip) {
                commonLoadData2dParamsTranspose.repeatTimes = n1 / BLOCK_SIZE;
                commonLoadData2dParamsTranspose.srcStride = base_len / BLOCK_SIZE;
                for (int32_t i = 0; i < base_len / BLOCK_SIZE; i++) {
                    AscendC::LoadData(
                        (*l0b_buf_tensor)[i * 64 * BLOCK_SIZE],
                        (*l1_b_buf_tensor)[m_idx * 128 * 64 + i * CUBE_MATRIX_SIZE],
                        commonLoadData2dParamsTranspose
                    );
                }
            }

            WAIT_FLAG(M, MTE1, this->ping_pong_flag_l0_a_);
            if (!is_skip) {
                commonLoadData2dParamsTranspose.repeatTimes = 128 / BLOCK_SIZE;
                commonLoadData2dParamsTranspose.srcStride = 1;
                for (int32_t i = 0; i < 128 / BLOCK_SIZE; i++) {
                    AscendC::LoadData(
                        (*l0a_buf_tensor)[i * 128 * BLOCK_SIZE],
                        (*l1_a_buf_tensor)[i * 128 * BLOCK_SIZE],
                        commonLoadData2dParamsTranspose
                    );
                }
            }

            SET_FLAG(MTE1, M, this->ping_pong_flag_l0_a_);
            WAIT_FLAG(MTE1, M, this->ping_pong_flag_l0_a_);

            bool init_c = (m_idx == 0);
            if (is_skip_up && m_idx == 1) {
                init_c = true;
            }
            bool out_c = (m_idx == (Ky - 1));
            int unit_flag = 0b10;
            if (out_c) {
                unit_flag = 0b11;
            }

            auto l0c_buf_tensor = this->ping_pong_flag_l0_c_ ?
                this->l0_c_ping_tensor : this->l0_c_ping_tensor[128 * 64];

            if (!is_skip) {
                AscendC::Mmad(
                    l0c_buf_tensor,
                    *l0a_buf_tensor,
                    *l0b_buf_tensor,
                    AscendC::MmadParams(
                        128,
                        n0,
                        128,
                        unit_flag,
                        false,
                        (init_c == 1) ? true : false
                    )
                );
            }
            SET_FLAG(M, MTE1, this->ping_pong_flag_l0_b_ + 2);
            SET_FLAG(M, MTE1, this->ping_pong_flag_l0_a_);
            if (out_c) {
                AscendC::SetAtomicAdd<float>();
                AscendC::SetAtomicType<float>();
                int32_t local_offset_gm_c = k_idx * base_len * out_matrix_cols + 128;
                auto intriParams = AscendC::FixpipeParamsV220(
                    n0,
                    128,
                    128,
                    out_matrix_cols,
                    false
                );
                intriParams.quantPre = QuantMode_t::NoQuant;
                intriParams.unitFlag = unit_flag;
                this->temp_tensor_fp32.SetGlobalBuffer(
                    reinterpret_cast<__gm__ float *>(result_gm + local_offset_gm_c));
                AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
                    this->temp_tensor_fp32,
                    l0c_buf_tensor,
                    intriParams
                );
                AscendC::SetAtomicNone();
                this->ping_pong_flag_l0_c_ = 1 - this->ping_pong_flag_l0_c_;
            }

            this->ping_pong_flag_l0_b_ = 1 - this->ping_pong_flag_l0_b_;
            this->ping_pong_flag_l0_a_ = 1 - this->ping_pong_flag_l0_a_;
        }
    }
}


template <typename TYPE, bool IF_BF16>
template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
__aicore__ inline __attribute__((always_inline))
void CubeBackwardBandOp192<TYPE, IF_BF16>::cube2_cube3_mix_op_192_right(
    const Address::PhyAddrBackwardCube2<T_LEFT, T_RIGHT, T_OUTPUT> *src_cube2, int64_t src_length,
    const Address::PhyAddrBackwardCube3<T_LEFT, T_RIGHT, T_OUTPUT> *src_cube3, int64_t vcore_num_per_head) {
    
    int64_t l1_m = CUBE2_LENGTH_M;               // 128
    int64_t l1_k = CUBE2_LENGTH_K;               // 128
    int64_t l1_n = 64;                           // 64
    int64_t l0_k = BASE_BLOCK_LENGTH;            // 128
    int64_t l0_m = BASE_BLOCK_LENGTH;            // 128
    int64_t l0_n = 64;                           // 64
    int64_t l0_k_block_num = l0_k / BLOCK_SIZE;  // 128/16
    int64_t l0_n_block_num = l0_n / BLOCK_SIZE;  // 128/16

    if (src_length == 0) {
        return;
    }

    int64_t Flag_l1b_PingPong = 1;
    int64_t Flag_l1a_PingPong = 1;
    int64_t Flag_L0a_PingPong = 1;
    int64_t Flag_L0b_PingPong = 1;
    int64_t Flag_L0c_PingPong = 1;
    int64_t mte1_mad_ping_flag = 1;
    int64_t Flag_l1b_PingPong_cube2 = 1;

    for (int64_t section_idx = 0; section_idx < src_length; section_idx ++) {
        auto Kx = src_cube2[section_idx].kx;
        auto Ky = src_cube2[section_idx].ky;
        auto lineStride = src_cube2[section_idx].lineStride;
        
        auto gm_left_cube2 = src_cube2[section_idx].left;    // dP  4
        auto gm_right_cube2 = src_cube2[section_idx].right;  // K   1
        auto gm_out_cube2 = src_cube2[section_idx].out;      // dQ  6

        auto gm_left_cube3 = src_cube3[section_idx].left;    // dP  4
        auto gm_right_cube3 = src_cube3[section_idx].right;  // Q   0
        auto gm_out_cube3 = src_cube3[section_idx].out;      // dK  7

        auto lowerLeft = src_cube3[section_idx].lowerLeft;
        auto upperRight = src_cube3[section_idx].upperRight;
        LocalTensor<TYPE>* l1_b_buf_cube3_tensor = this->ping_pong_flag_l1_b_ ?
                &this->l1_b_pong_tensor : &this->l1_b_ping_tensor;

        WAIT_FLAG(MTE1, MTE2, this->ping_pong_flag_l1_b_);
        commonNd2NzParams.nValue = 128;
        commonNd2NzParams.dValue = 64;
        commonNd2NzParams.srcDValue = 192;
        commonNd2NzParams.dstNzC0Stride = 128;
        for (auto idx = 0; idx < Ky; idx++) {
            this->temp_tensor_bf16.SetGlobalBuffer(
                reinterpret_cast<__gm__ TYPE *>(gm_right_cube3 + idx * 128 * 192 + 128));
            AscendC::DataCopy(
                (*l1_b_buf_cube3_tensor)[idx * 128 * 64],
                this->temp_tensor_bf16,
                commonNd2NzParams
            );
        }
        SET_FLAG(MTE2, MTE1, this->ping_pong_flag_l1_b_);
        WAIT_FLAG(MTE2, MTE1, this->ping_pong_flag_l1_b_);

        for (int64_t k_idx = 0 ; k_idx < Kx; k_idx ++) {
            bool is_skip_up = false;
            for (int64_t m_idx = 0; m_idx < Ky; m_idx ++) {
                LocalTensor<TYPE>* l1_buf_a_cube3_tensor =
                    Flag_l1a_PingPong ?
                    (this->ping_pong_flag_l1_a_ ? &this->l1_a_pong_tensor : &this->l1_a_ping_tensor)
                    : (this->ping_pong_flag_l1_a_ ? &this->l1_a_pong_double_tensor : &this->l1_a_ping_double_tensor);
                auto event_id_cube3 = Flag_l1a_PingPong ? EVENT_ID4 : EVENT_ID5;
                bool is_skip = false;
                if (k_idx == 0 && m_idx == Ky - 1 && lowerLeft) {
                    is_skip = true;
                }
                if (k_idx == Kx - 1 && m_idx == 0 && upperRight) {
                    is_skip = true;
                }

                WAIT_FLAG(MTE1, MTE2, event_id_cube3);
                if (!is_skip) {
                    this->temp_tensor_bf16.SetGlobalBuffer(
                        reinterpret_cast<__gm__ TYPE *>(gm_left_cube3 + lineStride * m_idx + k_idx * 128 * 128));
                    AscendC::DataCopy(
                        *l1_buf_a_cube3_tensor,
                        this->temp_tensor_bf16,
                        AscendC::Nd2NzParams(
                            vcore_num_per_head,
                            128 / vcore_num_per_head,
                            128,
                            128 / vcore_num_per_head * 128 * 2,
                            128,
                            128,
                            1,
                            128 * BLOCK_SIZE / vcore_num_per_head
                        )
                    );
                }
                SET_FLAG(MTE2, MTE1, event_id_cube3);
                WAIT_FLAG(MTE2, MTE1, event_id_cube3);

                cube3_base_matmul(l1_buf_a_cube3_tensor, l1_b_buf_cube3_tensor, gm_out_cube3,
                    k_idx, k_idx+1, m_idx, m_idx +1, Ky, Kx, 0, 0, 0, is_skip, is_skip_up);
                SET_FLAG(MTE1, MTE2, event_id_cube3);

                auto event_id_cube2 = Flag_l1b_PingPong ?  EVENT_ID3 : EVENT_ID4;
                auto l1_b_buf_cube2_tensor = Flag_l1b_PingPong ?
                        (*l1_b_buf_cube3_tensor)[128 * 64 * 2] : (*l1_b_buf_cube3_tensor)[128 * 64 * 3];

                WAIT_FLAG(MTE1, MTE2, event_id_cube2);
                if (!is_skip) {
                    commonNd2NzParams.nValue = 128;
                    commonNd2NzParams.dValue = 64;
                    commonNd2NzParams.srcDValue = src_stride_K;
                    commonNd2NzParams.dstNzC0Stride = 128;
                    this->temp_tensor_bf16.SetGlobalBuffer(
                        reinterpret_cast<__gm__ TYPE *>(gm_right_cube2 + k_idx * 128 * src_stride_K + 128));
                    AscendC::DataCopy(
                        l1_b_buf_cube2_tensor,
                        this->temp_tensor_bf16,
                        commonNd2NzParams
                    );
                }

                SET_FLAG(MTE2, MTE1, event_id_cube2);
                WAIT_FLAG(MTE2, MTE1, event_id_cube2);

                cube2_base_matmul(l1_buf_a_cube3_tensor, l1_b_buf_cube2_tensor, gm_out_cube2, k_idx, k_idx + 1,
                    m_idx, m_idx+1, Ky, Kx, 0, 0, 0, upperRight, is_skip);

                SET_FLAG(MTE1, MTE2, event_id_cube2);

                Flag_l1a_PingPong = 1 - Flag_l1a_PingPong;
                Flag_l1b_PingPong = 1 - Flag_l1b_PingPong;
                is_skip_up = is_skip;
            }

            this->ping_pong_flag_l1_a_ = 1 - this->ping_pong_flag_l1_a_;
        }
        SET_FLAG(MTE1, MTE2, this->ping_pong_flag_l1_b_);
        this->ping_pong_flag_l1_b_ = 1 - this->ping_pong_flag_l1_b_;
    }
}

template <typename TYPE, bool IF_BF16>
template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
__aicore__ __inline__ void CubeBackwardBandOp192<TYPE, IF_BF16>::cube2_matmul_op_192_left(
    const Address::PhyAddrBackwardCube2<T_LEFT, T_RIGHT, T_OUTPUT> *src, int64_t src_len, int64_t vcore_num_per_head)
{
    this->cube2_matmul_op(src, src_len, vcore_num_per_head, 0, 128, 192, 256);
}

template <typename TYPE, bool IF_BF16>
template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
__aicore__ __inline__ void CubeBackwardBandOp192<TYPE, IF_BF16>::cube3_matmul_op_192_left(
    const Address::PhyAddrBackwardCube3<T_LEFT, T_RIGHT, T_OUTPUT> *src, int64_t src_len, int64_t vcore_num_per_head)
{
    this->cube3_matmul_op(src, src_len, vcore_num_per_head, 128, 192, 256);
}

template <typename TYPE, bool IF_BF16>
__aicore__ inline void CubeBackwardBandOp192<TYPE, IF_BF16>::Run_op()
{
    AscendC::SetLoadDataPaddingValue<uint64_t>(0);
    uint64_t config = 0x1;
    AscendC::SetNdParaImpl(config);

    this->preset_flag();
    int64_t Z = this->address.get_total_rounds();

    if (Z == 1) {
        for (int64_t roundId = 0; roundId < Z; roundId++) {
            if (this->address.is_running(roundId)) {
                // cube1
                int64_t len_S = 0;
                Address::PhyAddrBackwardCube1<TYPE, TYPE, float> addr_S[MAX_SWITCH_TIME];
                this->address.addrMapping_cube1(this->gm_Q, this->gm_K, this->gm_S, addr_S,
                    len_S, roundId, 192, src_stride_K);
                cube1_matmul_op_192(addr_S, len_S);

                int64_t len_dP = 0;
                Address::PhyAddrBackwardCube1<TYPE, TYPE, float> addr_dP[MAX_SWITCH_TIME];
                this->address.addrMapping_cube1(this->gm_dO, this->gm_V, this->gm_dP,
                    addr_dP, len_dP, roundId, 128, 128);
                this->cube1_matmul_op(addr_dP, len_dP);
            }

            if (this->is_syn) {
                FftsCrossCoreSync<PIPE_FIX, 2>(AIC2AIVFLAGID);
                WaitFlagDev(AIV2AICFLAGID);
            }

            if (this->address.is_running(roundId) && this->is_cube2) {
                // cube3
                int64_t len_dV = 0;
                Address::PhyAddrBackwardCube3<float, TYPE, float> addr_dV[MAX_SWITCH_TIME];
                this->address.addrMapping_cube3(this->gm_S, this->gm_dO, this->gm_dV, addr_dV,
                    len_dV, roundId, 128, 128);
                this->cube3_matmul_op(addr_dV, len_dV, 2, 128, 128, 128);

                int64_t len_dK = 0;
                Address::PhyAddrBackwardCube3<float, TYPE, float> addr_dK[MAX_SWITCH_TIME];
                this->address.addrMapping_cube3(this->gm_dP, this->gm_Q, this->gm_dK, addr_dK,
                    len_dK, roundId, 192, 256);
                cube3_matmul_op_192_left(addr_dK, len_dK, 2);

                // cube2
                int64_t len_dQ = 0;
                Address::PhyAddrBackwardCube2<float, TYPE, float> addr_dQ[MAX_SWITCH_TIME];
                this->address.addrMapping_cube2(this->gm_dP, this->gm_K, this->gm_dQ, addr_dQ,
                    len_dQ, roundId, 192, 256);
                cube2_matmul_op_192_left(addr_dQ, len_dQ, 2);
                cube2_cube3_mix_op_192_right(addr_dQ, len_dQ, addr_dK, 2);
            }
        }
    } else {
        for (int64_t roundId = 0; roundId < 2; roundId++) {
            if (this->address.is_running(roundId)) {
                // cube1
                int64_t len_S = 0;
                Address::PhyAddrBackwardCube1<TYPE, TYPE, float> addr_S[MAX_SWITCH_TIME];
                this->address.addrMapping_cube1(this->gm_Q, this->gm_K, this->gm_S, addr_S, len_S, roundId, 192, 256);
                cube1_matmul_op_192(addr_S, len_S);

                int64_t len_dP = 0;
                Address::PhyAddrBackwardCube1<TYPE, TYPE, float> addr_dP[MAX_SWITCH_TIME];
                this->address.addrMapping_cube1(this->gm_dO, this->gm_V, this->gm_dP, addr_dP, len_dP,
                    roundId, 128, 128);
                this->cube1_matmul_op(addr_dP, len_dP);
            }

            if (this->is_syn) {
                if (roundId == 0) {
                    FftsCrossCoreSync<PIPE_FIX, 2>(AIC2AIVFLAGID);
                }
            }
        }
        // cube2 + cube3 + cube1
        for (int64_t roundId = 2; roundId < Z; roundId++) {
            if (this->is_syn) {
                WaitFlagDev(AIV2AICFLAGID);
                FftsCrossCoreSync<PIPE_FIX, 2>(AIC2AIVFLAGID);
            }

            if (this->address.is_running(roundId - 2)) {
                int64_t len_dV = 0;
                Address::PhyAddrBackwardCube3<float, TYPE, float> addr_dV[MAX_SWITCH_TIME];
                this->address.addrMapping_cube3(this->gm_S, this->gm_dO, this->gm_dV, addr_dV,
                    len_dV, roundId - 2, 128, 128);
                this->cube3_matmul_op(addr_dV, len_dV, 2, 128, 128, 128);

                int64_t len_dK = 0;
                Address::PhyAddrBackwardCube3<float, TYPE, float> addr_dK[MAX_SWITCH_TIME];
                this->address.addrMapping_cube3(this->gm_dP, this->gm_Q, this->gm_dK, addr_dK,
                    len_dK, roundId - 2, 192, 256);
                cube3_matmul_op_192_left(addr_dK, len_dK, 2);

                int64_t len_dQ = 0;
                Address::PhyAddrBackwardCube2<float, TYPE, float> addr_dQ[MAX_SWITCH_TIME];
                this->address.addrMapping_cube2(this->gm_dP, this->gm_K, this->gm_dQ, addr_dQ,
                    len_dQ, roundId - 2, 192, 256);
                cube2_matmul_op_192_left(addr_dQ, len_dQ, 2);

                cube2_cube3_mix_op_192_right(addr_dQ, len_dQ, addr_dK, 2);
            }

            if (this->address.is_running(roundId) && this->is_cube1) {
                // cube1
                int64_t len_S = 0;
                Address::PhyAddrBackwardCube1<TYPE, TYPE, float> addr_S[MAX_SWITCH_TIME];
                this->address.addrMapping_cube1(this->gm_Q, this->gm_K, this->gm_S, addr_S, len_S, roundId, 192, 256);
                cube1_matmul_op_192(addr_S, len_S);

                int64_t len_dP = 0;
                Address::PhyAddrBackwardCube1<TYPE, TYPE, float> addr_dP[MAX_SWITCH_TIME];
                this->address.addrMapping_cube1(this->gm_dO, this->gm_V, this->gm_dP, addr_dP,
                    len_dP, roundId, 128, 128);
                this->cube1_matmul_op(addr_dP, len_dP);
            }
        }

        // (cube2 + cube3) * 2
        for (int64_t roundId = 0; roundId < 2; roundId++) {
            if (this->is_syn) {
                WaitFlagDev(AIV2AICFLAGID);
                if (roundId == 0) {
                    FftsCrossCoreSync<PIPE_FIX, 2>(AIC2AIVFLAGID);
                }
            }
            if (this->address.is_running(roundId + Z - 2)) {
                // cube3
                int64_t len_dV = 0;
                Address::PhyAddrBackwardCube3<float, TYPE, float> addr_dV[MAX_SWITCH_TIME];
                this->address.addrMapping_cube3(this->gm_S, this->gm_dO, this-> gm_dV, addr_dV,
                    len_dV, roundId + Z - 2, 128, 128);
                this->cube3_matmul_op(addr_dV, len_dV, 2, 128, 128, 128);

                int64_t len_dK = 0;
                Address::PhyAddrBackwardCube3<float, TYPE, float> addr_dK[MAX_SWITCH_TIME];
                this->address.addrMapping_cube3(this->gm_dP, this->gm_Q, this->gm_dK, addr_dK,
                    len_dK, roundId + Z - 2, 192, 256);
                cube3_matmul_op_192_left(addr_dK, len_dK, 2);
            
                // cube2
                int64_t len_dQ = 0;
                Address::PhyAddrBackwardCube2<float, TYPE, float> addr_dQ[MAX_SWITCH_TIME];
                this->address.addrMapping_cube2(this->gm_dP, this->gm_K, this->gm_dQ, addr_dQ,
                    len_dQ, roundId + Z - 2, 192, 256);
                cube2_matmul_op_192_left(addr_dQ, len_dQ, 2);
                cube2_cube3_mix_op_192_right(addr_dQ, len_dQ, addr_dK, 2);
            }
        }
    }
    FftsCrossCoreSync<PIPE_FIX, 0>(AICFLAGID);
    WaitFlagDev(AICFLAGID);
    this->clear_flag();
}

}  // namespace CUBE_BACK_BAND_OP
#endif  // __DAV_C220_CUBE__

#endif  // __CUBEFORWARD_H__