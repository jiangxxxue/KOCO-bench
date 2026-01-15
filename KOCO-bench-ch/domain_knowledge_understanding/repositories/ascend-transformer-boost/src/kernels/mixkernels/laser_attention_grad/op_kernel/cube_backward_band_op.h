/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef __CUBEBACKWARD_BAND_OP_H__
#define __CUBEBACKWARD_BAND_OP_H__

#define SET_FLAG(trigger, waiter, e) AscendC::SetFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define WAIT_FLAG(trigger, waiter, e) AscendC::WaitFlag<AscendC::HardEvent::trigger##_##waiter>((e))

#include "address_mapping_cube.h"
#include "ppmatmul_const_grad.h"
#include "kernels/utils/kernel/utils.h"
#include "kernels/utils/kernel/iterator.h"

using namespace AscendC;

#ifdef __DAV_C220_CUBE__


namespace CUBE_BACK_BAND_OP {

constexpr int32_t SIZE_16 = 16;
constexpr int32_t SIZE_32 = 32;
constexpr int32_t SIZE_64 = 64;
constexpr int32_t SIZE_128 = 128;
constexpr int32_t SIZE_256 = 256;
constexpr int32_t SIZE_384 = 384;
constexpr int32_t SIZE_ONE_K = 1024;
constexpr int32_t SIZE_LONG_BLOCK = 16384;

using WORKSPACE_DTYPE = float;
using WORKSPACE_DTYPE_DP = float;

template <typename TYPE, bool IF_BF16>
class CubeBackwardBandOp {
public:
    __aicore__ inline CubeBackwardBandOp() {}

    __aicore__ inline void Init(__gm__ TYPE *__restrict__ gm_Q,
        __gm__ TYPE *__restrict__ gm_K,
        __gm__ TYPE *__restrict__ gm_V,
        __gm__ TYPE *__restrict__ gm_dO,
        __gm__ float *__restrict__ gm_dP,
        __gm__ float *__restrict__ gm_S,
        __gm__ float *__restrict__ gm_dQ,
        __gm__ float *__restrict__ gm_dK,
        __gm__ float *__restrict__ gm_dV,
        bool isTri, int64_t qSize, int64_t kSize, int64_t batch, int64_t head, int64_t group_size,
        int64_t base_length, int64_t sparseMode, int64_t windowLength, int64_t blockNumPerCore);
    
    __aicore__ inline void Run();
    
    __aicore__ inline void Run_mix();
    
    __aicore__ __inline__ void Run_op();

protected:
    __aicore__ __inline__ void clear_flag();

    __aicore__ __inline__ void preset_flag();

    template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
    __aicore__ inline __attribute__((always_inline)) void cube3_matmul_op(
        const Address::PhyAddrBackwardCube3<T_LEFT, T_RIGHT, T_OUTPUT> *src, int64_t src_len,
        int64_t vcore_num_per_head, int64_t copy_matrix_stride, int64_t right_matrix_stride, int64_t out_matrix_stride);

    template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
    __aicore__ inline __attribute__((always_inline)) void cube2_matmul_op(
        const Address::PhyAddrBackwardCube2<T_LEFT, T_RIGHT, T_OUTPUT> *src, int64_t src_len,
        int64_t vcore_num_per_head, uint64_t n_offset, uint64_t n_step, uint64_t n_size, uint64_t n_size_k);

    template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
    __aicore__ inline __attribute__((always_inline)) void cube1_matmul_op(
        const Address::PhyAddrBackwardCube1<T_LEFT, T_RIGHT, T_OUTPUT> *src, int64_t src_len);

    __aicore__ __inline__ void cube1_base_matmul(LocalTensor<TYPE> *l1_a_tensor, LocalTensor<TYPE>* l1_b_tensor,
        __gm__ float *gm_out, int32_t ky, int32_t out_put_matrix_line_strid, bool upper_right_flag);

protected:
    AsdopsBuffer<ArchType::ASCEND_V220> asdopsBuf;

    __gm__ TYPE *__restrict__ gm_Q{nullptr};
    __gm__ TYPE *__restrict__ gm_K{nullptr};
    __gm__ TYPE *__restrict__ gm_V{nullptr};
    __gm__ TYPE *__restrict__ gm_dO{nullptr};

    __gm__ float *__restrict__ gm_dP{nullptr};
    __gm__ float *__restrict__ gm_S{nullptr};
    __gm__ float *__restrict__ gm_dQ{nullptr};
    __gm__ float *__restrict__ gm_dK{nullptr};
    __gm__ float *__restrict__ gm_dV{nullptr};

    __gm__ uint8_t *__restrict__ ffts_addr{nullptr};
    __gm__ uint8_t *__restrict__ tiling_para_gm{nullptr};

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

    AscendC::Nd2NzParams commonNd2NzParamsFp32 {
        2,
        64,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH * BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        1,
        BASE_BLOCK_LENGTH * BLOCK_SIZE / 2
    };

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

    AscendC::MmadParams commonMadParams {
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        3,
        false,
        true
    };

    AscendC::FixpipeParamsV220 commonFixpipeParamsV220 {
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        BASE_BLOCK_LENGTH,
        false
    };

    LocalTensor<TYPE> l1_base_a_cube1_tensor;
    LocalTensor<TYPE> l1_base_b_cube1_tensor;
    LocalTensor<TYPE> l1_base_a_cube2_tensor;
    LocalTensor<TYPE> l1_base_b_cube2_tensor;

    LocalTensor<TYPE> l1_a_ping_tensor;
    LocalTensor<TYPE> l1_a_pong_tensor;
    LocalTensor<TYPE> l1_b_ping_tensor;
    LocalTensor<TYPE> l1_b_pong_tensor;
    
    LocalTensor<TYPE> l1_a_ping_double_tensor;
    LocalTensor<TYPE> l1_a_pong_double_tensor;
    LocalTensor<TYPE> l1_b_ping_double_tensor;
    LocalTensor<TYPE> l1_b_pong_double_tensor;

    GlobalTensor<TYPE> temp_tensor_bf16;
    GlobalTensor<float> temp_tensor_fp32;

    uint32_t ping_pong_flag_l1_a_ = 0;
    uint32_t ping_pong_flag_l1_b_ = 0;

    // L0A L0B
    LocalTensor<TYPE> l0_a_ping_tensor;
    LocalTensor<TYPE> l0_a_pong_tensor;
    LocalTensor<TYPE> l0_b_ping_tensor;
    LocalTensor<TYPE> l0_b_pong_tensor;

    // L0C
    LocalTensor<float> l0_c_buf_tensor;
    LocalTensor<float> l0_c_ping_tensor;
    LocalTensor<float> l0_c_pong_tensor;

    uint32_t ping_pong_flag_l0_a_ = 0;
    uint32_t ping_pong_flag_l0_b_ = 0;
    uint32_t ping_pong_flag_l0_c_ = 0;

    int64_t core_num;
    int64_t cur_core_index;
    int64_t blockNumPerCore;
    int64_t batchSize;
    int64_t headNum;
    int64_t gqa_group_num;
    int64_t headDim;
    int64_t seqLenQ;
    int64_t seqLenK;
    int64_t blocks_per_column;
    int64_t blocks_per_row;
    bool isTri;
    int64_t sparseMode;
    int64_t windowLength;

    bool is_cube1;
    bool is_cube2;
    bool is_cube3;
    bool is_syn;

    Address::AddressMappingCube<TYPE> address;
};

template <typename TYPE, bool IF_BF16>
__aicore__ __inline__ void CubeBackwardBandOp<TYPE, IF_BF16>::preset_flag()
{
    SET_FLAG(MTE1, MTE2, EVENT_ID0);
    SET_FLAG(MTE1, MTE2, EVENT_ID1);
    SET_FLAG(MTE1, MTE2, EVENT_ID2);
    SET_FLAG(MTE1, MTE2, EVENT_ID3);
    SET_FLAG(MTE1, MTE2, EVENT_ID4);
    SET_FLAG(MTE1, MTE2, EVENT_ID5);
    SET_FLAG(M, MTE1, EVENT_ID0);
    SET_FLAG(M, MTE1, EVENT_ID1);
    SET_FLAG(M, MTE1, EVENT_ID2);
    SET_FLAG(M, MTE1, EVENT_ID3);
    SET_FLAG(M, MTE1, EVENT_ID4);
    SET_FLAG(M, MTE1, EVENT_ID5);
}

template <typename TYPE, bool IF_BF16>
__aicore__ __inline__ void CubeBackwardBandOp<TYPE, IF_BF16>::clear_flag()
{
    WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID4);
    WAIT_FLAG(MTE1, MTE2, EVENT_ID5);

    WAIT_FLAG(M, MTE1, EVENT_ID0);
    WAIT_FLAG(M, MTE1, EVENT_ID1);
    WAIT_FLAG(M, MTE1, EVENT_ID2);
    WAIT_FLAG(M, MTE1, EVENT_ID3);
    WAIT_FLAG(M, MTE1, EVENT_ID4);
    WAIT_FLAG(M, MTE1, EVENT_ID5);
}

template <typename TYPE, bool IF_BF16>
__aicore__ __inline__ void CubeBackwardBandOp<TYPE, IF_BF16>::cube1_base_matmul(
    LocalTensor<TYPE> *l1_a_tensor, LocalTensor<TYPE> *l1_b_tensor,
    __gm__ float *gm_out, int32_t ky, int32_t out_put_matrix_line_strid, bool upper_right_flag)
{
    auto l1_n_size_ = 128;
    auto l1_m_size_ = ky * 128;
    auto k0_ = 128;
    auto n0_ = 128;
    auto m0_ = 128;
    auto n_ = 128;

    for (int n_offset = 0; n_offset < l1_n_size_; n_offset += SIZE_128) {
        LocalTensor<TYPE>* l0_b_tensor = ping_pong_flag_l0_b_ ? &l0_b_pong_tensor : &l0_b_ping_tensor;

        WAIT_FLAG(M, MTE1, ping_pong_flag_l0_b_ + 2);
        if (l1_n_size_ == SIZE_128) {
            commonLoadData2dParamsNoTranspose.repeatTimes = k0_ * n0_ / SIZE_256;
            commonLoadData2dParamsNoTranspose.srcStride = 1;
            AscendC::LoadData(
                *l0_b_tensor,
                *l1_b_tensor,
                commonLoadData2dParamsNoTranspose
            );
        } else {
            commonLoadData2dParamsNoTranspose.repeatTimes = n0_ / SIZE_16;
            commonLoadData2dParamsNoTranspose.srcStride = 1;
            for (int i = 0; i < k0_ / SIZE_16; i++) {
                AscendC::LoadData(
                    (*l0_b_tensor)[i * n0_ * SIZE_16],
                    (*l1_b_tensor)[i * l1_n_size_ * SIZE_16 + n_offset * SIZE_16],
                    commonLoadData2dParamsNoTranspose
                );
            }
        }

        SET_FLAG(MTE1, M, ping_pong_flag_l0_b_ + 2);
        WAIT_FLAG(MTE1, M, ping_pong_flag_l0_b_ + 2);

        for (int m_offset = 0; m_offset < l1_m_size_; m_offset += SIZE_128) {
            bool l0_skip_flag = (upper_right_flag && m_offset == 0);
            LocalTensor<TYPE>* l0_a_tensor = ping_pong_flag_l0_a_ ? &l0_a_pong_tensor : &l0_a_ping_tensor;
            LocalTensor<float>* l0_c_tensor = ping_pong_flag_l0_c_ ? &l0_c_pong_tensor : &l0_c_ping_tensor;

            WAIT_FLAG(M, MTE1, ping_pong_flag_l0_a_);
            if (!l0_skip_flag) {
                commonLoadData2dParamsNoTranspose.repeatTimes = k0_ / SIZE_16;
                commonLoadData2dParamsNoTranspose.srcStride = l1_m_size_ / SIZE_16;
                for (int32_t i = 0; i < m0_ / SIZE_16; i++) {
                    AscendC::LoadData(
                        (*l0_a_tensor)[i * k0_ * SIZE_16],
                        (*l1_a_tensor)[m_offset * SIZE_16 + i * SIZE_256],
                        commonLoadData2dParamsNoTranspose
                    );
                }
            }
            SET_FLAG(MTE1, M, ping_pong_flag_l0_a_);

            WAIT_FLAG(MTE1, M, ping_pong_flag_l0_a_);
            if (!l0_skip_flag) {
                commonMadParams.unitFlag = 3;
                commonMadParams.cmatrixInitVal = true;
                AscendC::Mmad(
                    *l0_c_tensor,
                    *l0_a_tensor,
                    *l0_b_tensor,
                    commonMadParams
                );
            }
            SET_FLAG(M, MTE1, ping_pong_flag_l0_a_);
            auto out_offset = (m_offset / 128) * out_put_matrix_line_strid + n_offset;

            if (!l0_skip_flag) {
                commonFixpipeParamsV220.dstStride = n_;
                temp_tensor_fp32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_out + out_offset));
                AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
                    temp_tensor_fp32,
                    *l0_c_tensor,
                    commonFixpipeParamsV220
                );
            }

            ping_pong_flag_l0_c_ = 1 - ping_pong_flag_l0_c_;
            ping_pong_flag_l0_a_ = 1 - ping_pong_flag_l0_a_;
        }

        SET_FLAG(M, MTE1, ping_pong_flag_l0_b_ + 2);
        ping_pong_flag_l0_b_ = 1 - ping_pong_flag_l0_b_;
    }
}

template <typename TYPE, bool IF_BF16>
template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
__aicore__ inline __attribute__((always_inline)) void CubeBackwardBandOp<TYPE, IF_BF16>::cube3_matmul_op(
    const Address::PhyAddrBackwardCube3<T_LEFT, T_RIGHT, T_OUTPUT> *src, int64_t src_len, int64_t vcore_num_per_head,
    int64_t copy_matrix_stride, int64_t right_matrix_stride, int64_t out_matrix_stride)
{
    LocalTensor<TYPE>* l1_base_b_cube3_tensor = ping_pong_flag_l1_b_ ? &l1_b_pong_tensor : &l1_b_ping_tensor;

    auto lineStride = src[0].lineStride;
    auto ky = src[0].ky;
    int64_t base_len = 128;
    int64_t switch_index_list[MAX_SWITCH_TIME] = {0};
    int64_t total_len = 0;
    for (int64_t src_index = 0; src_index < src_len; src_index++) {
        switch_index_list[src_index] = total_len;
        total_len += src[src_index].kx;
    }

    int64_t Flag_l1b_PingPong = 1;
    int64_t Flag_l1a_PingPong = 1;
    int64_t Flag_L0a_PingPong = 1;
    int64_t Flag_L0b_PingPong = 1;
    int64_t Flag_L0c_PingPong = 1;
    int64_t mte1_mad_ping_flag = 1;

    int32_t cur_switch_index = 0;
    int32_t cur_switch = switch_index_list[cur_switch_index];

    WAIT_FLAG(MTE1, MTE2, ping_pong_flag_l1_b_);

    for (auto idx = 0; idx < ky; idx++) {
        commonNd2NzParams.nValue = base_len;
        commonNd2NzParams.dValue = copy_matrix_stride;
        commonNd2NzParams.srcDValue = right_matrix_stride;
        commonNd2NzParams.dstNzC0Stride = base_len;
        temp_tensor_bf16.SetGlobalBuffer(
            reinterpret_cast<__gm__ TYPE *>(src[0].right + idx * 128 * right_matrix_stride));
        AscendC::DataCopy(
            (*l1_base_b_cube3_tensor)[idx * 128 * 128],
            temp_tensor_bf16,
            commonNd2NzParams
        );
    }

    SET_FLAG(MTE2, MTE1, ping_pong_flag_l1_b_);
    WAIT_FLAG(MTE2, MTE1, ping_pong_flag_l1_b_);

    WAIT_FLAG(MTE1, MTE2, ping_pong_flag_l1_a_ + 2);

    for (int32_t k_idx = 0; k_idx < total_len; k_idx++) {
        if (k_idx != 0 && (src_len >= 2 && k_idx == switch_index_list[cur_switch_index + 1])) {
            cur_switch = switch_index_list[cur_switch_index + 1];
            lineStride = src[cur_switch_index + 1].lineStride;
            ky = src[cur_switch_index + 1].ky;
            SET_FLAG(MTE1, MTE2, ping_pong_flag_l1_b_);
            WAIT_FLAG(MTE1, MTE2, ping_pong_flag_l1_b_);
            for (auto idx = 0; idx < ky; idx++) {
                commonNd2NzParams.nValue = base_len;
                commonNd2NzParams.dValue = copy_matrix_stride;
                commonNd2NzParams.srcDValue = right_matrix_stride;
                commonNd2NzParams.dstNzC0Stride = base_len;
                temp_tensor_bf16.SetGlobalBuffer(
                    reinterpret_cast<__gm__ TYPE *>(src[cur_switch_index + 1].right + idx * 128 * right_matrix_stride));
                AscendC::DataCopy(
                    (*l1_base_b_cube3_tensor)[idx * 128 * 128],
                    temp_tensor_bf16,
                    commonNd2NzParams
                );
            }
            SET_FLAG(MTE2, MTE1, ping_pong_flag_l1_b_);
            WAIT_FLAG(MTE2, MTE1, ping_pong_flag_l1_b_);
        }
        int32_t actual_k_idx = k_idx - cur_switch;
        int32_t iC = actual_k_idx;
        bool is_skip_up = false;
        for (int32_t m_idx = 0; m_idx < ky; m_idx++) {
            LocalTensor<TYPE>* l0a_buf_tensor = ping_pong_flag_l0_a_ ? &l0_a_pong_tensor : &l0_a_ping_tensor;
            LocalTensor<TYPE>* l1_buf_a_cube3_tensor = Flag_l1a_PingPong ?
                                (ping_pong_flag_l1_a_ ? &l1_a_pong_tensor : &l1_a_ping_tensor)
                                : (ping_pong_flag_l1_a_ ? &l1_a_pong_double_tensor : &l1_a_ping_double_tensor);

            auto event_id_cube3 = Flag_l1a_PingPong ? EVENT_ID4 : EVENT_ID5;

            LocalTensor<TYPE>* l0b_buf_tensor = ping_pong_flag_l0_b_ ? &l0_b_pong_tensor : &l0_b_ping_tensor;

            if (k_idx == 0 || ((src_len >= 2 && k_idx == switch_index_list[cur_switch_index + 1]))) {
                if (k_idx != 0) {
                    SET_FLAG(M, MTE1, ping_pong_flag_l0_b_ + 2);
                }
                WAIT_FLAG(M, MTE1, ping_pong_flag_l0_b_ + 2);
                commonLoadData2dParamsTranspose.repeatTimes = base_len / BLOCK_SIZE;
                commonLoadData2dParamsTranspose.srcStride = base_len / BLOCK_SIZE;
                for (int32_t i = 0; i < base_len / BLOCK_SIZE; i++) {
                    AscendC::LoadData(
                        (*l0b_buf_tensor)[i * base_len * BLOCK_SIZE],
                        (*l1_base_b_cube3_tensor)[m_idx * 128 * 128 + i * CUBE_MATRIX_SIZE],
                        commonLoadData2dParamsTranspose
                    );
                }
            }

            WAIT_FLAG(MTE1, MTE2, event_id_cube3);
            int32_t tmp_left_idx = cur_switch_index;
            if (src_len >= 2 && k_idx == switch_index_list[cur_switch_index + 1]) {
                tmp_left_idx = cur_switch_index + 1;
            }
            auto gm_left = src[tmp_left_idx].left + lineStride * m_idx;
            bool is_skip = false;
            if (iC == 0 && m_idx == ky - 1 && src[tmp_left_idx].lowerLeft) {
                is_skip = true;
            }
            if (iC == src[tmp_left_idx].kx - 1 && m_idx == 0 && src[tmp_left_idx].upperRight) {
                is_skip = true;
            }
            int32_t local_offset_a = actual_k_idx * base_len * base_len;

            if (!is_skip) {
                temp_tensor_bf16.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(gm_left + local_offset_a));
                AscendC::DataCopy(
                    *l1_buf_a_cube3_tensor,
                    temp_tensor_bf16,
                    commonNd2NzParamsFp32
                );
            }

            SET_FLAG(MTE2, MTE1, event_id_cube3);
            WAIT_FLAG(MTE2, MTE1, event_id_cube3);
            WAIT_FLAG(M, MTE1, ping_pong_flag_l0_a_);

            if (!is_skip) {
                commonLoadData2dParamsTranspose.repeatTimes = 128 / BLOCK_SIZE;
                commonLoadData2dParamsTranspose.srcStride = 1;
                for (int32_t i = 0; i < 128 / BLOCK_SIZE; i++) {
                    AscendC::LoadData(
                        (*l0a_buf_tensor)[i * 128 * BLOCK_SIZE],
                        (*l1_buf_a_cube3_tensor)[i * 128 * BLOCK_SIZE],
                        commonLoadData2dParamsTranspose
                    );
                }
            }
            SET_FLAG(MTE1, MTE2, event_id_cube3);

            SET_FLAG(MTE1, M, ping_pong_flag_l0_a_);
            WAIT_FLAG(MTE1, M, ping_pong_flag_l0_a_);

            bool init_c = (m_idx == 0);
            if (is_skip_up && m_idx == 1) {
                init_c = true;
            }

            bool out_c = (m_idx == (ky - 1));
            int unit_flag = 0b10;
            if (out_c) {
                unit_flag = 0b11;
            }
            LocalTensor<float> *l0c_buf_tensor = ping_pong_flag_l0_c_ ? &l0_c_pong_tensor : &l0_c_ping_tensor;

            if (!is_skip) {
                commonMadParams.unitFlag = unit_flag;
                commonMadParams.cmatrixInitVal = init_c;
                AscendC::Mmad(
                    *l0c_buf_tensor,
                    *l0a_buf_tensor,
                    *l0b_buf_tensor,
                    commonMadParams
                );
            }

            if (k_idx == total_len - 1) {
                SET_FLAG(M, MTE1, ping_pong_flag_l0_b_ + 2);
            }

            SET_FLAG(M, MTE1, ping_pong_flag_l0_a_);
            if (out_c) {
                AscendC::SetAtomicAdd<float>();
                AscendC::SetAtomicType<float>();
                int32_t local_offset_gm_c = actual_k_idx * base_len * out_matrix_stride;
                int32_t src_out_idx = cur_switch_index;
                if ((src_len >= 2 && k_idx == switch_index_list[cur_switch_index + 1])) {
                    src_out_idx += 1;
                }
                commonFixpipeParamsV220.dstStride = out_matrix_stride;
                temp_tensor_fp32.SetGlobalBuffer(
                    reinterpret_cast<__gm__ float *>(src[src_out_idx].out + local_offset_gm_c));
                AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
                    temp_tensor_fp32,
                    *l0c_buf_tensor,
                    commonFixpipeParamsV220
                );
                AscendC::SetAtomicNone();
                ping_pong_flag_l0_c_ = 1 - ping_pong_flag_l0_c_;
            }
            is_skip_up = is_skip;
            ping_pong_flag_l0_b_ = 1 - ping_pong_flag_l0_b_;
            ping_pong_flag_l0_a_ = 1 - ping_pong_flag_l0_a_;
            Flag_l1a_PingPong = 1 - Flag_l1a_PingPong;
        }

        if ((src_len >= 2 && k_idx == switch_index_list[cur_switch_index + 1])) {
            cur_switch_index++;
        }
    }
    SET_FLAG(MTE1, MTE2, ping_pong_flag_l1_b_);
    SET_FLAG(MTE1, MTE2, ping_pong_flag_l1_a_ + 2);

    ping_pong_flag_l1_b_ = 1 - ping_pong_flag_l1_b_;
    ping_pong_flag_l1_a_ = 1 - ping_pong_flag_l1_a_;
}

template <typename TYPE, bool IF_BF16>
template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
__aicore__ inline __attribute__((always_inline)) void CubeBackwardBandOp<TYPE, IF_BF16>::cube2_matmul_op(
    const Address::PhyAddrBackwardCube2<T_LEFT, T_RIGHT, T_OUTPUT> *src, int64_t src_len,
    int64_t vcore_num_per_head, uint64_t n_offset, uint64_t n_step, uint64_t n_size, uint64_t n_size_k)
{
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
    auto n_k = n_size_k;

    auto m0_ = 128;
    auto k0_ = 128;

    for (int32_t idx = 0; idx < src_len; ++idx) {
        auto left_start_addr = src[idx].left;
        auto right_start_addr = src[idx].right;
        auto result_gm = src[idx].out;
        int32_t Ky = src[idx].ky;
        int32_t Kx = src[idx].kx;
        int32_t lineStride = src[idx].lineStride;
        bool upperRight = src[idx].upperRight;
        bool lowerLeft = src[idx].lowerLeft;

        for (int32_t i = 0; i < Kx; i++) {
            bool l1_skip_flag = (upperRight && i == Kx - 1);
            bool last_k = (i >= Kx - 1);
            bool l0_c_init_flag = (i == 0);

            LocalTensor<TYPE>* l1_a_buf_tensor = ping_pong_flag_l1_a_ ? &l1_a_pong_tensor : &l1_a_ping_tensor;
            LocalTensor<TYPE>* l1_b_buf_tensor = ping_pong_flag_l1_b_ ? &l1_b_pong_tensor : &l1_b_ping_tensor;

            WAIT_FLAG(MTE1, MTE2, ping_pong_flag_l1_b_);
            WAIT_FLAG(MTE1, MTE2, ping_pong_flag_l1_a_ + 2);

            if (!l1_skip_flag) {
                temp_tensor_bf16.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(left_start_addr + i * l1_m * l1_k));
                AscendC::DataCopy(
                    *l1_a_buf_tensor,
                    temp_tensor_bf16,
                    commonNd2NzParamsFp32
                );
            }
            temp_tensor_bf16.SetGlobalBuffer(
                reinterpret_cast<__gm__ TYPE *>(left_start_addr + i * l1_k * l1_m + lineStride));
            AscendC::DataCopy(
                (*l1_a_buf_tensor)[SIZE_128 * SIZE_128],
                temp_tensor_bf16,
                commonNd2NzParamsFp32
            );
            
            commonNd2NzParams.nValue = l1_k;
            commonNd2NzParams.dValue = l1_n;
            commonNd2NzParams.srcDValue = n_k;
            commonNd2NzParams.dstNzC0Stride = l1_k;
            temp_tensor_bf16.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(right_start_addr + i * l1_k * n_k));
            AscendC::DataCopy(
                *l1_b_buf_tensor,
                temp_tensor_bf16,
                commonNd2NzParams
            );
            
            SET_FLAG(MTE2, MTE1, ping_pong_flag_l1_b_);
            WAIT_FLAG(MTE2, MTE1, ping_pong_flag_l1_b_);
            SET_FLAG(MTE2, MTE1, ping_pong_flag_l1_a_ + 2);
            WAIT_FLAG(MTE2, MTE1, ping_pong_flag_l1_a_ + 2);

            for (int32_t n_offset = 0; n_offset < l1_n; n_offset += SIZE_128) {
                LocalTensor<TYPE>* l0_b_buf_tensor = ping_pong_flag_l0_b_ ? &l0_b_pong_tensor : &l0_b_ping_tensor;
                WAIT_FLAG(M, MTE1, ping_pong_flag_l0_b_ + 2);
                commonLoadData2dParamsTranspose.repeatTimes = n0_ / SIZE_16;
                commonLoadData2dParamsTranspose.srcStride = l1_k / SIZE_16;
                for (int nn = 0; nn < k0_ / SIZE_16; nn++) {
                    AscendC::LoadData(
                        (*l0_b_buf_tensor)[nn * n0_ * SIZE_16],
                        (*l1_b_buf_tensor)[n_offset * l1_k + nn * SIZE_256],
                        commonLoadData2dParamsTranspose
                    );
                }
                SET_FLAG(MTE1, M, ping_pong_flag_l0_b_ + 2);
                WAIT_FLAG(MTE1, M, ping_pong_flag_l0_b_ + 2);

                for (int32_t j = 0; j < Ky; j++) {
                    bool l0_skip_flag = (l1_skip_flag && j == 0);
                    bool last_k = (i == Kx - 1 && j != 0) || (i == Kx - 1 && j == 0 && !upperRight) ||
                                  (upperRight && i == Kx - 2 && j == 0);
                    LocalTensor<TYPE>* l0_a_buf_tensor = ping_pong_flag_l0_a_ ? &l0_a_pong_tensor : &l0_a_ping_tensor;
                    LocalTensor<float>* l0_c_buf_tensor = ping_pong_flag_l0_c_ ? &l0_c_pong_tensor : &l0_c_ping_tensor;
                    WAIT_FLAG(M, MTE1, ping_pong_flag_l0_a_);
                    if (!l0_skip_flag) {
                        commonLoadData2dParamsNoTranspose.repeatTimes = k0_ / SIZE_16;
                        commonLoadData2dParamsNoTranspose.srcStride = l1_m / SIZE_16;
                        for (int32_t jj = 0; jj < m0_ / SIZE_16; jj++) {
                            AscendC::LoadData(
                                (*l0_a_buf_tensor)[jj * k0_ * SIZE_16],
                                (*l1_a_buf_tensor)[j * SIZE_128 * SIZE_128 + jj * SIZE_256],
                                commonLoadData2dParamsNoTranspose
                            );
                        }
                    }
                    SET_FLAG(MTE1, M, ping_pong_flag_l0_a_);
                    WAIT_FLAG(MTE1, M, ping_pong_flag_l0_a_);

                    commonMadParams.unitFlag = last_k ? 3 : 2;
                    commonMadParams.cmatrixInitVal = l0_c_init_flag;
                    AscendC::Mmad(
                        *l0_c_buf_tensor,
                        *l0_a_buf_tensor,
                        *l0_b_buf_tensor,
                        commonMadParams
                    );

                    if (last_k) {
                        AscendC::SetAtomicType<float>();
                        commonFixpipeParamsV220.dstStride = n_;
                        temp_tensor_fp32.SetGlobalBuffer(
                            reinterpret_cast<__gm__ float *>(result_gm + j * SIZE_128 * n_ + n_offset));
                        AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
                            temp_tensor_fp32,
                            *l0_c_buf_tensor,
                            commonFixpipeParamsV220
                        );
                        AscendC::SetAtomicNone();
                    }

                    SET_FLAG(M, MTE1, ping_pong_flag_l0_a_);
                    ping_pong_flag_l0_a_ = 1 - ping_pong_flag_l0_a_;
                    ping_pong_flag_l0_c_ = 1 - ping_pong_flag_l0_c_;
                }
                SET_FLAG(M, MTE1, ping_pong_flag_l0_b_ + 2);
                ping_pong_flag_l0_b_ = 1 - ping_pong_flag_l0_b_;
            }
            SET_FLAG(MTE1, MTE2, ping_pong_flag_l1_b_);
            SET_FLAG(MTE1, MTE2, ping_pong_flag_l1_a_ + 2);

            ping_pong_flag_l1_a_ = 1 - ping_pong_flag_l1_a_;
            ping_pong_flag_l1_b_ = 1 - ping_pong_flag_l1_b_;
        }
    }
}

template <typename TYPE, bool IF_BF16>
template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
__aicore__ inline __attribute__((always_inline)) void CubeBackwardBandOp<TYPE, IF_BF16>::cube1_matmul_op(
    const Address::PhyAddrBackwardCube1<T_LEFT, T_RIGHT, T_OUTPUT> *src, int64_t src_len)
{
    for (int32_t idx = 0; idx < src_len; ++idx) {
        auto kx = src[idx].kx;
        auto ky = src[idx].ky;
        auto lineStride = src[idx].lineStride;

        int32_t m_loop = 1;
        int32_t n_loop = kx;
        auto l1_m_size_ = ky * 128;
        auto l1_k_size_ = 128;
        auto l1_n_size_ = 128;
        auto k_ = 128;
        auto n_ = 128;

        auto gm_a = src[idx].left;
        auto gm_b = src[idx].right;
        auto gm_c = src[idx].out;
        bool upperRight = src[idx].upperRight;
        bool lowerLeft = src[idx].lowerLeft;

        commonNd2NzParams.nValue = l1_m_size_;
        commonNd2NzParams.dValue = l1_k_size_;
        commonNd2NzParams.srcDValue = l1_k_size_;
        commonNd2NzParams.dstNzC0Stride = l1_m_size_;
        for (int m_index = 0; m_index < m_loop; m_index++) {
            LocalTensor<TYPE>* l1_a_tensor = ping_pong_flag_l1_a_ ? &l1_a_pong_tensor : &l1_a_ping_tensor;
            WAIT_FLAG(MTE1, MTE2, ping_pong_flag_l1_a_ + 2);
            temp_tensor_bf16.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(gm_a));
            AscendC::DataCopy(
                *l1_a_tensor,
                temp_tensor_bf16,
                commonNd2NzParams
            );

            SET_FLAG(MTE2, MTE1, ping_pong_flag_l1_a_ + 2);
            WAIT_FLAG(MTE2, MTE1, ping_pong_flag_l1_a_ + 2);
            
            commonNd2NzParams.nValue = l1_k_size_;
            commonNd2NzParams.dValue = l1_n_size_;
            commonNd2NzParams.srcDValue = l1_n_size_;
            commonNd2NzParams.dstNzC0Stride = l1_k_size_;
            for (int n_index = 0; n_index < n_loop; n_index++) {
                bool upper_right_flag = (upperRight && n_index == n_loop - 1);
                LocalTensor<TYPE>* l1_b_tensor = ping_pong_flag_l1_b_ ? &l1_b_pong_tensor : &l1_b_ping_tensor;
                WAIT_FLAG(MTE1, MTE2, ping_pong_flag_l1_b_);
                temp_tensor_bf16.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(gm_b + n_index * l1_n_size_ * k_));
                AscendC::DataCopy(
                    *l1_b_tensor,
                    temp_tensor_bf16,
                    commonNd2NzParams
                );

                SET_FLAG(MTE2, MTE1, ping_pong_flag_l1_b_);
                WAIT_FLAG(MTE2, MTE1, ping_pong_flag_l1_b_);

                cube1_base_matmul(l1_a_tensor, l1_b_tensor, gm_c, ky, lineStride, upper_right_flag);

                SET_FLAG(MTE1, MTE2, ping_pong_flag_l1_b_);
                ping_pong_flag_l1_b_ = 1 - ping_pong_flag_l1_b_;
                gm_c += SIZE_128 * SIZE_128;
            }

            SET_FLAG(MTE1, MTE2, ping_pong_flag_l1_a_ + 2);
            ping_pong_flag_l1_a_ = 1 - ping_pong_flag_l1_a_;
        }
    }
}

template <typename TYPE, bool IF_BF16>
__aicore__ inline void CubeBackwardBandOp<TYPE, IF_BF16>::Run_op()
{
    AscendC::SetLoadDataPaddingValue<uint64_t>(0);
    uint64_t config = 0x1;
    AscendC::SetNdParaImpl(config);

    preset_flag();
    int64_t Z = address.get_total_rounds();

    if (Z == 1) {
        for (int64_t roundId = 0; roundId < Z; roundId++) {
            if (address.is_running(roundId) && is_cube1) {
                // cube1
                int64_t len_S = 0;
                Address::PhyAddrBackwardCube1<TYPE, TYPE, float> addr_S[MAX_SWITCH_TIME];
                address.addrMapping_cube1(gm_Q, gm_K, gm_S, addr_S, len_S, roundId, this->headDim, this->headDim);
                cube1_matmul_op(addr_S, len_S);

                int64_t len_dP = 0;
                Address::PhyAddrBackwardCube1<TYPE, TYPE, float> addr_dP[MAX_SWITCH_TIME];
                address.addrMapping_cube1(gm_dO, gm_V, gm_dP, addr_dP, len_dP, roundId, this->headDim, this->headDim);
                cube1_matmul_op(addr_dP, len_dP);
            }

            if (is_syn) {
                FftsCrossCoreSync<PIPE_FIX, 2>(AIC2AIVFLAGID);
                WaitFlagDev(AIV2AICFLAGID);
            }

            if (address.is_running(roundId)) {
                // cube3
                int64_t len_dV = 0;
                Address::PhyAddrBackwardCube3<float, TYPE, float> addr_dV[MAX_SWITCH_TIME];
                address.addrMapping_cube3(gm_S, gm_dO, gm_dV, addr_dV, len_dV, roundId, this->headDim, this->headDim);
                cube3_matmul_op(addr_dV, len_dV, 2, 128, 128, 128);

                int64_t len_dK = 0;
                Address::PhyAddrBackwardCube3<float, TYPE, float> addr_dK[MAX_SWITCH_TIME];
                address.addrMapping_cube3(gm_dP, gm_Q, gm_dK, addr_dK, len_dK, roundId, this->headDim, this->headDim);
                cube3_matmul_op(addr_dK, len_dK, 2, 128, 128, 128);

                // cube2
                int64_t len_dQ = 0;
                Address::PhyAddrBackwardCube2<float, TYPE, float> addr_dQ[MAX_SWITCH_TIME];
                address.addrMapping_cube2(gm_dP, gm_K, gm_dQ, addr_dQ, len_dQ, roundId, this->headDim, this->headDim);
                cube2_matmul_op(addr_dQ, len_dQ, 2, 0, 128, 128, 128);
            }
        }
    } else {
        for (int64_t roundId = 0; roundId < 2; roundId++) {
            if (address.is_running(roundId) && is_cube1) {
                int64_t len_S = 0;
                Address::PhyAddrBackwardCube1<TYPE, TYPE, float> addr_S[MAX_SWITCH_TIME];
                address.addrMapping_cube1(gm_Q, gm_K, gm_S, addr_S, len_S, roundId, this->headDim, this->headDim);
                cube1_matmul_op(addr_S, len_S);

                int64_t len_dP = 0;
                Address::PhyAddrBackwardCube1<TYPE, TYPE, float> addr_dP[MAX_SWITCH_TIME];
                address.addrMapping_cube1(gm_dO, gm_V, gm_dP, addr_dP, len_dP, roundId, this->headDim, this->headDim);
                cube1_matmul_op(addr_dP, len_dP);
            }

            if (is_syn) {
                if (roundId == 0) {
                    FftsCrossCoreSync<PIPE_FIX, 2>(AIC2AIVFLAGID);
                }
            }
        }
        // cube2 + cube3 + cube1
        for (int64_t roundId = 2; roundId < Z; roundId++) {
            if (is_syn) {
                WaitFlagDev(AIV2AICFLAGID);
                FftsCrossCoreSync<PIPE_FIX, 2>(AIC2AIVFLAGID);
            }

            if (address.is_running(roundId - 2)) {
                // cube3
                int64_t len_dV = 0;
                Address::PhyAddrBackwardCube3<float, TYPE, float> addr_dV[MAX_SWITCH_TIME];
                address.addrMapping_cube3(gm_S, gm_dO, gm_dV, addr_dV, len_dV,
                    roundId - 2, this->headDim, this->headDim);
                cube3_matmul_op(addr_dV, len_dV, 2, 128, 128, 128);

                int64_t len_dK = 0;
                Address::PhyAddrBackwardCube3<float, TYPE, float> addr_dK[MAX_SWITCH_TIME];
                address.addrMapping_cube3(gm_dP, gm_Q, gm_dK, addr_dK, len_dK,
                    roundId - 2, this->headDim, this->headDim);
                cube3_matmul_op(addr_dK, len_dK, 2, 128, 128, 128);

                // cube2
                int64_t len_dQ = 0;
                Address::PhyAddrBackwardCube2<float, TYPE, float> addr_dQ[MAX_SWITCH_TIME];
                address.addrMapping_cube2(gm_dP, gm_K, gm_dQ, addr_dQ, len_dQ,
                    roundId - 2, this->headDim, this->headDim);
                cube2_matmul_op(addr_dQ, len_dQ, 2, 0, 128, 128, 128);
            }

            if (address.is_running(roundId) && is_cube1) {
                // cube1
                int64_t len_S = 0;
                Address::PhyAddrBackwardCube1<TYPE, TYPE, float> addr_S[MAX_SWITCH_TIME];
                address.addrMapping_cube1(gm_Q, gm_K, gm_S, addr_S, len_S, roundId, this->headDim, this->headDim);
                cube1_matmul_op(addr_S, len_S);

                int64_t len_dP = 0;
                Address::PhyAddrBackwardCube1<TYPE, TYPE, float> addr_dP[MAX_SWITCH_TIME];
                address.addrMapping_cube1(gm_dO, gm_V, gm_dP, addr_dP, len_dP, roundId, this->headDim, this->headDim);
                cube1_matmul_op(addr_dP, len_dP);
            }
        }

        /**** (cube2 + cube3) * 2 ****/
        for (int64_t roundId = 0; roundId < 2; roundId++) {
            if (is_syn) {
                WaitFlagDev(AIV2AICFLAGID);
                if (roundId == 0) {
                    FftsCrossCoreSync<PIPE_FIX, 2>(AIC2AIVFLAGID);
                }
            }

            if (address.is_running(roundId + Z - 2)) {
                // cube3
                int64_t len_dV = 0;
                Address::PhyAddrBackwardCube3<float, TYPE, float> addr_dV[MAX_SWITCH_TIME];
                address.addrMapping_cube3(gm_S, gm_dO, gm_dV, addr_dV, len_dV,
                        roundId + Z - 2, this->headDim, this->headDim);
                cube3_matmul_op(addr_dV, len_dV, 2, 128, 128, 128);

                int64_t len_dK = 0;
                Address::PhyAddrBackwardCube3<float, TYPE, float> addr_dK[MAX_SWITCH_TIME];
                address.addrMapping_cube3(gm_dP, gm_Q, gm_dK, addr_dK, len_dK,
                        roundId + Z - 2, this->headDim, this->headDim);
                cube3_matmul_op(addr_dK, len_dK, 2, 128, 128, 128);

                // cube2
                int64_t len_dQ = 0;
                Address::PhyAddrBackwardCube2<float, TYPE, float> addr_dQ[MAX_SWITCH_TIME];
                address.addrMapping_cube2(gm_dP, gm_K, gm_dQ, addr_dQ,
                        len_dQ, roundId + Z - 2, this->headDim, this->headDim);
                cube2_matmul_op(addr_dQ, len_dQ, 2, 0, 128, 128, 128);
            }
        }
    }
    clear_flag();
}

template <typename TYPE, bool IF_BF16>
__aicore__ inline void CubeBackwardBandOp<TYPE, IF_BF16>::Init(
    __gm__ TYPE *__restrict__ gm_Q,
    __gm__ TYPE *__restrict__ gm_K,
    __gm__ TYPE *__restrict__ gm_V,
    __gm__ TYPE *__restrict__ gm_dO,
    __gm__ float *__restrict__ gm_dP,
    __gm__ float *__restrict__ gm_S,
    __gm__ float *__restrict__ gm_dQ,
    __gm__ float *__restrict__ gm_dK,
    __gm__ float *__restrict__ gm_dV,
    bool isTri, int64_t qSize, int64_t kSize, int64_t batch, int64_t head, int64_t group_size, int64_t base_length,
    int64_t sparseMode, int64_t windowLength, int64_t blockNumPerCore)
{
    this->gm_Q = gm_Q;
    this->gm_K = gm_K;
    this->gm_V = gm_V;
    this->gm_dO = gm_dO;
    this->gm_dP = gm_dP;
    this->gm_S = gm_S;

    this->gm_dQ = gm_dQ;
    this->gm_dK = gm_dK;
    this->gm_dV = gm_dV;

    this->seqLenQ = qSize;
    this->seqLenK = kSize;
    this->batchSize = batch;
    this->headNum = head;
    this->headDim = base_length;
    this->gqa_group_num = (head + group_size - 1) / group_size;
    this->blockNumPerCore = blockNumPerCore;
    this->isTri = isTri;
    this->sparseMode = sparseMode;
    this->windowLength = windowLength;

    this->core_num = get_block_num();
    this->cur_core_index = get_block_idx();

    this->blocks_per_column = this->isTri ? ((this->seqLenQ) / 128 / 2) : ((this->seqLenQ) / 128);
    this->blocks_per_row = this->isTri ? ((this->seqLenK) / 128 + 1) : ((this->seqLenK) / 128);

    if (this->isTri && this->sparseMode == 1) {
        this->blocks_per_column = (this->seqLenQ / 128) - (this->windowLength / 128 / 2);
        this->blocks_per_row = (this->windowLength / 128) + 1;
    }

    is_cube1 = true;
    is_cube2 = true;
    is_cube3 = true;
    is_syn = true;

    address.init(
        this->batchSize,
        this->headNum,
        this->seqLenQ,
        this->seqLenK,
        this->headDim,
        this->gqa_group_num,
        this->isTri,
        this->sparseMode,
        this->windowLength
    );

    address.set_backward_tiling(this->core_num, this->cur_core_index, this->blockNumPerCore, 2);

    temp_tensor_bf16.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(0));
    temp_tensor_fp32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(0));

    // init L1 tensor
    l1_base_a_cube1_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(0);
    l1_base_b_cube1_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_128 * SIZE_ONE_K);
    l1_base_a_cube2_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_256 * SIZE_ONE_K);
    l1_base_b_cube2_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_384 * SIZE_ONE_K);

    l1_a_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(0);
    l1_a_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_128 * SIZE_ONE_K);
    l1_b_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_256 * SIZE_ONE_K);
    l1_b_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_384 * SIZE_ONE_K);

    // workspace ping pong addr shift
    l1_a_ping_double_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_LONG_BLOCK * 2);
    l1_a_pong_double_tensor =
        asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_128 * SIZE_ONE_K + SIZE_LONG_BLOCK * 2);
    l1_b_ping_double_tensor =
        asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_256 * SIZE_ONE_K + SIZE_LONG_BLOCK * 2);
    l1_b_pong_double_tensor =
        asdopsBuf.GetBuffer<BufferType::ASCEND_CB, TYPE>(SIZE_384 * SIZE_ONE_K + SIZE_LONG_BLOCK * 2);

    // init L0A/L0B/L0C tensor
    l0_a_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0A, TYPE>(0);
    l0_a_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0A, TYPE>(SIZE_32 * SIZE_ONE_K);
    l0_b_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0B, TYPE>(0);
    l0_b_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0B, TYPE>(SIZE_32 * SIZE_ONE_K);
    l0_c_buf_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(0);
    l0_c_ping_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(0);
    l0_c_pong_tensor = asdopsBuf.GetBuffer<BufferType::ASCEND_L0C, float>(SIZE_64 * SIZE_ONE_K);

    commonFixpipeParamsV220.quantPre = QuantMode_t::NoQuant;
    commonFixpipeParamsV220.unitFlag = 3;
}

}  // namespace CUBE_BACK_BAND_OP
#endif  // __DAV_C220_CUBE__

#endif  // __CUBEFORWARD_H__