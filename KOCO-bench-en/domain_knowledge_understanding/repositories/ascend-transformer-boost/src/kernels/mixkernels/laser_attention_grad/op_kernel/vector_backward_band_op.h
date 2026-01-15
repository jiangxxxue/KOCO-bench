/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef __VECTORBACKWARD_BAND_OP_H__
#define __VECTORBACKWARD_BAND_OP_H__
#define USE_ASCENDC 1

#include "address_mapping_vector.h"
#include "ppmatmul_const_grad.h"

#include "kernel_operator.h"
#include "kernels/utils/kernel/common.h"
#include "kernels/utils/kernel/simd.h"
#include "kernels/utils/kernel/iterator.h"
#include "kernels/utils/kernel/utils.h"
using namespace AscendC;

#ifdef __DAV_C220_VEC__

namespace VEC_BACKWARD_BAND_OP {

using WORKSPACE_DTYPE = float;
using WORKSPACE_DTYPE_DP = float;
constexpr int32_t repeat_time_fp32 = 4;

template <typename TYPE>
struct UB_FOR_BACKWARD {
    LocalTensor<float> fp32_temp_0_ub_tensor_addr{};   // O,OdO,S,exp(s-L)
    LocalTensor<float> fp32_temp_1_ub_tensor_addr{};   // dO,dp
    LocalTensor<TYPE> fp16_temp_0_ub_tensor_addr{};    // O,OdO,S
    LocalTensor<TYPE> fp16_temp_1_ub_tensor_addr{};    // dO,dp
    LocalTensor<float> tensor_for_cacl_rowsum_fp32{};  // d=rowsum(odo)
    // __ubuf__ TYPE* buf_for_cacl_rowsum_fp16; //d=rowsum(odo)
    LocalTensor<float> tensor_for_brcb_rowsum_fp32{};  // 8倍rowsum
    LocalTensor<float> tensor_for_load_l_fp32{};       // 装载参数L
    LocalTensor<float> tensor_for_brcb_l_fp32{};       // 8倍L
    LocalTensor<float> tensor_for_rowmax_fp32{};
    LocalTensor<float> tensor_for_rowsum_fp32{};
    LocalTensor<TYPE> fp16_temp_2_ub_tensor_addr{};   // exp(s-l)转fp16结果,ds转fp16结果
    LocalTensor<TYPE> tensor_for_mask_fp16{};         // mask
    LocalTensor<float> tensor_for_mask_fp32{};        // mask
};                                                      // UB空间划分，求归一化

template <typename INPUT_T, bool IF_BF16>
class VectorBackwardBandOp {
public:
    __aicore__ inline VectorBackwardBandOp()
    {}
    __aicore__ inline void Init (__gm__ INPUT_T *__restrict__ gm_dO,
                                __gm__ INPUT_T *__restrict__ gm_O,
                                __gm__ WORKSPACE_DTYPE *__restrict__ gm_S,
                                __gm__ float *__restrict__ gm_rowmax,
                                __gm__ float *__restrict__ gm_rowsum,
                                __gm__ WORKSPACE_DTYPE_DP *__restrict__ gm_dP,
                                __gm__ INPUT_T *__restrict__ gm_mask, bool isTri,
                                int32_t qSize,
                                int32_t kSize,
                                int32_t batch,
                                int32_t head,
                                int32_t base_length,
                                int32_t blockNumPerCore,
                                int32_t isSparse,
                                int32_t windows_length,
                                int32_t maskSeqLength,
                                float scale);
    __aicore__ inline void Run_op();

private:
    __aicore__ inline void allocate_ubuf_for_norm(UB_FOR_BACKWARD<INPUT_T> *ub_bkd);
    __aicore__ inline void process_backward_fp32(
        int32_t cur_proc_lines,  // 总处理64行（2个vector处理一个基本块）BASE_BLOCK_SIZE_LEN_BACKWARD /2
        int32_t total_offset_o,  // O,dO的总偏移量，global_lines_in_heads * BASE_BLOCK_SIZE_LEN_BACKWARD    //
                                 // L是global_lines_in_heads
        int32_t block_number,    // S,dP一次处理几块  end_block - start_block
        int32_t total_offset_s,  // S,dP的总偏移量start_block * BASE_BLOCK_DATA_NUM_BACKWARD
        UB_FOR_BACKWARD<INPUT_T> ub_backward, bool head_apply_mask, bool tail_apply_mask);

private:
    GlobalTensor<INPUT_T>(gm_dO_tensor);             // dO
    GlobalTensor<INPUT_T>(gm_O_tensor);              // O
    GlobalTensor<WORKSPACE_DTYPE>(gm_S_tensor);      // S
    GlobalTensor<float>(gm_rowmax_tensor);
    GlobalTensor<float>(gm_rowsum_tensor);
    GlobalTensor<WORKSPACE_DTYPE_DP>(gm_dP_tensor);
    GlobalTensor<INPUT_T>(gm_mask_tensor);

    AsdopsBuffer<ArchType::ASCEND_V220> calBuf;

    Address::GLOBAL_INFO global_info;
    Address::LOCAL_INFO local_info;
    Address::SECTION_INFO section_info;
    UB_FOR_BACKWARD<INPUT_T> ub_backward;
    bool is_vector;
    bool is_syn;
    int32_t H;
    int32_t maskSeqLength;
    float SCALE;

    Address::AddressMappingVector address_vector;
};

template <typename INPUT_T, bool IF_BF16>
__aicore__ inline void VectorBackwardBandOp<INPUT_T, IF_BF16>::Run_op()
{
    set_atomic_none();
    SetMaskNorm();
    SetVectorMask<float, AscendC::MaskMode::NORMAL>((uint64_t)-1, (uint64_t)-1);

    for (int64_t round_id = 0; round_id < address_vector.get_total_rounds(); round_id++) {
        if (is_syn) {
            wait_flag_dev(AIC2AIVFLAGID);
        }
        if (address_vector.is_running(round_id) && is_vector) {
            auto section_vector = address_vector.get_section_info(round_id);
            for (int64_t section = 0; section < section_vector.sectionNum; section++) {
                auto start_block = section_vector.section_start_block[section];
                auto head_apply_mask = section_vector.head_apply_mask[section];
                auto tail_apply_mask = section_vector.tail_apply_mask[section];
                auto cur_process_blk_num = section_vector.len[section];
                auto processLines = section_vector.processLines;
                auto O_dO_offset = section_vector.O_dO_offset[section];
                auto S_dP_offset = section_vector.S_dP_offset[section];

                process_backward_fp32(processLines,
                    O_dO_offset,
                    cur_process_blk_num,
                    S_dP_offset,
                    ub_backward,
                    head_apply_mask,
                    tail_apply_mask);
            }
        }
        if (is_syn) {
            FftsCrossCoreSync<PIPE_MTE3, 2>(AIV2AICFLAGID);
        }
    }
}

template <typename INPUT_T, bool IF_BF16>
__aicore__ inline void VectorBackwardBandOp<INPUT_T, IF_BF16>::Init(__gm__ INPUT_T *__restrict__ gm_dO,
                                                            __gm__ INPUT_T *__restrict__ gm_O,
                                                            __gm__ WORKSPACE_DTYPE *__restrict__ gm_S,
                                                            __gm__ float *__restrict__ gm_rowmax,
                                                            __gm__ float *__restrict__ gm_rowsum,
                                                            __gm__ WORKSPACE_DTYPE_DP *__restrict__ gm_dP,
                                                            __gm__ INPUT_T *__restrict__ gm_mask,
                                                            bool isTri,
                                                            int32_t qSize,
                                                            int32_t kSize,
                                                            int32_t batch,
                                                            int32_t head,
                                                            int32_t base_length,
                                                            int32_t blockNumPerCore,
                                                            int32_t isSparse,
                                                            int32_t windows_length,
                                                            int32_t maskSeqLength,
                                                            float scale)
{
    this->maskSeqLength = maskSeqLength;
    this->SCALE = scale;
    this->gm_dO_tensor.SetGlobalBuffer(gm_dO);
    this->gm_O_tensor.SetGlobalBuffer(gm_O);
    this->gm_S_tensor.SetGlobalBuffer(gm_S);
    this->gm_rowmax_tensor.SetGlobalBuffer(gm_rowmax);
    this->gm_rowsum_tensor.SetGlobalBuffer(gm_rowsum);
    this->gm_dP_tensor.SetGlobalBuffer(gm_dP);
    this->gm_mask_tensor.SetGlobalBuffer(gm_mask);

    is_vector = true;
    is_syn = true;

    this->global_info.seqLenQ = qSize;  // tiling_para[10]
    this->global_info.seqLenK = kSize;
    this->global_info.headNum = head;                          // tiling_para[12]
    this->global_info.batchNum = batch;                        // tiling_para[0]
    this->global_info.blockNumPerCube = blockNumPerCore;  //  tiling_para[11] / 128          // 暂定下标1
    this->global_info.triMatrix = isTri;                       // 是不是也从tiling里读取？

    this->global_info.isSparse = isSparse;
    this->global_info.windowsBlockNum = windows_length / BASE_BLOCK_SIZE_LEN_BACKWARD;

    H = qSize / 128;
    allocate_ubuf_for_norm(&this->ub_backward);

    // 寻址全局信息初始化
    int64_t headDim = 128;
    int64_t cubeNum = get_block_num();
    address_vector.init_global_info(batch, head, qSize, kSize, headDim, isTri,
        isSparse, windows_length, cubeNum, blockNumPerCore, 2);

    // 寻址局部信息初始化
    int64_t cube_core_index = get_block_idx();
    int64_t vector_core_index = get_subblockid();
    address_vector.init_local_info(cube_core_index, vector_core_index);
}

/**
 * 为反向计算分配UB空间
 */
template <typename INPUT_T, bool IF_BF16>
__aicore__ inline void VectorBackwardBandOp<INPUT_T, IF_BF16>::allocate_ubuf_for_norm(UB_FOR_BACKWARD<INPUT_T> *ub_bkd)
{
    int32_t sizeof_fp32_temp_0_ub_buf_addr = MAX_LENG_PER_UB_PROC_BACKWARD * 4 * 2;  // o       //32kb
    int32_t sizeof_fp32_temp_1_ub_buf_addr = MAX_LENG_PER_UB_PROC_BACKWARD * 4 * 2;  // do       //32kb   //64kb,mask

    int32_t sizeof_fp16_temp_0_ub_buf_addr = MAX_LENG_PER_UB_PROC_BACKWARD * 2 * 2;  //  ping-pong    //16kb
    int32_t sizeof_fp16_temp_1_ub_buf_addr = MAX_LENG_PER_UB_PROC_BACKWARD * 2 * 2;  //              //16kb

    int32_t sizeof_buf_for_cacl_rowsum_fp32 = 128 * 4 * 2;  // 最多处理的FP32数字个数  //d=rowsum(odo)
    int32_t sizeof_buf_for_brcb_rowsum_fp32 =
        sizeof_buf_for_cacl_rowsum_fp32 * (32 / 4);  // 展开做32字节对齐       //8倍rowsum

    int32_t sizeof_buf_for_load_l_fp32 = 128 * 4 * 2;                            // L
    int32_t sizeof_buf_for_brcb_l_fp32 = sizeof_buf_for_load_l_fp32 * (32 / 4);  // 8倍L

    int32_t sizeof_buf_for_rowmax_fp32 = BASE_BLOCK_SIZE_LEN_BACKWARD * 4 * 2;
    int32_t sizeof_buf_for_rowsum_fp32 = BASE_BLOCK_SIZE_LEN_BACKWARD * 4 * 2;

    int32_t sizeof_buf_for_mask_fp16 = 64 * 128 * 2;
    int32_t sizeof_buf_for_mask_fp32 = 64 * 128 * 4;

    int32_t last_addr = 0;

    ub_bkd->fp32_temp_0_ub_tensor_addr = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(last_addr);
    last_addr += sizeof_fp32_temp_0_ub_buf_addr;

    ub_bkd->fp32_temp_1_ub_tensor_addr = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(last_addr);
    last_addr += sizeof_fp32_temp_1_ub_buf_addr;

    ub_bkd->fp16_temp_0_ub_tensor_addr = calBuf.GetBuffer<BufferType::ASCEND_UB, INPUT_T>(last_addr);
    last_addr += sizeof_fp16_temp_0_ub_buf_addr;

    ub_bkd->fp16_temp_1_ub_tensor_addr = calBuf.GetBuffer<BufferType::ASCEND_UB, INPUT_T>(last_addr);
    last_addr += sizeof_fp16_temp_1_ub_buf_addr;

    // cacl rowsum  32字节对齐后，输出到 buf_for_brcb_rowsum_fp32
    ub_bkd->tensor_for_cacl_rowsum_fp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(last_addr);
    last_addr += sizeof_buf_for_cacl_rowsum_fp32;

    // 32字节对齐后，输出到 buf_for_brcb_rowsum_fp32 last_addr += sizeof_buf_for_cacl_rowsum_fp16;
    ub_bkd->tensor_for_brcb_rowsum_fp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(last_addr);
    last_addr += sizeof_buf_for_brcb_rowsum_fp32;

    // D计算完毕后,s复用fp32_temp_0_ub_buf_addr，s-l也可以用该空间，exp(s-l)
    ub_bkd->tensor_for_load_l_fp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(last_addr); // load l
    last_addr += sizeof_buf_for_load_l_fp32;

    ub_bkd->tensor_for_brcb_l_fp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(last_addr);   // load 8倍l
    last_addr += sizeof_buf_for_brcb_l_fp32;

    ub_bkd->tensor_for_rowmax_fp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(last_addr); // load rowmax
    last_addr += sizeof_buf_for_rowmax_fp32;

    ub_bkd-> tensor_for_rowsum_fp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(last_addr);   // load rowsum
    last_addr += sizeof_buf_for_rowsum_fp32;

    ub_bkd-> tensor_for_mask_fp16 = calBuf.GetBuffer<BufferType::ASCEND_UB, INPUT_T>(last_addr);
    last_addr += sizeof_buf_for_mask_fp16;

    ub_bkd-> tensor_for_mask_fp32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(last_addr);
}

template <typename INPUT_T, bool IF_BF16>
__aicore__ inline void VectorBackwardBandOp<INPUT_T, IF_BF16>::process_backward_fp32(
    int32_t cur_proc_lines,  // 总处理64行（2个vector处理一个基本块）BASE_BLOCK_SIZE_LEN_BACKWARD /2
    int32_t total_offset_o,  // O,dO的总偏移量，global_lines_in_heads * BASE_BLOCK_SIZE_LEN_BACKWARD
                             // L是global_lines_in_heads
    int32_t block_number,    // S,dP一次处理几块  end_block - start_block
    int32_t total_offset_s,  // S,dP的总偏移量start_block * BASE_BLOCK_DATA_NUM_BACKWARD
    UB_FOR_BACKWARD<INPUT_T> ub_backward, bool head_apply_mask, bool tail_apply_mask)
{
    int32_t ping_flag = 0;

    SET_FLAG(V, MTE2, EVENT_ID0);
    SET_FLAG(V, MTE2, EVENT_ID1);

    for (int i = 0; i < 2; i++) {

        int32_t o_offset = cur_proc_lines / 2 * BASE_BLOCK_SIZE_LEN_BACKWARD * i;
        int32_t l_offset = o_offset / BASE_BLOCK_SIZE_LEN_BACKWARD;
        auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

        WAIT_FLAG(V, MTE2, event_id);

        LocalTensor<INPUT_T> fp16_temp_0_ub_tensor_addr_pingpong =
            ping_flag ? ub_backward.fp16_temp_0_ub_tensor_addr[MAX_LENG_PER_UB_PROC_BACKWARD]
                      : ub_backward.fp16_temp_0_ub_tensor_addr;
        LocalTensor<INPUT_T> fp16_temp_1_ub_tensor_addr_pingpong =
            ping_flag ? ub_backward.fp16_temp_1_ub_tensor_addr[MAX_LENG_PER_UB_PROC_BACKWARD]
                      : ub_backward.fp16_temp_1_ub_tensor_addr;
        LocalTensor<float> tensor_for_load_l_fp32_pingpong =
            ping_flag ? ub_backward.tensor_for_load_l_fp32[MAX_BLOCK_PER_ONE_PROC_BACKWARD]
                      : ub_backward.tensor_for_load_l_fp32;
        LocalTensor<float> tensor_for_rowmax_fp32_pingpong = ping_flag
            ? ub_backward.tensor_for_rowmax_fp32[MAX_BLOCK_PER_ONE_PROC_BACKWARD] : ub_backward.tensor_for_rowmax_fp32;
        LocalTensor<float> tensor_for_rowsum_fp32_pingpong = ping_flag
            ? ub_backward.tensor_for_rowsum_fp32[MAX_BLOCK_PER_ONE_PROC_BACKWARD] : ub_backward.tensor_for_rowsum_fp32;

        gm_to_ub<ArchType::ASCEND_V220, INPUT_T>(
            fp16_temp_0_ub_tensor_addr_pingpong,
            gm_O_tensor[total_offset_o + o_offset],
            0,
            1,
            MAX_LENG_PER_UB_PROC_BACKWARD * 2 / 32,  // 一个block=32B N*2/32
            0,
            0);

        gm_to_ub<ArchType::ASCEND_V220, INPUT_T>(
            fp16_temp_1_ub_tensor_addr_pingpong,
            gm_dO_tensor[total_offset_o + o_offset],
            0,
            1,
            MAX_LENG_PER_UB_PROC_BACKWARD * 2 / 32,
            0,
            0);

        gm_to_ub<ArchType::ASCEND_V220, float>(
            tensor_for_rowmax_fp32_pingpong,
            gm_rowmax_tensor[total_offset_o / 128 + l_offset],
            0,
            1,
            MAX_BLOCK_PER_ONE_PROC_BACKWARD * 4 / 32,
            0,
            0);

        gm_to_ub<ArchType::ASCEND_V220, float>(
            tensor_for_rowsum_fp32_pingpong,
            gm_rowsum_tensor[total_offset_o / 128 + l_offset],
            0,
            1,
            MAX_BLOCK_PER_ONE_PROC_BACKWARD * 4 / 32,
            0,
            0);

        SET_FLAG(MTE2, V, event_id);
        WAIT_FLAG(MTE2, V, event_id);

        // O,dO fp16--fp32
        SetVectorMask<float, AscendC::MaskMode::NORMAL>(0xffffffffffffffff, 0xffffffffffffffff);
        if constexpr (IF_BF16) {
            conv_v<ArchType::ASCEND_V220, INPUT_T, float>(
                ub_backward.fp32_temp_0_ub_tensor_addr,
                fp16_temp_0_ub_tensor_addr_pingpong,
                MAX_LENG_PER_UB_PROC_BACKWARD / 64,
                1,
                1,
                8,
                4);
            PIPE_BARRIER(V);
            conv_v<ArchType::ASCEND_V220, INPUT_T, float>(
                ub_backward.fp32_temp_1_ub_tensor_addr,
                fp16_temp_1_ub_tensor_addr_pingpong,
                MAX_LENG_PER_UB_PROC_BACKWARD / 64,
                1,
                1,
                8,
                4);
            PIPE_BARRIER(V);
        } else {
            conv_v<ArchType::ASCEND_V220, INPUT_T, float>(
                ub_backward.fp32_temp_0_ub_tensor_addr,
                fp16_temp_0_ub_tensor_addr_pingpong,
                MAX_LENG_PER_UB_PROC_BACKWARD / 64,
                1,
                1,
                8,
                4);
            PIPE_BARRIER(V);
            conv_v<ArchType::ASCEND_V220, INPUT_T, float>(
                ub_backward.fp32_temp_1_ub_tensor_addr,
                fp16_temp_1_ub_tensor_addr_pingpong,
                MAX_LENG_PER_UB_PROC_BACKWARD / 64,
                1,
                1,
                8,
                4);
            PIPE_BARRIER(V);
        }

        // O*dO
        SetVectorMask<float, AscendC::MaskMode::NORMAL>(0xffffffffffffffff, 0xffffffffffffffff);
        mul_v<ArchType::ASCEND_V220, float>(ub_backward.fp32_temp_0_ub_tensor_addr,
            ub_backward.fp32_temp_0_ub_tensor_addr,
            ub_backward.fp32_temp_1_ub_tensor_addr,
            MAX_LENG_PER_UB_PROC_BACKWARD / 64,
            1,
            1,
            1,
            8,
            8,
            8);

        PIPE_BARRIER(V);

        // 折半
        SetVectorMask<float, AscendC::MaskMode::NORMAL>(0x0, 0xffffffffffffffff);  // 只对每个repeat的前64个元素处理
        add_v<ArchType::ASCEND_V220, float>(ub_backward.fp32_temp_0_ub_tensor_addr,
            ub_backward.fp32_temp_0_ub_tensor_addr,
            ub_backward.fp32_temp_0_ub_tensor_addr[64],
            MAX_BLOCK_PER_ONE_PROC_BACKWARD,
            1,
            1,
            1,
            BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32,
            BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32,
            BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32);
        PIPE_BARRIER(V);

        SetVectorMask<float, AscendC::MaskMode::NORMAL>(0x0, 0xffffffff);  // 只对每个repeat的前32个元素处理
        add_v<ArchType::ASCEND_V220, float>(ub_backward.fp32_temp_0_ub_tensor_addr,
            ub_backward.fp32_temp_0_ub_tensor_addr,
            ub_backward.fp32_temp_0_ub_tensor_addr[32],
            MAX_BLOCK_PER_ONE_PROC_BACKWARD,
            1,
            1,
            1,
            BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32,
            BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32,
            BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32);
        PIPE_BARRIER(V);

        SetVectorMask<float, AscendC::MaskMode::NORMAL>(0x0, 0xffff);  // 只对每个repeat的前16个元素处理
        add_v<ArchType::ASCEND_V220, float>(ub_backward.fp32_temp_0_ub_tensor_addr,
            ub_backward.fp32_temp_0_ub_tensor_addr,
            ub_backward.fp32_temp_0_ub_tensor_addr[16],
            MAX_BLOCK_PER_ONE_PROC_BACKWARD,
            1,
            1,
            1,
            BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32,
            BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32,
            BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32);
        PIPE_BARRIER(V);

        SetVectorMask<float, AscendC::MaskMode::NORMAL>(0x0, 0xff);  // 只对每个repeat的前8个元素处理
        add_v<ArchType::ASCEND_V220, float>(ub_backward.fp32_temp_0_ub_tensor_addr,
            ub_backward.fp32_temp_0_ub_tensor_addr,
            ub_backward.fp32_temp_0_ub_tensor_addr[8],
            MAX_BLOCK_PER_ONE_PROC_BACKWARD,
            1,
            1,
            1,
            1,
            BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32,
            BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32);
        PIPE_BARRIER(V);

        // vcgadd将每32B的元素reduce add成一个数
        SetVectorMask<float, AscendC::MaskMode::NORMAL>(0x0, 0xffffffffffffffff);
        auto rowsum_offset = ping_flag ? MAX_BLOCK_PER_ONE_PROC_BACKWARD : 0;

        BlockReduceSum<float, false>(
            ub_backward.tensor_for_cacl_rowsum_fp32[rowsum_offset],
            ub_backward.fp32_temp_0_ub_tensor_addr,
            8, 0, 1, 1, 8);
        PIPE_BARRIER(V);

        SET_FLAG(V, MTE2, event_id);
        ping_flag = 1 - ping_flag;
    }

    WAIT_FLAG(V, MTE2, EVENT_ID0);
    WAIT_FLAG(V, MTE2, EVENT_ID1);

    SetVectorMask<float, AscendC::MaskMode::NORMAL>(0xffffffffffffffff, 0xffffffffffffffff);
    brcb_v<ArchType::ASCEND_V220, uint32_t>(
        ub_backward.tensor_for_brcb_rowsum_fp32.template ReinterpretCast<uint32_t>(),
        ub_backward.tensor_for_cacl_rowsum_fp32.template ReinterpretCast<uint32_t>(),
        1,
        8,
        8);
    PIPE_BARRIER(V);

    ln_v<ArchType::ASCEND_V220, float>(
        ub_backward.tensor_for_rowsum_fp32,
        ub_backward.tensor_for_rowsum_fp32,
        1,
        1,
        1,
        8,
        8);
    PIPE_BARRIER(V);

    add_v<ArchType::ASCEND_V220, float>(
        ub_backward.tensor_for_load_l_fp32,
        ub_backward.tensor_for_rowmax_fp32,
        ub_backward.tensor_for_rowsum_fp32,
        1, 1, 1, 1, BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32, BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32,
        BASE_BLOCK_SIZE_LEN_BACKWARD * 4 / 32);
    PIPE_BARRIER(V);

    brcb_v<ArchType::ASCEND_V220, uint32_t>(
        ub_backward.tensor_for_brcb_l_fp32.template ReinterpretCast<uint32_t>(),
        ub_backward.tensor_for_load_l_fp32.template ReinterpretCast<uint32_t>(),
        1,
        8,
        8);
    PIPE_BARRIER(V);

    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    SET_FLAG(MTE3, MTE2, EVENT_ID1);

    // 逐行计算
    for (int k = 0; k < block_number; k++) {
        // 每次copy 128 * block_num 个数据  ub = 128 * block_num * sizeof() * 2
        for (int n = 0; n < 2; n++) {
            int32_t s_offset =
                64 * 128 * get_subblockid() + k * 128 * 128 + n * MAX_LENG_PER_UB_PROC_BACKWARD;  // vector之间的偏移
            int32_t ss_offset = 64 * 128 * get_subblockid() + k * 128 * 128 + n * MAX_LENG_PER_UB_PROC_BACKWARD / 2;
            auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;
            WAIT_FLAG(MTE3, MTE2, event_id);

            LocalTensor<float> fp32_temp_0_ub_tensor_addr_pingpong =
                ping_flag ? ub_backward.fp32_temp_0_ub_tensor_addr[MAX_LENG_PER_UB_PROC_BACKWARD]
                      : ub_backward.fp32_temp_0_ub_tensor_addr;
            LocalTensor<INPUT_T> fp16_temp_0_ub_tensor_addr_pingpong =
                ping_flag ? ub_backward.fp16_temp_0_ub_tensor_addr[MAX_LENG_PER_UB_PROC_BACKWARD]
                        : ub_backward.fp16_temp_0_ub_tensor_addr;
            LocalTensor<INPUT_T> tensor_for_mask_fp16_pingpong =
                ping_flag ? ub_backward.tensor_for_mask_fp16[MAX_LENG_PER_UB_PROC_BACKWARD]
                        : ub_backward.tensor_for_mask_fp16;
            LocalTensor<float> tensor_for_mask_fp32_pingpong =
                ping_flag ? ub_backward.tensor_for_mask_fp32[MAX_LENG_PER_UB_PROC_BACKWARD]
                        : ub_backward.tensor_for_mask_fp32;

            gm_to_ub<ArchType::ASCEND_V220, float>(
                fp32_temp_0_ub_tensor_addr_pingpong,
                gm_S_tensor[total_offset_s + s_offset],
                0,                                // sid
                1,                                // nBurst
                MAX_LENG_PER_UB_PROC_BACKWARD * repeat_time_fp32 / 32,  // lenBurst
                0,                                // srcGap
                0                                 // dstGap
                );

            SET_FLAG(MTE2, V, event_id);
            WAIT_FLAG(MTE2, V, event_id);

            muls_v<ArchType::ASCEND_V220, float>(
                fp32_temp_0_ub_tensor_addr_pingpong,
                fp32_temp_0_ub_tensor_addr_pingpong,
                (float)SCALE,
                MAX_BLOCK_PER_ONE_PROC_BACKWARD * 2,
                1,
                1,
                8,
                8); // S*scale
            PIPE_BARRIER(V);

            if (head_apply_mask && k == 0) {
                SET_FLAG(V, MTE2, event_id);
                WAIT_FLAG(V, MTE2, event_id);

                gm_to_ub<ArchType::ASCEND_V220, INPUT_T>(
                    tensor_for_mask_fp16_pingpong,
                    gm_mask_tensor[64 * maskSeqLength * get_subblockid() + 1 + n * 32 * maskSeqLength],
                    0,
                    32,            // 1
                    128 * 2 / 32,  // MAX_LENG_PER_UB_PROC_BACKWARD * 2 *2 / 32
                    (maskSeqLength - 128) / 16,
                    0);

                SET_FLAG(MTE2, V, event_id);
                WAIT_FLAG(MTE2, V, event_id);

                Duplicate<float, false>(
                    ub_backward.fp32_temp_1_ub_tensor_addr,
                    float(1.0),
                    uint64_t(0),
                    MAX_BLOCK_PER_ONE_PROC_BACKWARD * 2,
                    1,
                    8);
                PIPE_BARRIER(V);

                conv_v<ArchType::ASCEND_V220, INPUT_T, float>(tensor_for_mask_fp32_pingpong,
                    tensor_for_mask_fp16_pingpong,
                    MAX_BLOCK_PER_ONE_PROC_BACKWARD * 2,
                    1,
                    1,
                    8,
                    4);
                PIPE_BARRIER(V);

                sub_v<ArchType::ASCEND_V220, float>(
                    tensor_for_mask_fp32_pingpong,
                    ub_backward.fp32_temp_1_ub_tensor_addr,
                    tensor_for_mask_fp32_pingpong,
                    MAX_BLOCK_PER_ONE_PROC_BACKWARD * 2,
                    1,
                    1,
                    1,
                    8,
                    8,
                    8);
                PIPE_BARRIER(V);

                muls_v<ArchType::ASCEND_V220, float>(
                    tensor_for_mask_fp32_pingpong,
                    tensor_for_mask_fp32_pingpong,
                    (float)(PADDING_FOR_MAX),
                    32 * 2,
                    1,
                    1,
                    8,
                    8);
                PIPE_BARRIER(V);

                add_v<ArchType::ASCEND_V220, float>(
                    fp32_temp_0_ub_tensor_addr_pingpong,
                    fp32_temp_0_ub_tensor_addr_pingpong,
                    tensor_for_mask_fp32_pingpong,
                    32 * 2,
                    1,
                    1,
                    1,
                    8,
                    8,
                    8);  //
                PIPE_BARRIER(V);
            }

            if (tail_apply_mask && k == block_number - 1) {

                SET_FLAG(V, MTE2, event_id);
                WAIT_FLAG(V, MTE2, event_id);

                gm_to_ub<ArchType::ASCEND_V220, INPUT_T>(
                    tensor_for_mask_fp16_pingpong,
                    gm_mask_tensor[64 * maskSeqLength * get_subblockid() + n * 32 * maskSeqLength],
                    0,
                    32,            // 1
                    128 * 2 / 32,  // MAX_LENG_PER_UB_PROC_BACKWARD * 2 *2 / 32
                    (maskSeqLength - 128) / 16,
                    0);

                SET_FLAG(MTE2, V, event_id);
                WAIT_FLAG(MTE2, V, event_id);

                conv_v<ArchType::ASCEND_V220, INPUT_T, float>(
                    tensor_for_mask_fp32_pingpong,
                    tensor_for_mask_fp16_pingpong,
                    MAX_BLOCK_PER_ONE_PROC_BACKWARD * 2,
                    1,
                    1,
                    8,
                    4);
                PIPE_BARRIER(V);

                muls_v<ArchType::ASCEND_V220, float>(
                    tensor_for_mask_fp32_pingpong,
                    tensor_for_mask_fp32_pingpong,
                    (float)(PADDING_FOR_MAX),
                    32 * 2,
                    1,
                    1,
                    8,
                    8);
                PIPE_BARRIER(V);

                add_v<ArchType::ASCEND_V220, float>(
                    fp32_temp_0_ub_tensor_addr_pingpong,
                    fp32_temp_0_ub_tensor_addr_pingpong,
                    tensor_for_mask_fp32_pingpong,
                    32 * 2,
                    1,
                    1,
                    1,
                    8,
                    8,
                    8);  //
                PIPE_BARRIER(V);
            }

            auto rowsum_offset = ping_flag ? 32 * 8 : 0;
            sub_v<ArchType::ASCEND_V220, float>(
                fp32_temp_0_ub_tensor_addr_pingpong,
                fp32_temp_0_ub_tensor_addr_pingpong,
                ub_backward.tensor_for_brcb_l_fp32[rowsum_offset],
                MAX_BLOCK_PER_ONE_PROC_BACKWARD,
                1,
                1,
                0,
                BASE_BLOCK_SIZE_LEN_BACKWARD / 8,
                BASE_BLOCK_SIZE_LEN_BACKWARD / 8,
                1);
            PIPE_BARRIER(V);
            sub_v<ArchType::ASCEND_V220, float>(
                fp32_temp_0_ub_tensor_addr_pingpong[64],
                fp32_temp_0_ub_tensor_addr_pingpong[64],
                ub_backward.tensor_for_brcb_l_fp32[rowsum_offset],
                MAX_BLOCK_PER_ONE_PROC_BACKWARD,
                1,
                1,
                0,
                BASE_BLOCK_SIZE_LEN_BACKWARD / 8,
                BASE_BLOCK_SIZE_LEN_BACKWARD / 8,
                1);
            PIPE_BARRIER(V);

            exp_v<ArchType::ASCEND_V220, float>(
                fp32_temp_0_ub_tensor_addr_pingpong,
                fp32_temp_0_ub_tensor_addr_pingpong,
                MAX_BLOCK_PER_ONE_PROC_BACKWARD * 2,
                1,
                1,
                8,
                8);  // exp(s-L)
            PIPE_BARRIER(V);

            if constexpr (IF_BF16) {
                convr_v<ArchType::ASCEND_V220, float, INPUT_T>(
                fp16_temp_0_ub_tensor_addr_pingpong,
                fp32_temp_0_ub_tensor_addr_pingpong,
                MAX_BLOCK_PER_ONE_PROC_BACKWARD * 2,
                1,
                1,
                4,
                8);
            } else {
                conv_v<ArchType::ASCEND_V220, float, INPUT_T>(
                fp16_temp_0_ub_tensor_addr_pingpong,
                fp32_temp_0_ub_tensor_addr_pingpong,
                MAX_BLOCK_PER_ONE_PROC_BACKWARD * 2,
                1,
                1,
                4,
                8);
            }

            PIPE_BARRIER(V);

            SET_FLAG(V, MTE3, event_id);
            WAIT_FLAG(V, MTE3, event_id);

            const auto &tmpGmTensor = gm_S_tensor[total_offset_s + ss_offset];
            GlobalTensor<INPUT_T> &tmp_gm_S_tensor = (GlobalTensor<INPUT_T>&)(tmpGmTensor);

            ub_to_gm<ArchType::ASCEND_V220, INPUT_T>(
                tmp_gm_S_tensor,
                fp16_temp_0_ub_tensor_addr_pingpong,
                0,
                1,
                MAX_LENG_PER_UB_PROC_BACKWARD * 2 / 32,
                0,
                0);

            SET_FLAG(MTE3, MTE2, event_id);

            ping_flag = 1 - ping_flag;
        }

        for (int m = 0; m < 2; m++) {
            int32_t dp_offset =
                64 * 128 * get_subblockid() + k * 128 * 128 + m * MAX_LENG_PER_UB_PROC_BACKWARD;  // vector之间的偏移
            int32_t p_offset = 64 * 128 * get_subblockid() + k * 128 * 128 + m * MAX_LENG_PER_UB_PROC_BACKWARD / 2;
            auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;
            WAIT_FLAG(MTE3, MTE2, event_id);

            LocalTensor<INPUT_T> fp16_temp_0_ub_tensor_addr_pingpong =
                ping_flag ? ub_backward.fp16_temp_0_ub_tensor_addr[MAX_LENG_PER_UB_PROC_BACKWARD]
                      : ub_backward.fp16_temp_0_ub_tensor_addr;
            LocalTensor<INPUT_T> fp16_temp_1_ub_tensor_addr_pingpong =
                ping_flag ? ub_backward.fp16_temp_1_ub_tensor_addr[MAX_LENG_PER_UB_PROC_BACKWARD]
                        : ub_backward.fp16_temp_1_ub_tensor_addr;
            LocalTensor<float> fp32_temp_0_ub_tensor_addr_pingpong =
                ping_flag ? ub_backward.fp32_temp_0_ub_tensor_addr[MAX_LENG_PER_UB_PROC_BACKWARD]
                        : ub_backward.fp32_temp_0_ub_tensor_addr;
            LocalTensor<float> fp32_temp_1_ub_tensor_addr_pingpong =
                ping_flag ? ub_backward.fp32_temp_1_ub_tensor_addr[MAX_LENG_PER_UB_PROC_BACKWARD]
                        : ub_backward.fp32_temp_1_ub_tensor_addr;
            // copy dp to ub
            gm_to_ub<ArchType::ASCEND_V220, float>(
                    fp32_temp_1_ub_tensor_addr_pingpong,
                    gm_dP_tensor[total_offset_s + dp_offset],
                    0,
                    1,            // 1
                    MAX_LENG_PER_UB_PROC_BACKWARD * 4 / 32,
                    0,
                    0);

            SET_FLAG(MTE2, V, event_id);
            WAIT_FLAG(MTE2, V, event_id);

            // (dP - D)manm
            auto rowsum_offset = ping_flag ? 32 * 8 : 0;
            sub_v<ArchType::ASCEND_V220, float>(
                fp32_temp_1_ub_tensor_addr_pingpong,
                fp32_temp_1_ub_tensor_addr_pingpong,
                ub_backward.tensor_for_brcb_rowsum_fp32[256 * m],
                MAX_BLOCK_PER_ONE_PROC_BACKWARD,
                1,
                1,
                0,
                16,
                16,
                1);
            PIPE_BARRIER(V);
            sub_v<ArchType::ASCEND_V220, float>(
                fp32_temp_1_ub_tensor_addr_pingpong[64],
                fp32_temp_1_ub_tensor_addr_pingpong[64],
                ub_backward.tensor_for_brcb_rowsum_fp32[256 * m],
                MAX_BLOCK_PER_ONE_PROC_BACKWARD,
                1,
                1,
                0,
                16,
                16,
                1);
            PIPE_BARRIER(V);

            // P * (dP - D)
            mul_v<ArchType::ASCEND_V220, float>(
                fp32_temp_1_ub_tensor_addr_pingpong,
                fp32_temp_0_ub_tensor_addr_pingpong,
                fp32_temp_1_ub_tensor_addr_pingpong,
                MAX_BLOCK_PER_ONE_PROC_BACKWARD * 2,
                1,
                1,
                1,
                8,
                8,
                8);
            PIPE_BARRIER(V);

            if constexpr (IF_BF16) {
                convr_v<ArchType::ASCEND_V220, float, INPUT_T>(
                fp16_temp_1_ub_tensor_addr_pingpong,
                fp32_temp_1_ub_tensor_addr_pingpong,
                MAX_BLOCK_PER_ONE_PROC_BACKWARD * 2,
                1,
                1,
                4,
                8);
                PIPE_BARRIER(V);
            } else {
                conv_v<ArchType::ASCEND_V220, float, INPUT_T>(
                fp16_temp_1_ub_tensor_addr_pingpong,
                fp32_temp_1_ub_tensor_addr_pingpong,
                MAX_BLOCK_PER_ONE_PROC_BACKWARD * 2,
                1,
                1,
                4,
                8);
                PIPE_BARRIER(V);
            }


            SET_FLAG(V, MTE3, event_id);
            WAIT_FLAG(V, MTE3, event_id);

            const auto &tmpGmTensor = gm_dP_tensor[total_offset_s + p_offset];
            GlobalTensor<INPUT_T> &tmp_gm_dP_tensor = (GlobalTensor<INPUT_T>&)(tmpGmTensor);
            ub_to_gm<ArchType::ASCEND_V220, INPUT_T>(
                tmp_gm_dP_tensor,
                fp16_temp_1_ub_tensor_addr_pingpong,
                0,
                1,
                MAX_LENG_PER_UB_PROC_BACKWARD * 2 / 32,
                0,
                0);

            SET_FLAG(MTE3, MTE2, event_id);
            ping_flag = 1 - ping_flag;
        }
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
    // }
}
}  // namespace VEC_BACKWARD_FP32_OP
#endif

#endif