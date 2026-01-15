/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __TRANSPOSE_OUT_H__
#define __TRANSPOSE_OUT_H__

#define USE_ASCENDC 1
#include "ppmatmul_const_grad.h"
#include "kernels/utils/kernel/common.h"
#include "kernels/utils/kernel/common_func.h"
#include "kernels/utils/kernel/hardware.h"
#include "kernels/utils/kernel/simd.h"
#include "kernels/utils/kernel/iterator.h"
#include "kernels/utils/kernel/mma.h"
#include "kernels/utils/kernel/utils.h"

using namespace AscendC;

#ifdef __DAV_C220_VEC__

template <typename TYPE, bool IF_BF16> class TransposeWithDtype {
public:
    __aicore__ inline TransposeWithDtype() {}
    __aicore__ inline void Init(__gm__ float *__restrict__ gm_in,     // input
                                __gm__ float *__restrict__ gm_rowsum, // rowsum
                                __gm__ TYPE *__restrict__ gm_out,     // output
                                int32_t batch_size, int32_t head_num, int32_t seq_size, int32_t head_dim,
                                int32_t transpose_type, int32_t is_forward, bool needMuls, float scale);
    __aicore__ inline void Run();

private:
    __aicore__ __inline__ void SBHForward();
    __aicore__ __inline__ void SBHBackward();
    __aicore__ __inline__ void ComputeBackward(int32_t in_offset, int32_t out_offset, int32_t lenth, int32_t row);
    __aicore__ __inline__ void ComputeForward(int32_t in_offset, int32_t rowsum_offset, int32_t out_offset,
                                              int32_t lenth, int32_t row);

private:
    AsdopsBuffer<ArchType::ASCEND_V220> calBuf;
    LocalTensor<TYPE> cur_tensor;
    LocalTensor<float> cur_tensor_f32;
    LocalTensor<float> cur_tensor_out;
    LocalTensor<TYPE> tensor_transpose;
    LocalTensor<float> tensor_transpose_f32;
    LocalTensor<TYPE> tensor_rowsum_brcb;
    LocalTensor<float> tensor_rowsum_brcb_f32;
    LocalTensor<uint32_t> tensor_mask;
    GlobalTensor<float> gm_in_tensor;
    GlobalTensor<float> gm_rowsum_tensor;
    GlobalTensor<TYPE> gm_out_tensor;
    int32_t B;              // B
    int32_t N;              // N
    int32_t S;              // S
    int32_t D;              // D
    int32_t H;              // H
    int32_t transpose_type; // shape类型转换
    int32_t is_forward;     // 前向还是反向
    int32_t vector_num;     // vector的数量
    int32_t cur_vector_idx; // 当前编号
    bool needMuls = true;
    float scale;
    int UB_MAX = 192 * 1024 / sizeof(float) / 3;
};

template <typename TYPE, bool IF_BF16>
__aicore__ inline void TransposeWithDtype<TYPE, IF_BF16>::Init(__gm__ float *__restrict__ gm_in,     // input
                                                               __gm__ float *__restrict__ gm_rowsum, // rowsum
                                                               __gm__ TYPE *__restrict__ gm_out,     // output
                                                               int32_t batch_size, int32_t head_num, int32_t seq_size,
                                                               int32_t head_dim, int32_t transpose_type,
                                                               int32_t is_forward, bool needMuls, float scale)
{
    gm_in_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_in));
    gm_rowsum_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_rowsum));
    gm_out_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE *>(gm_out));

    cur_tensor = calBuf.GetBuffer<BufferType::ASCEND_UB, TYPE>(0);
    cur_tensor_f32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    cur_tensor_out = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    tensor_transpose = calBuf.GetBuffer<BufferType::ASCEND_UB, TYPE>(192 * 1024 / 3 * 2);
    tensor_transpose_f32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(192 * 1024 / 3 * 2);
    tensor_rowsum_brcb = calBuf.GetBuffer<BufferType::ASCEND_UB, TYPE>(192 * 1024 / 6 * 5);
    tensor_rowsum_brcb_f32 = calBuf.GetBuffer<BufferType::ASCEND_UB, float>(192 * 1024 / 6 * 5);
    tensor_mask = calBuf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(192 * 1024 / sizeof(TYPE) / 6 * 5);


    this->B = batch_size;
    this->N = head_num;
    this->S = seq_size;
    this->D = head_dim;
    this->H = head_num * head_dim;
    this->transpose_type = transpose_type;
    this->is_forward = is_forward;
    this->vector_num = get_block_num() * 2;
    this->cur_vector_idx = get_block_idx() * 2 + get_subblockid(); // 当前编号
    this->needMuls = needMuls;
    this->scale = scale;
}

template <typename TYPE, bool IF_BF16> __aicore__ inline void TransposeWithDtype<TYPE, IF_BF16>::Run()
{
    if (transpose_type == 0) {
        if (is_forward == 1) {
            SBHForward();
        } else {
            SBHBackward();
        }
    }
}

template <typename TYPE, bool IF_BF16> __aicore__ __inline__ void TransposeWithDtype<TYPE, IF_BF16>::SBHBackward()
{
    int length, times, remain, inOffset, outOffset = 0;
    int length_per_core = S / vector_num / 8 * 8;
    int tail_length = S - length_per_core * vector_num;
    int tail = tail_length / 8;

    if (cur_vector_idx < tail) {
        length_per_core += 8;
        inOffset = length_per_core * cur_vector_idx * D;
        outOffset = length_per_core * cur_vector_idx * B * N * D;
    } else {
        inOffset = (length_per_core * cur_vector_idx + tail_length) * D;
        outOffset = (length_per_core * cur_vector_idx + tail_length) * B * N * D;
    }

    if (UB_MAX >= B * N * D * length_per_core) {
        length = length_per_core;
        times = 1;
        remain = 0;
    } else {
        if (UB_MAX / (B * N * D) < 8) {
            // B * N > 16 走这个分支
            length = 8;
        } else {
            length = UB_MAX / (B * N * D) / 8 * 8;
        }
        times = length_per_core / length;
        remain = length_per_core % length;
    }

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    if (UB_MAX / (B * N * D) < 8) {
        auto row = (UB_MAX / D / length); // UB_MAX = 16384
        auto loopBN = (B * N) / row;
        auto remainBN = (B * N) % row; // 尾行
        for (int j = 0; j < loopBN; j++) {
            for (int i = 0; i < times; i++) {
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                ComputeBackward(inOffset + i * length * D + j * S * D * row,
                                outOffset + i * length * B * N * D + j * row * D,
                                // outOffset + i * length * D + j * S * D * row, // no transpose copy
                                length, row);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }
        if (remainBN > 0) {
            // 存在尾行时 要单独处理
            for (int i = 0; i < times; i++) {
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                ComputeBackward(inOffset + i * length * D + loopBN * S * D * row,
                                outOffset + i * length * B * N * D + loopBN * row * D,
                                // outOffset + i * length * D + j * S * D * remainBN, // no transpose copy
                                length, remainBN);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }
    } else {
        for (int i = 0; i < times; i++) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            ComputeBackward(inOffset + i * length * D, outOffset + i * length * B * N * D, length, 0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    if (remain > 0) {
        ComputeBackward(inOffset + times * length * D, outOffset + times * length * B * N * D, remain, 0);
    }
}

template <typename TYPE, bool IF_BF16>
__aicore__ __inline__ void TransposeWithDtype<TYPE, IF_BF16>::ComputeBackward(int32_t in_offset, int32_t out_offset,
                                                                              int32_t length, int32_t row)
{
    int32_t nBurst = 0;
    if (transpose_type == 0) {
        // SBH
        nBurst = (UB_MAX / (B * N * D) < 8) ? row : B * N;
    }

    AscendC::DataCopyPad(cur_tensor_f32, gm_in_tensor[in_offset],
                         AscendC::DataCopyExtParams(nBurst, D * length * 4, (S - length) * D * 4, 0, 0),
                         AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));

    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    if (needMuls) {
        Muls(cur_tensor_f32, cur_tensor_f32, this->scale, nBurst * D * length);
        PIPE_BARRIER(V);
    }

    int repeatTimes =
        nBurst * length * D / (256 / sizeof(float)); // B * N * length * D <= UB_MAX (16384)  即 repeatTimes <=256
    if constexpr (IF_BF16) {
        if (repeatTimes < 256) {
            LocalTensor<TYPE> cur_tensor_res = cur_tensor.template ReinterpretCast<TYPE>();
            convr_v<ArchType::ASCEND_V220, float, TYPE>(cur_tensor_res, cur_tensor_f32, repeatTimes, 1, 1, 4, 8);
        } else {
            int halfTimes = repeatTimes / 2;
            LocalTensor<TYPE> cur_tensor_res = cur_tensor.template ReinterpretCast<TYPE>();
            convr_v<ArchType::ASCEND_V220, float, TYPE>(cur_tensor_res, cur_tensor_f32, halfTimes, 1, 1, 4, 8);
            convr_v<ArchType::ASCEND_V220, float, TYPE>(cur_tensor_res[halfTimes * 64], cur_tensor_f32[halfTimes * 64],
                                                        halfTimes, 1, 1, 4, 8);
        }
    } else {
        if (repeatTimes < 256) {
            LocalTensor<TYPE> cur_tensor_res = cur_tensor.template ReinterpretCast<TYPE>();
            conv_v<ArchType::ASCEND_V220, float, TYPE>(cur_tensor_res, cur_tensor_f32, repeatTimes, 1, 1, 4, 8);
        } else {
            int halfTimes = repeatTimes / 2;
            LocalTensor<TYPE> cur_tensor_res = cur_tensor.template ReinterpretCast<TYPE>();
            conv_v<ArchType::ASCEND_V220, float, TYPE>(cur_tensor_res, cur_tensor_f32, halfTimes, 1, 1, 4, 8);
            conv_v<ArchType::ASCEND_V220, float, TYPE>(cur_tensor_res[halfTimes * 64], cur_tensor_f32[halfTimes * 64],
                                                       halfTimes, 1, 1, 4, 8);
        }
    }

    pipe_barrier(PIPE_V);
    if (nBurst >= length) {
        for (int j = 0; j < length; j++) {
            AscendC::DataCopy(tensor_transpose[j * nBurst * D], cur_tensor[j * D],
                              AscendC::DataCopyParams(nBurst, D / 16, (length - 1) * D / 16, 0));
        }
    } else {
        for (int j = 0; j < nBurst; j++) {
            AscendC::DataCopy(tensor_transpose[j * D], cur_tensor[j * D * length],
                              AscendC::DataCopyParams(length, D / 16, 0, (nBurst - 1) * D / 16));
        }
    }
    pipe_barrier(PIPE_V);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    if (transpose_type == 0) {
        if (UB_MAX / (B * N * D) < 8) {
            AscendC::DataCopyPad(gm_out_tensor[out_offset], tensor_transpose,
                                 AscendC::DataCopyExtParams(length, D * row * 2, 0, (B * N - row) * D * 2, 0));
        } else {
            int ub_size = B * N * D * length;
            AscendC::DataCopy(gm_out_tensor[out_offset], tensor_transpose,
                              AscendC::DataCopyParams(1, ub_size / 16, 0, 0));
        }
    } else if (transpose_type == 1) {
        if (UB_MAX / (N * D) < 8) {
            AscendC::DataCopyPad(gm_out_tensor[out_offset], tensor_transpose,
                                 AscendC::DataCopyExtParams(length, D * row * 2, 0, (N - row) * D * 2, 0));
        } else {
            int ub_size = N * D * length;
            AscendC::DataCopy(gm_out_tensor[out_offset], tensor_transpose,
                              AscendC::DataCopyParams(1, ub_size / 16, 0, 0));
        }
    }
}

template <typename TYPE, bool IF_BF16> __aicore__ __inline__ void TransposeWithDtype<TYPE, IF_BF16>::SBHForward()
{
    int length, times, remain, inOffset, rowsumOffset, outOffset = 0;
    int length_per_core = S / vector_num / 8 * 8;       // 16
    int tail_length = S - length_per_core * vector_num; // 256 288
    int tail = tail_length / 8;                         // 32

    if (cur_vector_idx < tail) {
        length_per_core += 8;
        inOffset = length_per_core * cur_vector_idx * D;
        rowsumOffset = length_per_core * cur_vector_idx;
        outOffset = length_per_core * cur_vector_idx * B * N * D;
    } else {
        inOffset = (length_per_core * cur_vector_idx + tail_length) * D;
        rowsumOffset = length_per_core * cur_vector_idx + tail_length;
        outOffset = (length_per_core * cur_vector_idx + tail_length) * B * N * D;
    }

    if (UB_MAX >= B * N * D * length_per_core) {
        length = length_per_core;
        times = 1;
        remain = 0;
    } else {
        if (UB_MAX / (B * N * D) < 8) {
            length = 8;
        } else {
            length = UB_MAX / (B * N * D) / 8 * 8; // 16
        }
        times = length_per_core / length;  // 1
        remain = length_per_core % length; // 8
    }

    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    if (UB_MAX / (B * N * D) < 8) {
        auto row = (UB_MAX / D / length); // UB_MAX = 16384
        auto loopBN = (B * N) / row;
        auto remainBN = (B * N) % row;
        for (int j = 0; j < loopBN; j++) {
            for (int i = 0; i < times; i++) {
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                ComputeForward(inOffset + i * length * D + j * S * D * row, rowsumOffset + i * length + j * S * row,
                               outOffset + i * length * B * N * D + j * row * D, length, row);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }
        // 尾行逻辑
        if (remainBN > 0) {
            for (int i = 0; i < times; i++) {
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                ComputeForward(inOffset + i * length * D + loopBN * S * D * row,
                               rowsumOffset + i * length + loopBN * S * row,
                               outOffset + i * length * B * N * D + loopBN * row * D, length, remainBN);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        }
    } else {
        for (int i = 0; i < times; i++) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            ComputeForward(inOffset + i * length * D, rowsumOffset + i * length, outOffset + i * length * B * N * D,
                           length, 0);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    if (remain > 0) {
        ComputeForward(inOffset + times * length * D, rowsumOffset + times * length,
                       outOffset + times * length * B * N * D, remain, 0);
    }
}

template <typename TYPE, bool IF_BF16>
__aicore__ __inline__ void TransposeWithDtype<TYPE, IF_BF16>::ComputeForward(int32_t in_offset, int32_t rowsum_offset,
                                                                             int32_t out_offset, int32_t length,
                                                                             int32_t row)
{
    int32_t nBurst = 0;
    if (transpose_type == 0) {
        // SBH
        nBurst = (UB_MAX / (B * N * D) < 8) ? row : B * N;
    }
    AscendC::DataCopyPad(cur_tensor_out, gm_in_tensor[in_offset],
                         AscendC::DataCopyExtParams(nBurst, D * length * 4, (S - length) * D * 4, 0, 0),
                         AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));

    AscendC::DataCopyPad(
        tensor_rowsum_brcb_f32, gm_rowsum_tensor[rowsum_offset],
        AscendC::DataCopyExtParams(nBurst, length * sizeof(float), (S - length) * 1 * sizeof(float), 0, 0),
        AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));

    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    auto lines_per_loop = nBurst * length;
    // 展开D * length个数据
    brcb_v<ArchType::ASCEND_V220, uint32_t>(tensor_transpose.template ReinterpretCast<uint32_t>(),
                                            tensor_rowsum_brcb.template ReinterpretCast<uint32_t>(), 1, 8,
                                            lines_per_loop);
    pipe_barrier(PIPE_V);
    // D个FP32 需要两个repeat； 因此，需要先算前半段，再算后半段
    AscendC::Div<float, false>(cur_tensor_out, cur_tensor_out, tensor_transpose_f32, (uint64_t)0, lines_per_loop,
                               AscendC::BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    pipe_barrier(PIPE_V);
    AscendC::Div<float, false>(cur_tensor_out[HEAD_DIM / 2], cur_tensor_out[HEAD_DIM / 2], tensor_transpose_f32,
                               (uint64_t)0, lines_per_loop, AscendC::BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    pipe_barrier(PIPE_V);

    int repeatTimes =
        nBurst * length * D / (256 / sizeof(float)); // B * N * length * D <= UB_MAX (16384)  即 repeatTimes <=256
    if constexpr (IF_BF16) {
        if (repeatTimes < 256) {
            LocalTensor<TYPE> cur_tensor_res = cur_tensor.template ReinterpretCast<TYPE>();
            convr_v<ArchType::ASCEND_V220, float, TYPE>(cur_tensor_res, cur_tensor_f32, repeatTimes, 1, 1, 4, 8);
        } else {
            int halfTimes = repeatTimes / 2;
            LocalTensor<TYPE> cur_tensor_res = cur_tensor.template ReinterpretCast<TYPE>();
            convr_v<ArchType::ASCEND_V220, float, TYPE>(cur_tensor_res, cur_tensor_f32, halfTimes, 1, 1, 4, 8);
            convr_v<ArchType::ASCEND_V220, float, TYPE>(cur_tensor_res[halfTimes * 64], cur_tensor_f32[halfTimes * 64],
                                                        halfTimes, 1, 1, 4, 8);
        }
    } else {
        if (repeatTimes < 256) {
            LocalTensor<TYPE> cur_tensor_res = cur_tensor.template ReinterpretCast<TYPE>();
            conv_v<ArchType::ASCEND_V220, float, TYPE>(cur_tensor_res, cur_tensor_f32, repeatTimes, 1, 1, 4, 8);
        } else {
            int halfTimes = repeatTimes / 2;
            LocalTensor<TYPE> cur_tensor_res = cur_tensor.template ReinterpretCast<TYPE>();
            conv_v<ArchType::ASCEND_V220, float, TYPE>(cur_tensor_res, cur_tensor_f32, halfTimes, 1, 1, 4, 8);
            conv_v<ArchType::ASCEND_V220, float, TYPE>(cur_tensor_res[halfTimes * 64], cur_tensor_f32[halfTimes * 64],
                                                       halfTimes, 1, 1, 4, 8);
        }
    }
    pipe_barrier(PIPE_V);

    if (nBurst >= length) {
        for (int j = 0; j < length; j++) {
            AscendC::DataCopy(tensor_transpose[j * nBurst * D], cur_tensor[j * D],
                              AscendC::DataCopyParams(nBurst, D / 16, (length - 1) * D / 16, 0));
        }
    } else {
        for (int j = 0; j < nBurst; j++) {
            AscendC::DataCopy(tensor_transpose[j * D], cur_tensor[j * D * length],
                              AscendC::DataCopyParams(length, D / 16, 0, (nBurst - 1) * D / 16));
        }
    }
    pipe_barrier(PIPE_V);

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    if (transpose_type == 0) {
        if (UB_MAX / (B * N * D) < 8) {
            AscendC::DataCopyPad(gm_out_tensor[out_offset], tensor_transpose,
                                 AscendC::DataCopyExtParams(length, D * row * 2, 0, (B * N - row) * D * 2, 0));
        } else {
            int ub_size = B * N * D * length;
            AscendC::DataCopy(gm_out_tensor[out_offset], tensor_transpose,
                              AscendC::DataCopyParams(1, ub_size / 16, 0, 0));
        }
    } else if (transpose_type == 1) {
        if (UB_MAX / (N * D) < 8) {
            AscendC::DataCopyPad(gm_out_tensor[out_offset], tensor_transpose,
                                 AscendC::DataCopyExtParams(length, D * row * 2, 0, (N - row) * D * 2, 0));
        } else {
            int ub_size = N * D * length;
            AscendC::DataCopy(gm_out_tensor[out_offset], tensor_transpose,
                              AscendC::DataCopyParams(1, ub_size / 16, 0, 0));
        }
    }
}

#endif
#endif