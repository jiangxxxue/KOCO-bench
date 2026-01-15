/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */                                                                                                                    \
#ifndef __TRANSPOSE_CUSTOM_H__
#define __TRANSPOSE_CUSTOM_H__

#define USE_ASCENDC

#include "ppmatmul_const.h"
#include "kernels/utils/kernel/common.h"
#include "kernels/utils/kernel/common_func.h"
#include "kernels/utils/kernel/hardware.h"
#include "kernels/utils/kernel/simd.h"

#include "kernels/utils/kernel/iterator.h"
#include "kernels/utils/kernel/mma.h"
#include "kernels/utils/kernel/utils.h"

using namespace AscendC;

#ifdef __DAV_C220_VEC__

template <typename TYPE, bool IF_BF16> class TransposeCustom {
public:
    __aicore__ inline TransposeCustom() {}
    __aicore__ inline void Init(__gm__ uint8_t *__restrict__ gm_in_tensor,  // input
                                __gm__ uint8_t *__restrict__ gm_out_tensor, // output
                                int32_t batch_size, int32_t head_num, int32_t seq_size, int32_t head_dim,
                                int32_t transpose_type);
    __aicore__ inline void Run();
    __aicore__ inline void transposeSBHToBNSD1(int32_t vector_num, int32_t cur_vector_idx, int S, int B, int N, int D);
    __aicore__ inline void transposeSBHToBNSD2(int32_t vector_num, int32_t cur_vector_idx, int S, int B, int N, int D);

private:
    GlobalTensor<TYPE> gm_in_tensor;
    GlobalTensor<TYPE> gm_out_tensor;
    LocalTensor<TYPE> ub_tensor;
    int32_t batch_size;     // B
    int32_t head_num;       // N
    int32_t seq_size;       // S
    int32_t head_dim;       // D
    int32_t transpose_type; // shape类型转换  0：SBH->BNSD
    AsdopsBuffer<ArchType::ASCEND_V220> calBuf;
};

template <typename TYPE, bool IF_BF16>
__aicore__ inline void TransposeCustom<TYPE, IF_BF16>::Init(__gm__ uint8_t *__restrict__ gm_in,  // input
                                                            __gm__ uint8_t *__restrict__ gm_out, // output
                                                            int32_t batch_size, int32_t head_num, int32_t seq_size,
                                                            int32_t head_dim, int32_t transpose_type)
{
    gm_in_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ __bf16 *>(gm_in));
    gm_out_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ __bf16 *>(gm_out));
    ub_tensor = calBuf.GetBuffer<BufferType::ASCEND_UB, TYPE>(0);
    this->batch_size = batch_size;         // B
    this->head_num = head_num;             // N
    this->seq_size = seq_size;             // S
    this->head_dim = head_dim;             // D
    this->transpose_type = transpose_type; // shape类型转换  0：SBH->BNSD  1:BSH->BNSD  2:BNSD->SBH 3:BNSD->BSH
}

template <typename TYPE, bool IF_BF16> __aicore__ inline void TransposeCustom<TYPE, IF_BF16>::Run()
{
    int32_t vector_num = get_subblockdim() * get_block_num();                        // vector的数量
    int32_t cur_vector_idx = get_block_idx() * get_subblockdim() + get_subblockid(); // 当前编号
    int S = seq_size;
    int B = batch_size;
    int N = head_num;
    int D = head_dim;
    int ub_max = 96 * 1024 / sizeof(TYPE); // 48 * 1024  一半拷贝进来，一半排序
    int segmentation = ub_max / D;
    // SBH  -> BNSD
    if (transpose_type == 0) {
        if (B * N > segmentation) {
            transposeSBHToBNSD2(vector_num, cur_vector_idx, S, B, N, D);
        } else {
            transposeSBHToBNSD1(vector_num, cur_vector_idx, S, B, N, D);
        }
    }
}

// 典型场景测试  S<=8192且B*N <=192
template <typename TYPE, bool IF_BF16>
__aicore__ inline void TransposeCustom<TYPE, IF_BF16>::transposeSBHToBNSD1(int32_t vector_num, int32_t cur_vector_idx,
                                                                           int S, int B, int N, int D)
{
    int H = N * D;
    // UB 192K
    // UB 192K   5_16_8192_128
    int ub_size = 96 * 1024 / sizeof(TYPE); // 48 * 1024  一半拷贝进来，一半排序
    int ub_max = 96 * 1024 / sizeof(TYPE);  // 48 * 1024  一半拷贝进来，一半排序
    int length = ub_max / (B * H);          // 3 每次处理3行
    if (length >= 1) {
        ub_max = length * (B * H); // ub_max设置为B*H的整数倍
    }
    int seq_pre_core_length = S / (vector_num * length);
    int seq_pre_core = seq_pre_core_length * length;                             // 168
    int tail = S - seq_pre_core * vector_num;                                    // 8192 - 168*48 = 128
    int tail_core = tail / length;                                               // 128/3=42余数为2
    int inOffset = (seq_pre_core * cur_vector_idx + tail_core * length) * B * H; //
    int outOffset = (seq_pre_core * cur_vector_idx + tail_core * length) * D;
    if (cur_vector_idx < tail_core) {
        seq_pre_core += length; // 168+3=171
        inOffset = seq_pre_core * cur_vector_idx * B * H;
        outOffset = seq_pre_core * cur_vector_idx * D;
    }

    int times = seq_pre_core / length; // 56或者57

    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    for (int i = 0; i < times; i++) {
        
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

        AscendC::DataCopy(ub_tensor, gm_in_tensor[inOffset + ub_max * i],
                          AscendC::DataCopyParams(1,
                                                  ub_max / 16, // 一个block=32B N*2/32
                                                  0, 0));

        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);

        for (int j = 0; j < length; j++) {
            AscendC::DataCopy(ub_tensor[ub_size + D * j], ub_tensor[B * H * j],
                              AscendC::DataCopyParams(B * H / D, D / 16, 0, (length - 1) * D / 16));
        }

        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

        AscendC::DataCopyPad(gm_out_tensor[outOffset + length * i * D], ub_tensor[ub_size],
                             AscendC::DataCopyExtParams(B * H / D, D * length * 2, 0, (S - length) * D * 2, 0));
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    // 处理不能整除的数据
    int remain = tail - tail_core * length; // 128-42 * 3 =2
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    if (remain > 0 && cur_vector_idx == (vector_num - 1)) {
        inOffset = (seq_pre_core * cur_vector_idx + tail_core * length + seq_pre_core) * B * H; // seq_pre_core=168
        outOffset = (seq_pre_core * cur_vector_idx + tail_core * length + seq_pre_core) * D;

        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

        AscendC::DataCopy(ub_tensor, gm_in_tensor[inOffset], AscendC::DataCopyParams(1, remain * (B * H) / 16, 0, 0));

        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);

        for (int j = 0; j < remain; j++) {
            AscendC::DataCopy(ub_tensor[ub_size + D * j], ub_tensor[B * H * j],
                              AscendC::DataCopyParams(B * H / D, D / 16, 0, (remain - 1) * D / 16));
        }

        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(gm_out_tensor[outOffset], ub_tensor[ub_size],
                             AscendC::DataCopyExtParams(B * H / D, D * remain * 2, 0, (S - remain) * D * 2, 0));
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}

// 分段拷入，连续拷出  SBH  -> BNSD  应用场景  S>8192或B*N > 192
template <typename TYPE, bool IF_BF16>
__aicore__ inline void TransposeCustom<TYPE, IF_BF16>::transposeSBHToBNSD2(int32_t vector_num, int32_t cur_vector_idx,
                                                                           int S, int B, int N, int D)
{
    int H = N * D;
    // UB 192K  10_256_32_128
    int ub_max = 96 * 1024 / sizeof(TYPE);    // 48 * 1024  一半拷贝进来，一半排序
    int seq_pre_core = S / vector_num;        // 每个核平均处理个数5
    int tail = S - seq_pre_core * vector_num; // 16
    int inOffset = (seq_pre_core * cur_vector_idx + tail) * B * H;
    int outOffset = (seq_pre_core * cur_vector_idx + tail) * D;
    if (cur_vector_idx < tail) {
        seq_pre_core += 1;
        inOffset = seq_pre_core * cur_vector_idx * B * H;
        outOffset = seq_pre_core * cur_vector_idx * D;
    }
    for (int batchSize = 0; batchSize < B; batchSize++) {
        for (int headSize = 0; headSize < N; headSize++) {
            int times = seq_pre_core * D / ub_max;
            if (times > 0) {

                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);

                for (int i = 0; i < times; i++) {
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    AscendC::DataCopy(
                        ub_tensor, gm_in_tensor[inOffset + ub_max / D * i * B * H + headSize * D + batchSize * N * D],
                        AscendC::DataCopyParams(ub_max / D, D / 16, (B * N - 1) * D / 16, 0));

                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);

                    AscendC::DataCopy(ub_tensor[ub_max], ub_tensor, AscendC::DataCopyParams(1, ub_max / 16, 0, 0));

                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

                    AscendC::DataCopy(gm_out_tensor[outOffset + ub_max * i + headSize * S * D + batchSize * N * S * D],
                                      ub_tensor[ub_max], AscendC::DataCopyParams(1, ub_max / 16, 0, 0));
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                }
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
            // 处理尾块
            int tail_block = seq_pre_core * D - ub_max * times; // 5 * 128 = 640
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            if (tail_block > 0) {
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                AscendC::DataCopy(
                    ub_tensor, gm_in_tensor[inOffset + ub_max / D * B * H * times + headSize * D + batchSize * N * D],
                    AscendC::DataCopyParams(tail_block / D, D / 16, (B * N - 1) * D / 16, 0));

                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);

                AscendC::DataCopy(ub_tensor[ub_max], ub_tensor, AscendC::DataCopyParams(1, tail_block / 16, 0, 0));

                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

                AscendC::DataCopy(gm_out_tensor[outOffset + ub_max * times + headSize * S * D + batchSize * N * S * D],
                                  ub_tensor[ub_max], AscendC::DataCopyParams(1, tail_block / 16, 0, 0));
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }
}

#endif
#endif