/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __TRANSPOSE_GRAD_H__
#define __TRANSPOSE_GRAD_H__

#include "ppmatmul_const_grad.h"
#include "TransposeCustom.h"

#ifdef __DAV_C220_VEC__

template <typename TYPE, bool IF_BF16> class TransposeGrad {
public:
    __aicore__ TransposeGrad() {}
    __aicore__ void Init(__gm__ TYPE *__restrict__ gm_Q, __gm__ TYPE *__restrict__ gm_K, __gm__ TYPE *__restrict__ gm_V,
                         __gm__ TYPE *__restrict__ gm_dO, __gm__ TYPE *__restrict__ gm_O,
                         __gm__ TYPE *__restrict__ gm_Q_trans, __gm__ TYPE *__restrict__ gm_K_trans,
                         __gm__ TYPE *__restrict__ gm_V_trans, __gm__ TYPE *__restrict__ gm_dO_trans,
                         __gm__ TYPE *__restrict__ gm_O_trans, int32_t batch_size, int32_t head_num, int32_t q_size,
                         int32_t k_size, int32_t inputLayout, int32_t gqa_head);
    __aicore__ void Run();

private:
    __gm__ TYPE *__restrict__ gm_Q{nullptr};
    __gm__ TYPE *__restrict__ gm_K{nullptr};
    __gm__ TYPE *__restrict__ gm_V{nullptr};
    __gm__ TYPE *__restrict__ gm_dO{nullptr};
    __gm__ TYPE *__restrict__ gm_O{nullptr};
    __gm__ TYPE *__restrict__ gm_K_trans{nullptr};
    __gm__ TYPE *__restrict__ gm_Q_trans{nullptr};
    __gm__ TYPE *__restrict__ gm_V_trans{nullptr};
    __gm__ TYPE *__restrict__ gm_dO_trans{nullptr};
    __gm__ TYPE *__restrict__ gm_O_trans{nullptr};
    int32_t batch_size;  // B
    int32_t head_num;    // N
    int32_t q_size;      // S
    int32_t k_size;      // D
    int32_t inputLayout; // shape类型转换  0：SBH->BNSD
    int32_t gqa_head;
};

template <typename TYPE, bool IF_BF16>
__aicore__ void TransposeGrad<TYPE, IF_BF16>::Init(
    __gm__ TYPE *__restrict__ gm_Q, __gm__ TYPE *__restrict__ gm_K, __gm__ TYPE *__restrict__ gm_V,
    __gm__ TYPE *__restrict__ gm_dO, __gm__ TYPE *__restrict__ gm_O, __gm__ TYPE *__restrict__ gm_Q_trans,
    __gm__ TYPE *__restrict__ gm_K_trans, __gm__ TYPE *__restrict__ gm_V_trans, __gm__ TYPE *__restrict__ gm_dO_trans,
    __gm__ TYPE *__restrict__ gm_O_trans, int32_t batch_size, int32_t head_num, int32_t q_size, int32_t k_size,
    int32_t inputLayout, int32_t gqa_head)
{
    this->gm_Q = gm_Q;
    this->gm_K = gm_K;
    this->gm_V = gm_V;
    this->gm_dO = gm_dO;
    this->gm_O = gm_O;
    this->gm_Q_trans = gm_Q_trans;
    this->gm_K_trans = gm_K_trans;
    this->gm_V_trans = gm_V_trans;
    this->gm_dO_trans = gm_dO_trans;
    this->gm_O_trans = gm_O_trans;
    this->batch_size = batch_size;   // B
    this->head_num = head_num;       // N
    this->q_size = q_size;           // S
    this->k_size = k_size;           // D
    this->inputLayout = inputLayout; // shape类型转换  0：SBH->BNSD  1:BSH->BNSD  2:BNSD->SBH 3:BNSD->BSH
    this->gqa_head = gqa_head;
}

template <typename TYPE, bool IF_BF16> __aicore__ void TransposeGrad<TYPE, IF_BF16>::Run()
{
    if (inputLayout != 3) {
        TransposeCustom<TYPE, true> op_in;
        op_in.Init((__gm__ uint8_t *)gm_Q, (__gm__ uint8_t *)gm_Q_trans, batch_size, head_num, q_size, 192,
                   inputLayout);
        op_in.Run();
        op_in.Init((__gm__ uint8_t *)gm_K, (__gm__ uint8_t *)gm_K_trans, batch_size, gqa_head, k_size, 256,
                   inputLayout);
        op_in.Run();
        op_in.Init((__gm__ uint8_t *)gm_V, (__gm__ uint8_t *)gm_V_trans, batch_size, gqa_head, k_size, 128,
                   inputLayout);
        op_in.Run();
        op_in.Init((__gm__ uint8_t *)gm_dO, (__gm__ uint8_t *)gm_dO_trans, batch_size, head_num, q_size, 128,
                   inputLayout);
        op_in.Run();
        op_in.Init((__gm__ uint8_t *)gm_O, (__gm__ uint8_t *)gm_O_trans, batch_size, head_num, q_size, 128,
                   inputLayout);
        op_in.Run();
        uint64_t mode = 0;
        uint64_t config = 1 | (mode << 4) | (AIVFLAGID << 8);
        ffts_cross_core_sync(PIPE_MTE3, config);
        wait_flag_dev(AIVFLAGID);
        mode = 2;
        config = 1 | (mode << 4) | (AIV2AICFLAGID << 8);
        ffts_cross_core_sync(PIPE_MTE3, config);
    }
}

#endif
#endif