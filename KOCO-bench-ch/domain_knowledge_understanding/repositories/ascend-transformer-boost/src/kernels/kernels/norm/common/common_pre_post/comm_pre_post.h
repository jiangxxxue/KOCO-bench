/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PRE_POSTNORM_COMMMON_SHORT_H
#define PRE_POSTNORM_COMMMON_SHORT_H

#include "kernel_operator.h"
#include "kernels/norm/rmsnormforward/op_kernel/rms_norm_base.h"
#include "kernels/utils/kernel/kernel_utils.h"
const constexpr uint32_t NUM_TWO = 2;
const constexpr uint32_t NUM_THREE = 3;
const constexpr uint32_t NUM_FOUR = 4;
const constexpr uint32_t BLOCK_BYTE = 32;
const constexpr uint32_t FP32_PER_REPEAT = 64;
const constexpr uint32_t FP16_PER_REPEAT = 16;

template<typename T>
__aicore__ inline  void BiasIn(AscendC::LocalTensor<T>& fp16_x,
                               AscendC::LocalTensor<T>& fp16_bias,
                               AscendC::LocalTensor<float>& fp32_bias,
                               AscendC::GlobalTensor<T>& gm_bias,
                               uint32_t countAlign,
                               uint32_t gmOffset = 0)
{
    // do copyIn and Cast
    DataCopy(fp16_bias, gm_bias[gmOffset], countAlign);
    AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    Cast(fp32_bias, fp16_bias, AscendC::RoundMode::CAST_NONE, countAlign);
    AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
}

template<typename T>
__aicore__ inline  void CopyInXResIn(AscendC::LocalTensor<T>& fp16_x,
                                     AscendC::LocalTensor<T>& fp16_res_in,
                                     AscendC::GlobalTensor<T>& gm_x_,
                                     AscendC::GlobalTensor<T>& gm_res_in_,
                                     uint64_t& gmOffset, uint32_t& numel)
{
    DataCopy(fp16_x, gm_x_[gmOffset], numel);
    DataCopy(fp16_res_in, gm_res_in_[gmOffset], numel); // res_in
}

template<typename T>
__aicore__ inline  void AddResBiasAndCast(AscendC::LocalTensor<T>& fp16_x,
                                          AscendC::LocalTensor<T>& fp16_res_in,
                                          AscendC::LocalTensor<T>& fp16_bias,
                                          AscendC::LocalTensor<float>& fp32_xy,
                                          AscendC::LocalTensor<float>& buf,
                                          AscendC::LocalTensor<float>& buf_bias,
                                          uint32_t count)
{
    Cast(fp32_xy, fp16_x, AscendC::RoundMode::CAST_NONE, count);
    Cast(buf, fp16_res_in, AscendC::RoundMode::CAST_NONE, count); // res_in
    AscendC::PipeBarrier<PIPE_V>();
    Add(fp32_xy, fp32_xy, buf, count);
    AscendC::PipeBarrier<PIPE_V>();
    Add(fp32_xy, fp32_xy, buf_bias, count);
    AscendC::PipeBarrier<PIPE_V>();
}

template<typename T>
__aicore__ inline  void AddResAndCast(AscendC::LocalTensor<T>& fp16_x,
                                      AscendC::LocalTensor<T>& fp16_res_in,
                                      AscendC::LocalTensor<float>& fp32_xy,
                                      AscendC::LocalTensor<float>& buf,
                                      uint32_t count)
{
    Cast(fp32_xy, fp16_x, AscendC::RoundMode::CAST_NONE, count);
    Cast(buf, fp16_res_in, AscendC::RoundMode::CAST_NONE, count); // res_in
    AscendC::PipeBarrier<PIPE_V>();
    Add(fp32_xy, fp32_xy, buf, count);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline  void FigureOutNorm(AscendC::LocalTensor<float>& sqx,
                                      AscendC::LocalTensor<float>& fp32_xy,
                                      AscendC::LocalTensor<float>& fp32_reduce_workspace,
                                      float& avgFactor_, float& epsilon_,
                                      uint32_t& count, uint32_t& countAlign)
{
    // x / sqrt(sum(x^2) * 1 / len(sum) + eps) --> norm operation
    Mul(sqx, fp32_xy, fp32_xy, countAlign);
    AscendC::PipeBarrier<PIPE_V>();
    ReduceSumCustom(sqx, sqx, fp32_reduce_workspace, count);

    Muls(sqx, sqx, avgFactor_, 1);
    AscendC::PipeBarrier<PIPE_V>();
    Adds(sqx, sqx, epsilon_, 1);
    AscendC::PipeBarrier<PIPE_V>();

    Sqrt(sqx, sqx, 1);
    AscendC::SetFlag<HardEvent::V_S>(EVENT_ID2);

    AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID2);
    float factor = 1 / sqx.GetValue(0);
    Muls(fp32_xy, fp32_xy, factor, countAlign);
    AscendC::PipeBarrier<PIPE_V>();
}

template<typename T>
__aicore__ inline  void MultiplyGamma(AscendC::LocalTensor<T>& fp16_gamma,
                                      AscendC::LocalTensor<float>& sqx,
                                      AscendC::LocalTensor<float>& fp32_xy,
                                      AscendC::LocalTensor<T>& out_buf,
                                      uint32_t numCol_, uint32_t& numColAlignFp32,
                                      uint32_t& numColAlignFp16, uint32_t& precisionMode_)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        Cast(sqx, fp16_gamma, AscendC::RoundMode::CAST_NONE, numColAlignFp16); // g
        AscendC::PipeBarrier<PIPE_V>();
        Mul(fp32_xy, fp32_xy, sqx, numColAlignFp32);
        AscendC::PipeBarrier<PIPE_V>();
        CastFrom32To16(out_buf, fp32_xy, numCol_);
    }
#endif
    if constexpr (std::is_same<T, half>::value) {
        if (precisionMode_ == 1) {
            AscendC::PipeBarrier<PIPE_V>();
            CastFrom32To16(out_buf, fp32_xy, numCol_);
            AscendC::PipeBarrier<PIPE_V>();
            Mul(out_buf, out_buf, fp16_gamma, numColAlignFp16);
        } else {
            AscendC::PipeBarrier<PIPE_V>();
            Cast(sqx, fp16_gamma, AscendC::RoundMode::CAST_NONE, numColAlignFp16); // g
            AscendC::PipeBarrier<PIPE_V>();
            Mul(fp32_xy, sqx, fp32_xy, numColAlignFp32);
            AscendC::PipeBarrier<PIPE_V>();
            CastFrom32To16(out_buf, fp32_xy, numCol_);
        }
    }
}

template<typename T>
__aicore__ inline void CopyInG(AscendC::LocalTensor<T>& fp16_gamma,
                               AscendC::GlobalTensor<T>& gm_g_,
                               uint32_t offset,
                               uint32_t& numelAlignFp16)
{
    DataCopy(fp16_gamma, gm_g_[offset], numelAlignFp16);
}

__aicore__ inline  void FigureOutNorm(AscendC::LocalTensor<float>& sqx,
                                      AscendC::LocalTensor<float>& fp32_xy,
                                      AscendC::LocalTensor<float>& fp32_workspace,
                                      float& avgF, float& eps, uint32_t& count,
                                      uint32_t& repeatTimes, uint32_t& alignOffset, uint32_t& tailNum)
{
    Mul(sqx, fp32_xy, fp32_xy, count);
    AscendC::PipeBarrier<PIPE_V>();
    ReduceSumCustom(sqx, sqx, fp32_workspace, count);

    Muls(sqx, sqx, avgF, 1);
    AscendC::PipeBarrier<PIPE_V>();

    Adds(sqx, sqx, eps, 1);
    AscendC::PipeBarrier<PIPE_V>();

    Sqrt(sqx, sqx, 1);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Brcb(fp32_workspace, sqx, 1, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Div(fp32_xy, fp32_xy, fp32_workspace, FP32_PER_REPEAT, repeatTimes, {1, 1, 0, 8, 8, 0});
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Div(fp32_xy[alignOffset], fp32_xy[alignOffset], fp32_workspace, tailNum, 1, {1, 1, 0, 8, 8, 0});
    AscendC::PipeBarrier<PIPE_V>();
}

template<typename T>
__aicore__ inline  void MultiplyGamma(AscendC::LocalTensor<T>& fp16_gamma,
                                      AscendC::LocalTensor<float>& fp32_tmp,
                                      AscendC::LocalTensor<float>& fp32_xy,
                                      AscendC::LocalTensor<float>& fp32_y,
                                      uint32_t numel)
{
    Cast(fp32_tmp, fp16_gamma, AscendC::RoundMode::CAST_NONE, numel); // g
    AscendC::PipeBarrier<PIPE_V>();
    Mul(fp32_y, fp32_xy, fp32_tmp, numel);
    AscendC::PipeBarrier<PIPE_V>();
}

template<typename T>
__aicore__ inline  void AddResAndBias(AscendC::LocalTensor<T>& fp16_x,
                                      AscendC::LocalTensor<T>& fp16_res_in,
                                      AscendC::LocalTensor<T>& fp16_bias,
                                      AscendC::LocalTensor<float>& fp32_xy,
                                      AscendC::LocalTensor<float>& buf,
                                      uint32_t count)
{
    Cast(fp32_xy, fp16_x, AscendC::RoundMode::CAST_NONE, count);
    Cast(buf, fp16_res_in, AscendC::RoundMode::CAST_NONE, count); // res_in
    AscendC::PipeBarrier<PIPE_V>();
    Add(fp32_xy, fp32_xy, buf, count);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(buf, fp16_bias, AscendC::RoundMode::CAST_NONE, count);
    AscendC::PipeBarrier<PIPE_V>();
    Add(fp32_xy, fp32_xy, buf, count);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline  void FigureOutNorm(AscendC::LocalTensor<float>& sqx,
                                      AscendC::LocalTensor<float>& brcb_buf,
                                      AscendC::LocalTensor<float>& fp32_xy,
                                      float& avgF, float& eps, uint32_t& count, uint32_t& countAlign,
                                      uint32_t& repeatTimes, uint32_t& alignOffset, uint32_t& tailNum)
{
    Mul(sqx, fp32_xy, fp32_xy, countAlign);
    AscendC::PipeBarrier<PIPE_V>();
    ReduceSumCustom(sqx, sqx, brcb_buf, count);
 
    Muls(sqx, sqx, avgF, 1);
    AscendC::PipeBarrier<PIPE_V>();
 
    Adds(sqx, sqx, eps, 1);
    AscendC::PipeBarrier<PIPE_V>();
 
    Sqrt(sqx, sqx, 1);
    AscendC::PipeBarrier<PIPE_V>();
 
    AscendC::Brcb(brcb_buf, sqx, 1, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
 
    AscendC::Div(fp32_xy, fp32_xy, brcb_buf, FP32_PER_REPEAT, repeatTimes, {1, 1, 0, 8, 8, 0});
    AscendC::PipeBarrier<PIPE_V>();
 
    AscendC::Div(fp32_xy[alignOffset], fp32_xy[alignOffset], brcb_buf, tailNum, 1, {1, 1, 0, 8, 8, 0});
    AscendC::PipeBarrier<PIPE_V>();
}

#endif