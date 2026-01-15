/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ASCEND_OPS_FFN_GELU_H
#define ASCEND_OPS_FFN_GELU_H

#include "kernel_operator.h"
#include "ffn_common.h"

namespace FFN {
template <typename T>
__aicore__ inline void FastGeluV2ClipParams(const AscendC::LocalTensor<T>& tempTensorA,
    const AscendC::LocalTensor<T>& srcLocal, const AscendC::GeluParams<T>& params)
{
    const T coefficientsA = -0.1444;
    const T coefficientsB = -1.769;
    const T coefficientsBInv = 1.769;
    const T coefficientsC = 0.7071;
    const T coefficientsD = 0.5;

    const AscendC::UnaryRepeatParams unaryParams;
    const AscendC::BinaryRepeatParams binaryParams;

    AscendC::Muls<T, false>(
        tempTensorA, srcLocal, coefficientsC, AscendC::MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Abs<T, false>(
        tempTensorA, tempTensorA, AscendC::MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mins<T, false>(
        tempTensorA, tempTensorA, coefficientsBInv, AscendC::MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds<T, false>(
        tempTensorA, tempTensorA, coefficientsB, AscendC::MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<T, false>(
        tempTensorA, tempTensorA, tempTensorA, AscendC::MASK_PLACEHOLDER, params.repeatTimes, binaryParams);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls<T, false>(
        tempTensorA, tempTensorA, coefficientsA, AscendC::MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds<T, false>(
        tempTensorA, tempTensorA, coefficientsD, AscendC::MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T, bool HIGH_PERFORMANCE = false>
__aicore__ inline void FastGeluV2SgnParams(const AscendC::LocalTensor<T>& tempTensorB,
    const AscendC::LocalTensor<T>& tempTensorC, const AscendC::LocalTensor<T>& srcLocal,
    const AscendC::GeluParams<T>& params)
{
    T coefficients;
    if constexpr (AscendC::IsSameType<T, half>::value) {
        coefficients = 0.0000001;
    } else {
        coefficients = 0.000000000001;
    }

    const AscendC::UnaryRepeatParams unaryParams;
    const AscendC::BinaryRepeatParams binaryParams;

    AscendC::Adds<T, false>(
        tempTensorB, srcLocal, coefficients, AscendC::MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Abs<T, false>(
        tempTensorC, tempTensorB, AscendC::MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (HIGH_PERFORMANCE) {
        AscendC::Reciprocal<T, false>(
            tempTensorC, tempTensorC, AscendC::MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Mul<T, false>(
            tempTensorB, tempTensorB, tempTensorC, AscendC::MASK_PLACEHOLDER, params.repeatTimes, binaryParams);
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        AscendC::Div<T, false>(
            tempTensorB, tempTensorB, tempTensorC, AscendC::MASK_PLACEHOLDER, params.repeatTimes, binaryParams);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void FastGeluV2Params(const AscendC::LocalTensor<T>& tempTensorA,
    const AscendC::LocalTensor<T>& tempTensorB, const AscendC::LocalTensor<T>& dstLocal,
    const AscendC::LocalTensor<T>& srcLocal, const AscendC::GeluParams<T>& params)
{
    const T coefficientsHalf = 0.5;

    const AscendC::UnaryRepeatParams unaryParams;
    const AscendC::BinaryRepeatParams binaryParams;

    AscendC::Mul<T, false>(
        tempTensorA, tempTensorA, tempTensorB, AscendC::MASK_PLACEHOLDER, params.repeatTimes, binaryParams);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds<T, false>(
        tempTensorA, tempTensorA, coefficientsHalf, AscendC::MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul<T, false>(
        dstLocal, srcLocal, tempTensorA, AscendC::MASK_PLACEHOLDER, params.repeatTimes, binaryParams);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T, bool HIGH_PERFORMANCE = false>
__aicore__ inline void FasterGeluV2Func(const AscendC::LocalTensor<T>& dstLocal,
    const AscendC::LocalTensor<T>& srcLocal, const AscendC::GeluParams<T>& params)
{
    const AscendC::UnaryRepeatParams unaryParams;
    const AscendC::LocalTensor<T>& tempTensorA = params.tempTensorA;
    const AscendC::LocalTensor<T>& tempTensorB = params.tempTensorB;
    const AscendC::LocalTensor<T>& tempTensorC = params.tempTensorC;
    const AscendC::LocalTensor<T>& tempTensorConv = params.tempTensorConv;

    // x1 = (-0.1444) * (clip(|0.7071 * x|, max=1.769) - 1.769) ^ 2 + 0.5
    FFN::FastGeluV2ClipParams(tempTensorA, srcLocal, params);
    // x2 = (x + 0.0000001) / |(x + 0.0000001)|
    FFN::FastGeluV2SgnParams<T, HIGH_PERFORMANCE>(tempTensorB, tempTensorC, srcLocal, params);
    // fast_gelu_v2(x) = x * (x2 * x1 + 0.5)
    FFN::FastGeluV2Params(tempTensorA, tempTensorB, dstLocal, srcLocal, params);
}

template <typename T, bool HIGH_PRECISION = false, bool HIGH_PERFORMANCE = false>
__aicore__ inline void FasterGeluV2(const AscendC::LocalTensor<T>& dstLocal,
    const AscendC::LocalTensor<T>& srcLocal, const AscendC::LocalTensor<uint8_t>& sharedTmpBuffer,
    const uint32_t dataSize)
{
    if constexpr (HIGH_PRECISION && (AscendC::IsSameType<T, half>::value)) {
        AscendC::GeluClass<CONST_3>(dstLocal, srcLocal, sharedTmpBuffer, dataSize,
            FasterGeluV2Func<float, HIGH_PERFORMANCE>);
    } else {
        AscendC::GeluClass<T, CONST_3>(dstLocal, srcLocal, sharedTmpBuffer, dataSize,
            FasterGeluV2Func<T, HIGH_PERFORMANCE>);
    }
}
} // namespace FFN
#endif