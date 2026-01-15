/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NORM_COMMMON_QUANT_H
#define NORM_COMMMON_QUANT_H

#include "kernel_operator.h"

static constexpr uint32_t BLOCK_SIZE = 32;

using AscendC::HardEvent;

template <typename S, typename O>
__aicore__ inline void GetQuantInfo(S &varScale, O &varOffset, __gm__ uint8_t *scale, __gm__ uint8_t *offset,
                                    AscendC::TBuf<AscendC::TPosition::VECCALC> &buf)
{
    AscendC::GlobalTensor<half> gm_s;
    AscendC::GlobalTensor<int8_t> gm_o;
    gm_s.SetGlobalBuffer((__gm__ half *)scale);
    gm_o.SetGlobalBuffer((__gm__ int8_t *)offset);

    AscendC::LocalTensor<half> scale_buffer = buf.Get<half>();
    DataCopy(scale_buffer, gm_s, BLOCK_SIZE / sizeof(half));
    AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    varScale = 1 / static_cast<S>(scale_buffer.GetValue(0));

    AscendC::LocalTensor<int8_t> offset_buffer = buf.Get<int8_t>();
    AscendC::SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    DataCopy(offset_buffer, gm_o, BLOCK_SIZE / sizeof(int8_t));
    AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    varOffset = static_cast<O>(offset_buffer.GetValue(0));
    AscendC::SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
}

template<typename T>
__aicore__ inline void GetScaleAndOffset(float &varScale, float &varOffset, __gm__ uint8_t *scale,
                                         __gm__ uint8_t *offset, AscendC::TBuf<AscendC::TPosition::VECCALC> &buf)
{
    AscendC::GlobalTensor<T> gm_s;
    gm_s.SetGlobalBuffer((__gm__ T *)scale);
    AscendC::LocalTensor<T> scale_buffer = buf.Get<T>();
    DataCopy(scale_buffer, gm_s, BLOCK_SIZE / sizeof(T));
    if constexpr (AscendC::IsSameType<T, half>::value) {
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        varScale = 1 / (float)(scale_buffer.GetValue(0));
    } else {
        AscendC::LocalTensor<float> tmpFp32 = buf.Get<float>();
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(tmpFp32, scale_buffer, AscendC::RoundMode::CAST_NONE, 1);
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
        varScale = 1 / (float)(tmpFp32.GetValue(0));
    }
    AscendC::GlobalTensor<int8_t> gm_o;
    gm_o.SetGlobalBuffer((__gm__ int8_t *)offset);
    AscendC::LocalTensor<int8_t> tmpInt8 = buf.Get<int8_t>();
    DataCopy(tmpInt8, gm_o, BLOCK_SIZE / sizeof(int8_t));
    AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    varOffset = (float)(tmpInt8.GetValue(0));
}

#endif