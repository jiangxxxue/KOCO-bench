/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "simple_broadcast_base.h"
#include "kernels/utils/kernel/kernel_utils.h"

using AscendC::HardEvent;

template <typename T, bool HAS_OFFSET, bool CUT_N, bool QUANT_MIN_NEG_127>
class AtbQuantPerChannelKernel : public SimpleBroadcastBase {
public:
    __aicore__ inline AtbQuantPerChannelKernel() {}

    __aicore__ inline void Process(GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y)
    {
        InitBroadcastGlobalTensor<T>(xGm_, x);
        InitBroadcastQueue<T>(xQueue_);
        InitNormalGlobalTensor<T>(scaleGm_, scale);
        InitNormalQueue<T>(scaleQueue_);
        InitNormalGlobalTensor<int8_t>(offsetGm_, offset);
        InitNormalQueue<int8_t>(offsetQueue_);
        InitBroadcastGlobalTensor<int8_t>(yGm_, y);
        InitBroadcastQueue<int8_t>(yQueue_);

        InitNormalTBuf<half>(f16TmpBuf_);
        f16Local_ = f16TmpBuf_.Get<half>();
        InitNormalTBuf<half>(offsetCastedBuf_);
        offsetCastedLocal_ = offsetCastedBuf_.Get<half>();

        quantMin_ = QUANT_MIN_NEG_127 ? -127 : -128; // -127/-128: lower bound

        if constexpr (!IsSameType<T, half>::value) {
            // BF16需要先转换为FP32，需要额外的内存空间
            InitDedicatedFp32Buf();
        }

        if constexpr (!CUT_N) {
            CopyInScaleAndOffset();
        }

        do {
            CopyInBroadcast(xGm_, xQueue_);
            LocalTensor<T> xLocal = xQueue_.DeQue<T>();

            if constexpr (CUT_N) {
                CopyInScaleAndOffset();
            }

            LocalTensor<int8_t> yLocal = yQueue_.AllocTensor<int8_t>();
            do {
                LocalTensor<T> xLocalInner = GetInnerTensor(xLocal);
                if constexpr (IsSameType<T, half>::value) {
                    Div(f16Local_, xLocalInner, scaleLocal_, innerCount_);
                } else {
                    Cast(xCastedLocal_, xLocalInner, RoundMode::CAST_NONE, innerCount_);
                    PipeBarrier<PIPE_V>();
                    Div(f32Local_, xCastedLocal_, scaleCastedLocal_, innerCount_);
                    PipeBarrier<PIPE_V>();
                    Cast(f16Local_, f32Local_, RoundMode::CAST_NONE, innerCount_);
                }

                if constexpr (HAS_OFFSET) {
                    PipeBarrier<PIPE_V>();
                    Add(f16Local_, f16Local_, offsetCastedLocal_, innerCount_);
                }
                PipeBarrier<PIPE_V>();

                LocalTensor<int8_t> yLocalInner = GetInnerTensor(yLocal);
                CastFromF16ToI8(yLocalInner, f16Local_, quantMin_, innerCount_);
            } while (InnerNext());

            yQueue_.EnQue(yLocal);
            xQueue_.FreeTensor(xLocal);
            if constexpr (CUT_N) {
                scaleQueue_.FreeTensor(scaleLocal_);
                if constexpr (HAS_OFFSET) {
                    offsetQueue_.FreeTensor(offsetLocal_);
                }
            }

            CopyOut(yGm_, yQueue_);
        } while (OuterNext());

        if constexpr (!CUT_N) {
            scaleQueue_.FreeTensor(scaleLocal_);
            if constexpr (HAS_OFFSET) {
                offsetQueue_.FreeTensor(offsetLocal_);
            }
        }
    }
private:
    __aicore__ inline void InitDedicatedFp32Buf()
    {
        InitNormalTBuf<float>(f32TmpBuf_);
        f32Local_ = f32TmpBuf_.Get<float>();

        InitBroadcastTBuf<float>(xCastedBuf_);
        xCastedLocal_ = xCastedBuf_.Get<float>();

        InitNormalTBuf<float>(scaleCastedBuf_);
        scaleCastedLocal_ = scaleCastedBuf_.Get<float>();
    }

    __aicore__ inline void CopyInScaleAndOffset()
    {
        CopyInNormal(scaleGm_, scaleQueue_);
        scaleLocal_ = scaleQueue_.DeQue<T>();
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        if constexpr (IsSameType<T, bfloat16_t>::value) {
            Cast(scaleCastedLocal_, scaleLocal_, RoundMode::CAST_NONE, innerCount_);
            PipeBarrier<PIPE_V>();
        }
#endif
        if constexpr (HAS_OFFSET) {
            CopyInNormal(offsetGm_, offsetQueue_);
            offsetLocal_ = offsetQueue_.DeQue<int8_t>();
            Cast(offsetCastedLocal_, offsetLocal_, RoundMode::CAST_NONE, innerCount_);
            PipeBarrier<PIPE_V>();
        }
    }

private:
    GlobalTensor<T> xGm_;
    GlobalTensor<T> scaleGm_;
    GlobalTensor<int8_t> offsetGm_;
    GlobalTensor<int8_t> yGm_;

    LocalTensor<T> scaleLocal_;
    LocalTensor<int8_t> offsetLocal_;

    TBuf<QuePosition::VECCALC> xCastedBuf_;
    TBuf<QuePosition::VECCALC> scaleCastedBuf_;
    TBuf<QuePosition::VECCALC> offsetCastedBuf_;
    LocalTensor<float> xCastedLocal_;
    LocalTensor<float> scaleCastedLocal_;
    LocalTensor<half> offsetCastedLocal_;

    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue_;
    TQue<QuePosition::VECIN, 1> scaleQueue_;
    TQue<QuePosition::VECIN, 1> offsetQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue_;

    TBuf<QuePosition::VECCALC> f16TmpBuf_;
    TBuf<QuePosition::VECCALC> f32TmpBuf_;
    LocalTensor<float> f32Local_;
    LocalTensor<half> f16Local_;

    half quantMin_{-128};
};

template <typename T, bool HAS_OFFSET, bool QUANT_MIN_NEG_127>
class QuantPerTensorKernel : public SimpleBroadcastBase {
public:
    __aicore__ inline QuantPerTensorKernel() {}

    __aicore__ inline void Process(GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y)
    {
        InitBroadcastGlobalTensor<T>(xGm_, x);
        InitBroadcastGlobalTensor<int8_t>(yGm_, y);
        InitBroadcastQueue<T>(xQueue_);
        InitBroadcastQueue<int8_t>(yQueue_);

        InitNormalTBuf<half>(f16TmpBuf_);
        f16Local_ = f16TmpBuf_.Get<half>();

        quantMin_ = QUANT_MIN_NEG_127 ? -127 : -128; // -127/-128: lower bound

        if constexpr (!IsSameType<T, half>::value) {
            // BF16需要先转换为FP32，需要额外的内存空间
            InitDedicatedFp32Buf();
        }

        float scaleValue = GetScaleValue(scale);
        half offsetValue = GetOffsetValue(offset);

        do {
            CopyInBroadcast(xGm_, xQueue_);
            LocalTensor<T> xLocal = xQueue_.DeQue<T>();

            LocalTensor<int8_t> yLocal = yQueue_.AllocTensor<int8_t>();
            do {
                LocalTensor<T> xLocalInner = GetInnerTensor(xLocal);
                if constexpr (IsSameType<T, half>::value) {
                    Muls(f16Local_, xLocalInner, (half)scaleValue, innerCount_);
                } else {
                    Cast(xCastedLocal_, xLocal, RoundMode::CAST_NONE, innerCount_);
                    PipeBarrier<PIPE_V>();
                    Muls(f32Local_, xCastedLocal_, scaleValue, innerCount_);
                    PipeBarrier<PIPE_V>();
                    Cast(f16Local_, f32Local_, RoundMode::CAST_NONE, innerCount_);
                }

                if constexpr (HAS_OFFSET) {
                    PipeBarrier<PIPE_V>();
                    Adds(f16Local_, f16Local_, offsetValue, innerCount_);
                }
                PipeBarrier<PIPE_V>();

                LocalTensor<int8_t> yLocalInner = GetInnerTensor(yLocal);
                CastFromF16ToI8(yLocalInner, f16Local_, quantMin_, innerCount_);
            } while (InnerNext());

            yQueue_.EnQue(yLocal);
            xQueue_.FreeTensor(xLocal);

            CopyOut(yGm_, yQueue_);
        } while (OuterNext());
    }
private:
    __aicore__ inline void InitDedicatedFp32Buf()
    {
        InitNormalTBuf<float>(f32TmpBuf_);
        f32Local_ = f32TmpBuf_.Get<float>();

        InitBroadcastTBuf<float>(xCastedBuf_);
        xCastedLocal_ = xCastedBuf_.Get<float>();
    }

    __aicore__ inline float GetScaleValue(GM_ADDR addr)
    {
        GlobalTensor<T> scaleGm;
        scaleGm.SetGlobalBuffer((__gm__ T *)addr);

        LocalTensor<T> tmpLocal = f16TmpBuf_.Get<T>();
        DataCopy(tmpLocal, scaleGm, BLOCK_SIZE / sizeof(T));
        if constexpr (IsSameType<T, half>::value) {
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            return (1.0f / (float)tmpLocal.GetValue(0));
        }

        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(f32Local_, tmpLocal, RoundMode::CAST_NONE, 1);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        return (1.0f / f32Local_.GetValue(0));
    }

    __aicore__ inline half GetOffsetValue(GM_ADDR addr)
    {
        if constexpr (!HAS_OFFSET) {
            return 0.0f;
        }

        GlobalTensor<int8_t> offsetGm;
        offsetGm.SetGlobalBuffer((__gm__ int8_t *)addr);

        LocalTensor<int8_t> tmpInt8Local = f16TmpBuf_.Get<int8_t>();
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        DataCopy(tmpInt8Local, offsetGm, BLOCK_SIZE / sizeof(int8_t));
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        return (half)(tmpInt8Local.GetValue(0));
    }

private:
    GlobalTensor<T> xGm_;
    GlobalTensor<int8_t> yGm_;

    TBuf<QuePosition::VECCALC> xCastedBuf_;
    LocalTensor<float> xCastedLocal_;

    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue_;

    TBuf<QuePosition::VECCALC> f16TmpBuf_;
    TBuf<QuePosition::VECCALC> f32TmpBuf_;
    LocalTensor<float> f32Local_;
    LocalTensor<half> f16Local_;

    half quantMin_{-128};
};

extern "C" __global__ __aicore__ void quant_per_channel(
    GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(2000000000)) {
        // per tensor, no offset, -128, fp16
        QuantPerTensorKernel<half, false, false> op;
        op.Init(&tilingData);
        op.Process(x, scale, offset, y);
    } else if (TILING_KEY_IS(2100000000)) {
        // per channel, no offset, cut n, -128, fp16
        AtbQuantPerChannelKernel<half, false, true, false> op;
        op.Init(&tilingData);
        op.Process(x, scale, offset, y);
    } else if (TILING_KEY_IS(2200000000)) {
        // per channel, no offset, cut b, -128, fp16
        AtbQuantPerChannelKernel<half, false, false, false> op;
        op.Init(&tilingData);
        op.Process(x, scale, offset, y);
    } else if (TILING_KEY_IS(2000000001)) {
        // per tensor, has offset, -128, fp16
        QuantPerTensorKernel<half, true, false> op;
        op.Init(&tilingData);
        op.Process(x, scale, offset, y);
    } else if (TILING_KEY_IS(2100000001)) {
        // per channel, has offset, cut n, -128, fp16
        AtbQuantPerChannelKernel<half, true, true, false> op;
        op.Init(&tilingData);
        op.Process(x, scale, offset, y);
    } else if (TILING_KEY_IS(2200000001)) {
        // per channel, has offset, cut b, -128, fp16
        AtbQuantPerChannelKernel<half, true, false, false> op;
        op.Init(&tilingData);
        op.Process(x, scale, offset, y);
    } else if (TILING_KEY_IS(2000000010)) {
        // per tensor, no offset, -127, fp16
        QuantPerTensorKernel<half, false, true> op;
        op.Init(&tilingData);
        op.Process(x, scale, offset, y);
    } else if (TILING_KEY_IS(2100000010)) {
        // per channel, no offset, cut n, -127, fp16
        AtbQuantPerChannelKernel<half, false, true, true> op;
        op.Init(&tilingData);
        op.Process(x, scale, offset, y);
    } else if (TILING_KEY_IS(2200000010)) {
        // per channel, no offset, cut b, -127, fp16
        AtbQuantPerChannelKernel<half, false, false, true> op;
        op.Init(&tilingData);
        op.Process(x, scale, offset, y);
    } else if (TILING_KEY_IS(2000000011)) {
        // per tensor, has offset, -127, fp16
        QuantPerTensorKernel<half, true, true> op;
        op.Init(&tilingData);
        op.Process(x, scale, offset, y);
    } else if (TILING_KEY_IS(2100000011)) {
        // per channel, has offset, cut n, -127, fp16
        AtbQuantPerChannelKernel<half, true, true, true> op;
        op.Init(&tilingData);
        op.Process(x, scale, offset, y);
    } else if (TILING_KEY_IS(2200000011)) {
        // per channel, has offset, cut b, -127, fp16
        AtbQuantPerChannelKernel<half, true, false, true> op;
        op.Init(&tilingData);
        op.Process(x, scale, offset, y);
    }
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if (TILING_KEY_IS(2010000000)) {
        // per tensor, no offset, -128, bf16
        QuantPerTensorKernel<bfloat16_t, false, false> op;
        op.Init(&tilingData);
        op.Process(x, scale, offset, y);
    } else if (TILING_KEY_IS(2110000000)) {
        // per channel, no offset, cut n, -128, bf16
        AtbQuantPerChannelKernel<bfloat16_t, false, true, false> op;
        op.Init(&tilingData);
        op.Process(x, scale, offset, y);
    } else if (TILING_KEY_IS(2210000000)) {
        // per channel, no offset, cut b, -128, bf16
        AtbQuantPerChannelKernel<bfloat16_t, false, false, false> op;
        op.Init(&tilingData);
        op.Process(x, scale, offset, y);
    } else if (TILING_KEY_IS(2010000001)) {
        // per tensor, has offset, -128, bf16
        QuantPerTensorKernel<bfloat16_t, true, false> op;
        op.Init(&tilingData);
        op.Process(x, scale, offset, y);
    } else if (TILING_KEY_IS(2110000001)) {
        // per channel, has offset, cut n, -128, bf16
        AtbQuantPerChannelKernel<bfloat16_t, true, true, false> op;
        op.Init(&tilingData);
        op.Process(x, scale, offset, y);
    } else if (TILING_KEY_IS(2210000001)) {
        // per channel, has offset, cut b, -128, bf16
        AtbQuantPerChannelKernel<bfloat16_t, true, false, false> op;
        op.Init(&tilingData);
        op.Process(x, scale, offset, y);
    } else if (TILING_KEY_IS(2010000010)) {
        // per tensor, no offset, -127, bf16
        QuantPerTensorKernel<bfloat16_t, false, true> op;
        op.Init(&tilingData);
        op.Process(x, scale, offset, y);
    } else if (TILING_KEY_IS(2110000010)) {
        // per channel, no offset, cut n, -127, bf16
        AtbQuantPerChannelKernel<bfloat16_t, false, true, true> op;
        op.Init(&tilingData);
        op.Process(x, scale, offset, y);
    } else if (TILING_KEY_IS(2210000010)) {
        // per channel, no offset, cut b, -127, bf16
        AtbQuantPerChannelKernel<bfloat16_t, false, false, true> op;
        op.Init(&tilingData);
        op.Process(x, scale, offset, y);
    } else if (TILING_KEY_IS(2010000011)) {
        // per tensor, has offset, -127, bf16
        QuantPerTensorKernel<bfloat16_t, true, true> op;
        op.Init(&tilingData);
        op.Process(x, scale, offset, y);
    } else if (TILING_KEY_IS(2110000011)) {
        // per channel, has offset, cut n, -127, bf16
        AtbQuantPerChannelKernel<bfloat16_t, true, true, true> op;
        op.Init(&tilingData);
        op.Process(x, scale, offset, y);
    } else if (TILING_KEY_IS(2210000011)) {
        // per channel, has offset, cut b, -127, bf16
        AtbQuantPerChannelKernel<bfloat16_t, true, false, true> op;
        op.Init(&tilingData);
        op.Process(x, scale, offset, y);
    }
#endif
}