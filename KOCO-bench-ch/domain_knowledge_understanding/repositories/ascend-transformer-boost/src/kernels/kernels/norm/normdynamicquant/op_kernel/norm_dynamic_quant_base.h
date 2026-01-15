/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_NORM_DYNAMIC_QUANT_BASE_H
#define ASCEND_OPS_NORM_DYNAMIC_QUANT_BASE_H

#include "kernel_operator.h"
#include "kernels/utils/kernel/kernel_utils.h"
#include "../tiling/norm_dynamic_quant_tiling_data.h"

using AscendC::HardEvent;

class NormDynamicQuantBase {
public:
    __aicore__ inline NormDynamicQuantBase() {}
    __aicore__ inline void InitBase(__gm__ uint8_t *x, __gm__ uint8_t *g, __gm__ uint8_t *b,
                                    __gm__ uint8_t *y, __gm__ uint8_t *scale, __gm__ uint8_t *offset,
                                    AsdOps::NormDynamicQuantTilingData &tilingData)
    {
        avgFactor_ = tilingData.avgFactor;
        epsilon_ = tilingData.epsilon;
        quantMin_ = tilingData.quantMin;

        uint32_t rowNumPerCore = tilingData.numRowPerCore;
        rowNum_ = (AscendC::GetBlockIdx() != tilingData.numCore - 1) ?
            rowNumPerCore : (tilingData.numRow - (tilingData.numCore - 1) * rowNumPerCore);
        colNum_ = tilingData.numCol;

        int64_t gmOffset = rowNumPerCore * colNum_;
        xGm_.SetGlobalBuffer((__gm__ half *)x + AscendC::GetBlockIdx() * gmOffset);
        gGm_.SetGlobalBuffer((__gm__ half *)g);
        bGm_.SetGlobalBuffer((__gm__ half *)b);
        yGm_.SetGlobalBuffer((__gm__ int8_t *)y + AscendC::GetBlockIdx() * gmOffset);
        sGm_.SetGlobalBuffer((__gm__ float *)scale + AscendC::GetBlockIdx() * rowNumPerCore);
        oGm_.SetGlobalBuffer((__gm__ float *)offset + AscendC::GetBlockIdx() * rowNumPerCore);

        sliceSize_ = colNum_; // 当前尾轴支持大小有限
        pipe.InitBuffer(xQue_, BUFFER_NUM, sliceSize_ * sizeof(half));
        pipe.InitBuffer(gQue_, BUFFER_NUM, sliceSize_ * sizeof(half));
        pipe.InitBuffer(bQue_, BUFFER_NUM, sliceSize_ * sizeof(half));
        pipe.InitBuffer(yQue_, BUFFER_NUM, sliceSize_ * sizeof(int8_t));

        pipe.InitBuffer(tmpFp32Buf_, sliceSize_ * sizeof(float));
        pipe.InitBuffer(calcFp32Buf_, sliceSize_ * sizeof(float));
        pipe.InitBuffer(scaleFp32Buf_, 32);
        pipe.InitBuffer(offsetFp32Buf_, 32);
        tmpFp32_ = tmpFp32Buf_.Get<float>();
        calcFp32_ = calcFp32Buf_.Get<float>();
        tmpFp16_ = tmpFp32Buf_.Get<half>();
        calcFp16_ = calcFp32Buf_.Get<half>();
        scaleFp32_ = scaleFp32Buf_.Get<float>();
        offsetFp32_ = offsetFp32Buf_.Get<float>();
    }

protected:
    __aicore__ inline void CopyInAll(uint64_t offset, uint32_t count)
    {
        CopyInAndCastF32(calcFp32_, xGm_, xQue_, offset, count);
        CopyIn(gGm_, gQue_, 0, count);
        CopyIn(bGm_, bQue_, 0, count);
    }

    template <bool IS_SYMMETRIC>
    __aicore__ inline void CopyOutAll(uint64_t rowIdx, uint64_t offset, uint32_t count)
    {
        if ((rowIdx + 1) % 8 == 0 || rowIdx + 1 == rowNum_) {
            DataCopy(sGm_[rowIdx / 8 * 8], scaleFp32_, 8);
            if constexpr (!IS_SYMMETRIC) {
                DataCopy(oGm_[rowIdx / 8 * 8], offsetFp32_, 8);
            }
        }
        CopyOut(yGm_, yQue_, offset, count);
    }

    /* 对称量化 */
    __aicore__ inline void SymmetricQuant(uint64_t rowIdx, const AscendC::LocalTensor<int8_t> &out,
        const AscendC::LocalTensor<half> &in,
        const AscendC::LocalTensor<half> &tmp, uint32_t count)
    {
        Abs(tmp, in, count);
        AscendC::PipeBarrier<PIPE_V>();
        ReduceMax(tmp, tmp, tmp, count, false);
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float maxValue = tmp.GetValue(0);
        float scaleOut = maxValue / 127;
        uint64_t idx = rowIdx % 8;
        scaleFp32_.SetValue(idx, scaleOut);
        AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Muls(tmp, in, (half)(1 / scaleOut), count);
        AscendC::PipeBarrier<PIPE_V>();
        CastFromF16ToI8(out, tmp, quantMin_, count);
    }

    /* 非对称量化 */
    __aicore__ inline void AsymmetricQuant(uint64_t rowIdx, const AscendC::LocalTensor<int8_t> &out,
        const AscendC::LocalTensor<half> &in, const AscendC::LocalTensor<half> &tmp, uint32_t count)
    {
        ReduceMax(tmp, in, tmp, count, false);
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float maxValue = tmp.GetValue(0);
        AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
        ReduceMin(tmp, in, tmp, count, false);
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float minValue = tmp.GetValue(0);
        float subValue = maxValue - minValue;
        float scaleOut = subValue / 255;
        float offsetOut = -(maxValue + minValue) / (scaleOut * 2);  // 0.5f: half of (maxValue + minValue)
        uint64_t idx = rowIdx % 8;
        scaleFp32_.SetValue(idx, scaleOut);
        offsetFp32_.SetValue(idx, offsetOut);
        AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Muls(tmp, in, (half)(1 / scaleOut), count);
        AscendC::PipeBarrier<PIPE_V>();
        Adds(tmp, tmp, (half)offsetOut, count);
        AscendC::PipeBarrier<PIPE_V>();
        CastFromF16ToI8(out, tmp, quantMin_, count);
    }

protected:
    static constexpr uint32_t BUFFER_NUM = 1;

    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> xQue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> gQue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> bQue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> yQue_;

    AscendC::GlobalTensor<half> xGm_;
    AscendC::GlobalTensor<half> gGm_;
    AscendC::GlobalTensor<half> bGm_;
    AscendC::GlobalTensor<int8_t> yGm_;
    AscendC::GlobalTensor<float> sGm_;
    AscendC::GlobalTensor<float> oGm_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scaleFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> offsetFp32Buf_;
    AscendC::LocalTensor<float> calcFp32_;
    AscendC::LocalTensor<float> tmpFp32_;
    AscendC::LocalTensor<half> calcFp16_;
    AscendC::LocalTensor<half> tmpFp16_;
    AscendC::LocalTensor<float> scaleFp32_;
    AscendC::LocalTensor<float> offsetFp32_;

    uint32_t rowNum_{0};
    uint32_t colNum_{0};
    uint32_t sliceSize_{0};
    float avgFactor_{1.0f};
    float epsilon_{1e-12f};
    half quantMin_{-128};
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata,
                                      AsdOps::NormDynamicQuantTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->numCore = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->numCol = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->numRow = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
    tilingdata->avgFactor = (*(const __gm__ float *)(p_tilingdata + 12));
    tilingdata->epsilon = (*(const __gm__ float *)(p_tilingdata + 16));
    tilingdata->quantMin = (*(const __gm__ float *)(p_tilingdata + 20));
    tilingdata->numRowPerCore = (*(const __gm__ uint32_t *)(p_tilingdata + 24));
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AsdOps::NormDynamicQuantTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->numCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->numCol = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    tilingdata->numRow = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 8));
    tilingdata->avgFactor = (*(__ubuf__ float *)(tilingdata_in_ub + 12));
    tilingdata->epsilon = (*(__ubuf__ float *)(tilingdata_in_ub + 16));
    tilingdata->quantMin = (*(__ubuf__ float *)(tilingdata_in_ub + 20));
    tilingdata->numRowPerCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 24));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

#define GET_TILING_DATA(tilingData, tiling_arg)           \
    AsdOps::NormDynamicQuantTilingData tilingData;        \
    InitTilingData((tiling_arg), &(tilingData))

#endif