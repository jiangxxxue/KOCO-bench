/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDOPS_POST_LAYER_NORM_BASE
#define ASDOPS_POST_LAYER_NORM_BASE

#include "kernel_operator.h"
#include "kernels/utils/kernel/kernel_utils.h"
#include "kernels/norm/postlayernorm/tiling/tiling_data.h"

using AscendC::HardEvent;

template <typename T>
class PostLayerNormBase {
public:
    __aicore__ inline PostLayerNormBase() {}

    __aicore__ inline void InitBase(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *gamma,
                                    __gm__ uint8_t *beta, AsdOps::PostLayerNormTilingData &tilingData)
    {
        zoomScale_ = tilingData.zoomScale;
        coreNum_ = tilingData.numCore;
        colNum_ = tilingData.numLastDim;
        uint32_t rowNumPerCore = tilingData.nlFirstdimPerCore;
        uint32_t rowNumLastCore = tilingData.lFirstdimPerCore;
        uint32_t rowNumPerTimes = tilingData.firstDimPerTimes;
        aveNum_ = tilingData.aveStr;
        eps_ = tilingData.epsStr;
        sliceNum_ = tilingData.sliceNum;
        sliceSize_ = tilingData.sliceSize;
        tailSliceSize_ = tilingData.tailSliceSize;
        if (AscendC::GetBlockIdx() != coreNum_ - 1) {
            rowNumCurrCore_ = rowNumPerCore;
            rowNumPerStep_ = rowNumPerTimes;
        } else {
            rowNumCurrCore_ = rowNumLastCore;
            rowNumPerStep_ = Min(rowNumPerTimes, rowNumCurrCore_);
        }

        rowNumLastStep_ = (rowNumCurrCore_ % rowNumPerTimes == 0) ? rowNumPerTimes :
                                                                    (rowNumCurrCore_ % rowNumPerTimes);

        gmOffset_ = static_cast<uint64_t>(rowNumPerCore) * colNum_ * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ T *)x + gmOffset_);
        yGm.SetGlobalBuffer((__gm__ T *)y + gmOffset_);
        gammaGm.SetGlobalBuffer((__gm__ T *)gamma);
        betaGm.SetGlobalBuffer((__gm__ T *)beta);

        pipe.InitBuffer(xQue, BUFFER_NUM, rowNumPerStep_ * RoundUp(sliceSize_) * sizeof(T));
        pipe.InitBuffer(yQue, BUFFER_NUM, rowNumPerStep_ * RoundUp(sliceSize_) * sizeof(T));
        pipe.InitBuffer(betaQue, BUFFER_NUM, RoundUp(sliceSize_) * sizeof(T));
        pipe.InitBuffer(gammaQue, BUFFER_NUM, RoundUp(sliceSize_) * sizeof(T));
        pipe.InitBuffer(xBufFp32, RoundUp(sliceSize_) * sizeof(float));
        pipe.InitBuffer(yBufFp32, RoundUp(sliceSize_) * sizeof(float));
        yLocalFp32 = yBufFp32.Get<float>();
        xLocalFp32 = xBufFp32.Get<float>();
        xLocalFp16 = xBufFp32.Get<T>();
    }

    __aicore__ inline void CopyInMultiRow(uint64_t procId, int32_t size)
    {
        uint64_t offset = procId * rowNumPerStep_ * colNum_;
        CopyIn(xGm, xQue, offset, size);
        CopyIn(yGm, yQue, offset, size);
        CopyIn(betaGm, betaQue, 0, colNum_);
        CopyIn(gammaGm, gammaQue, 0, colNum_);
    }

    __aicore__ inline void ComputeResidual(uint64_t offset, uint32_t count)
    {
        CopyInAndCastF32(xLocalFp32, xGm, xQue, offset, count);
        CopyInAndCastF32(yLocalFp32, yGm, yQue, offset, count);
        Muls(yLocalFp32, yLocalFp32, zoomScale_, count);
        AscendC::PipeBarrier<PIPE_V>();
        ComputeResidualAdd(xLocalFp32, xLocalFp32, yLocalFp32, count);
    }

    __aicore__ inline void GetSizeAndOffset(uint64_t rowIdx, uint64_t sliceIdx,
        uint32_t &size, uint64_t &offset, uint64_t &sliceOffset)
    {
        size = (sliceIdx != sliceNum_ - 1) ? sliceSize_ : tailSliceSize_;
        sliceOffset = sliceIdx * sliceSize_;
        offset = rowIdx * colNum_ + sliceOffset;
    }

    __aicore__ inline float ComputeRowMean(uint64_t rowIdx)
    {
        float sum = 0;
        for (uint64_t sliceIdx = 0; sliceIdx < sliceNum_; sliceIdx++) {
            uint32_t size = 0;
            uint64_t offset = 0;
            uint64_t sliceOffset = 0;
            GetSizeAndOffset(rowIdx, sliceIdx, size, offset, sliceOffset);
            ComputeResidual(offset, size);
            sum += ComputeSum(xLocalFp32, yLocalFp32, yLocalFp32, size);
            AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
        }

        return sum * aveNum_;
    }

    __aicore__ inline float ComputeRowStd(uint64_t rowIdx, float mean)
    {
        float squareSum = 0;
        for (uint64_t sliceIdx = 0; sliceIdx < sliceNum_; sliceIdx++) {
            uint32_t size = 0;
            uint64_t offset = 0;
            uint64_t sliceOffset = 0;
            GetSizeAndOffset(rowIdx, sliceIdx, size, offset, sliceOffset);
            ComputeResidual(offset, size);
            Adds(yLocalFp32, xLocalFp32, -mean, size);
            AscendC::PipeBarrier<PIPE_V>();
            squareSum += ComputeSliceSquareSum(yLocalFp32, yLocalFp32, yLocalFp32, size);
            AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
        }

        float var = squareSum * aveNum_ + eps_;
        return sqrt(var);
    }

    __aicore__ inline void ComputeSliceLayernorm(uint64_t offset, uint64_t sliceOffset, uint32_t size,
        float mean, float std)
    {
        ComputeResidual(offset, size);
        Adds(yLocalFp32, xLocalFp32, -mean, size);
        AscendC::PipeBarrier<PIPE_V>();
        Muls(yLocalFp32, yLocalFp32, (float)1.0 / std, size);
        AscendC::PipeBarrier<PIPE_V>();
        CopyInAndCastF32(xLocalFp32, gammaGm, gammaQue, sliceOffset, size);
        Mul(yLocalFp32, xLocalFp32, yLocalFp32, size);
        AscendC::PipeBarrier<PIPE_V>();
        CopyInAndCastF32(xLocalFp32, betaGm, betaQue, sliceOffset, size);
        Add(yLocalFp32, xLocalFp32, yLocalFp32, size);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeRes(uint32_t rid, const AscendC::LocalTensor<T> &xLocal,
        const AscendC::LocalTensor<T> &yLocal)
    {
        const AscendC::LocalTensor<T> &xLocalInner = xLocal[rid * colNum_];
        const AscendC::LocalTensor<T> &yLocalInner = yLocal[rid * colNum_];
        CastFrom16To32(xLocalFp32, xLocalInner, colNum_);
        CastFrom16To32(yLocalFp32, yLocalInner, colNum_);
        Muls(yLocalFp32, yLocalFp32, zoomScale_, colNum_);
        AscendC::PipeBarrier<PIPE_V>();
        ComputeResidualAdd(xLocalFp32, xLocalFp32, yLocalFp32, colNum_);
    }

    __aicore__ inline void ComputeRowLayernorm(const AscendC::LocalTensor<T> gamma,
        const AscendC::LocalTensor<T> beta)
    {
        ComputeMean(yLocalFp32, xLocalFp32, aveNum_, colNum_);
        ComputeLayerNorm(yLocalFp32, xLocalFp32, yLocalFp32, eps_, aveNum_, gamma, beta, colNum_);
    }
protected:
    static constexpr int32_t BUFFER_NUM = 1;

protected:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> xQue;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> yQue;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> gammaQue;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> betaQue;

    AscendC::TBuf<AscendC::TPosition::VECCALC> xBufFp32;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yBufFp32;
    AscendC::LocalTensor<float> yLocalFp32;
    AscendC::LocalTensor<float> xLocalFp32;
    AscendC::LocalTensor<T> xLocalFp16;
    AscendC::GlobalTensor<T> xGm, yGm, gammaGm, betaGm;

    uint32_t coreNum_{0};
    uint32_t colNum_{0};
    uint32_t rowNumCurrCore_{0};
    uint32_t rowNumPerStep_{0};
    uint32_t rowNumLastStep_{0};
    float aveNum_{0};
    float eps_{0};
    float zoomScale_{0};
    uint32_t sliceNum_{0};
    uint32_t sliceSize_{0};
    uint32_t tailSliceSize_{0};
    uint64_t gmOffset_{0};
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AsdOps::PostLayerNormTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->numCore = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->numLastDim = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->numFirstDim = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
    tilingdata->nlFirstdimPerCore = (*(const __gm__ uint32_t *)(p_tilingdata + 12));
    tilingdata->lFirstdimPerCore = (*(const __gm__ uint32_t *)(p_tilingdata + 16));
    tilingdata->firstDimPerTimes = (*(const __gm__ uint32_t *)(p_tilingdata + 20));
    tilingdata->epsStr = (*(const __gm__ float *)(p_tilingdata + 24));
    tilingdata->aveStr = (*(const __gm__ float *)(p_tilingdata + 28));
    tilingdata->normMode = (*(const __gm__ uint32_t *)(p_tilingdata + 32));
    tilingdata->zoomScale = (*(const __gm__ float *)(p_tilingdata + 36));
    tilingdata->sliceNum = (*(const __gm__ uint32_t *)(p_tilingdata + 40));
    tilingdata->sliceSize = (*(const __gm__ uint32_t *)(p_tilingdata + 44));
    tilingdata->tailSliceSize = (*(const __gm__ uint32_t *)(p_tilingdata + 48));
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AsdOps::PostLayerNormTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->numCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->numLastDim = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    tilingdata->numFirstDim = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 8));
    tilingdata->nlFirstdimPerCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 12));
    tilingdata->lFirstdimPerCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 16));
    tilingdata->firstDimPerTimes = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 20));
    tilingdata->epsStr = (*(__ubuf__ float *)(tilingdata_in_ub + 24));
    tilingdata->aveStr = (*(__ubuf__ float *)(tilingdata_in_ub + 28));
    tilingdata->normMode = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 32));
    tilingdata->zoomScale = (*(__ubuf__ float *)(tilingdata_in_ub + 36));
    tilingdata->sliceNum = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 40));
    tilingdata->sliceSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 44));
    tilingdata->tailSliceSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 48));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    AsdOps::PostLayerNormTilingData tiling_data; \
    InitTilingData(tiling_arg, &(tiling_data))

#endif