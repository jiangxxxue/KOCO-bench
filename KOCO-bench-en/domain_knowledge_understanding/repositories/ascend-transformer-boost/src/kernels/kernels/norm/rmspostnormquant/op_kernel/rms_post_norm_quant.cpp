/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "kernels/utils/kernel/kernel_utils.h"
#include "kernels/norm/common/common_tiling_data.h"
#include "kernels/norm/common/op_kernel/common_quant.h"

using AscendC::HardEvent;

static constexpr uint32_t BUFFER_NUM = 1;
static constexpr uint32_t DOUBLE_BUFFER_NUM = 2;       // split the UB to 2 equal part to enable ping-pong techniques.

template <typename T, bool FastComputeMode = false>
class RmsPostNormQuant {
public:
    __aicore__ inline RmsPostNormQuant(AscendC::TPipe *pipe, __gm__ uint8_t *x, __gm__ uint8_t *r, __gm__ uint8_t *g,
                                      __gm__ uint8_t *scale, __gm__ uint8_t *offset, __gm__ uint8_t *y,
                                      __gm__ uint8_t *res, AsdOps::RmsNormQuantCommonTilingData &tilingData)
    {
        numCore_ = tilingData.numCore;
        numCol_ = tilingData.numCol;
        avgFactor_ = *reinterpret_cast<float *>(&tilingData.avgFactor);
        epsilon_ = *reinterpret_cast<float *>(&tilingData.epsilon);
        quantMin_ = tilingData.quantMin;
        sliceSize_ = tilingData.sliceSize;
        numSlice_ = tilingData.sliceNum;
        tailSize_ = tilingData.tailSliceSize;
        uint32_t numRow = tilingData.numRow;
        uint32_t rowWork = numRow / numCore_;
        uint32_t rowRes = numRow - rowWork * numCore_;
        if (AscendC::GetBlockIdx() < rowRes) {
            rowWork_ = rowWork + 1;
        } else {
            rowWork_ = rowWork;
        }
        gmOffset_ = (rowWork + 1) * numCol_;
        
        // for load balancing.
        uint64_t gmExtraOffset = AscendC::GetBlockIdx() - rowRes > 0 ? (AscendC::GetBlockIdx() - rowRes) * numCol_ : 0;
        uint64_t gmTotalOffset = AscendC::GetBlockIdx() * gmOffset_ - gmExtraOffset;
        gmX_.SetGlobalBuffer((__gm__ T *)x + gmTotalOffset);
        gmResIn_.SetGlobalBuffer((__gm__ T *)r + gmTotalOffset);
        gmGamma_.SetGlobalBuffer((__gm__ T *)g);
        gmY_.SetGlobalBuffer((__gm__ int8_t *)y + gmTotalOffset);
        gmResOut_.SetGlobalBuffer((__gm__ T *)res + gmTotalOffset);

        // double buffer is used in the case of small shape.
        if constexpr (FastComputeMode) {
            pipe->InitBuffer(fp16BufX_, DOUBLE_BUFFER_NUM * sliceSize_ * sizeof(T));
            pipe->InitBuffer(fp16BufResIn_, DOUBLE_BUFFER_NUM * sliceSize_ * sizeof(T));
        } else {
            pipe->InitBuffer(fp16BufX_, BUFFER_NUM * sliceSize_ * sizeof(T));
            pipe->InitBuffer(fp16BufResIn_, BUFFER_NUM * sliceSize_ * sizeof(T));
        }
        pipe->InitBuffer(fp16BufGamma_, sliceSize_ * sizeof(T));
        pipe->InitBuffer(int8BufY_, sliceSize_ * sizeof(int8_t));
        pipe->InitBuffer(fp16BufResOut_, sliceSize_ * sizeof(T));
        pipe->InitBuffer(fp32BufXy_, sliceSize_ * sizeof(float));
        pipe->InitBuffer(calcBuf_, sliceSize_ * sizeof(float));

        GetScaleAndOffset<T>(inputScale_, inputOffset_, scale, offset, calcBuf_);
    }

    __aicore__ inline void Launch()
    {
        if constexpr (FastComputeMode) {
            FastCompute();
        } else {
            SliceCompute();
        }
    }

private:
    __aicore__ inline void SliceCompute()
    {
        for (uint64_t pid = 0; pid < rowWork_; pid++) {
            uint64_t rowOffset = pid * numCol_;
            float squareSum = 0.0f;
            for (uint64_t sid = 0; sid < numSlice_; sid++) {
                uint64_t colOffset = rowOffset + sid * sliceSize_;
                uint32_t eleNum = (sid == (numSlice_ - 1)) ? tailSize_ : sliceSize_;
                AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);

                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
                squareSum += ComputeSquareSum(colOffset, eleNum);
            }

            float rms = sqrt(avgFactor_ * squareSum + epsilon_);
            float reciprocalOfRms = 1 / rms;

            for (uint64_t sid = 0; sid < numSlice_; sid++) {
                uint64_t sliceOffset = sid * sliceSize_;
                uint64_t totalOffset = rowOffset + sliceOffset;
                uint32_t eleNum = (sid == (numSlice_ - 1)) ? tailSize_ : sliceSize_;
                AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);

                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
                ComputeNorm(reciprocalOfRms, totalOffset, sliceOffset, eleNum);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            }
        }
    }

    __aicore__ inline void FastCompute()
    {
        event_t eventCur;
        AscendC::LocalTensor<T> g = fp16BufGamma_.Get<T>();
        DataCopy(g, gmGamma_[0], numCol_);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);  // Synchronised EventId between ping and pong

        for (uint64_t pid = 0; pid < rowWork_; pid++) {
            eventCur = pid % DOUBLE_BUFFER_NUM == 0 ? EVENT_ID0: EVENT_ID1;
            uint64_t offset = pid * numCol_;
            uint64_t bufOffset = (pid % DOUBLE_BUFFER_NUM) * numCol_;

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventCur);
            CopyInAll(offset, numCol_, bufOffset);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventCur);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventCur);
            Compute(numCol_, bufOffset);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventCur);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventCur);
            CopyOutAll(offset, numCol_);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventCur);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
    }

private:
    __aicore__ inline float ComputeSquareSum(uint64_t offset, uint32_t numel)
    {
        AscendC::LocalTensor<T> xIn = fp16BufX_.Get<T>();
        AscendC::LocalTensor<T> resIn = fp16BufResIn_.Get<T>();
        AscendC::LocalTensor<float> fp32Xy = fp32BufXy_.Get<float>();
        AscendC::LocalTensor<float> fp32Tmp = calcBuf_.Get<float>();

        DataCopy(xIn, gmX_[offset], numel);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        DataCopy(resIn, gmResIn_[offset], numel);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        CastFrom16To32(fp32Xy, xIn, numel);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
        CastFrom16To32(fp32Tmp, resIn, numel);

        ComputeResidualAdd(fp32Xy, fp32Xy, fp32Tmp, numel);

        return ComputeSliceSquareSum(fp32Xy, fp32Tmp, fp32Tmp, numel);
    }

    __aicore__ inline void ComputeNorm(float reciprocalOfRms, uint64_t totalOffset,
                                       uint64_t sliceOffset, uint32_t numel)
    {
        AscendC::LocalTensor<T> fp16X = fp16BufX_.Get<T>();
        AscendC::LocalTensor<T> fp16ResIn = fp16BufResIn_.Get<T>();
        AscendC::LocalTensor<T> fp16G = fp16BufGamma_.Get<T>();

        AscendC::LocalTensor<T> fp16Y = fp16BufResOut_.Get<T>();
        AscendC::LocalTensor<int8_t> int8Y = int8BufY_.Get<int8_t>();

        AscendC::LocalTensor<float> fp32Xy = fp32BufXy_.Get<float>();
        AscendC::LocalTensor<float> fp32Tmp = calcBuf_.Get<float>();
        AscendC::LocalTensor<half> fp16Tmp = calcBuf_.Get<half>();

        DataCopy(fp16X, gmX_[totalOffset], numel);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        DataCopy(fp16ResIn, gmResIn_[totalOffset], numel);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
        DataCopy(fp16G, gmGamma_[sliceOffset], numel);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        CastFrom16To32(fp32Xy, fp16X, numel);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
        CastFrom16To32(fp32Tmp, fp16ResIn, numel);
        ComputeResidualAdd(fp32Xy, fp32Xy, fp32Tmp, numel);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
        ComputeRmsNorm(fp32Xy, fp32Xy, reciprocalOfRms, fp16G, fp32Tmp, fp16Y, numel);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);

        ComputeHighPrecisionFp32ToI8Quant(int8Y, fp32Xy, fp16Tmp, inputScale_, inputOffset_, quantMin_, numel);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        DataCopy(gmResOut_[totalOffset], fp16Y, numel);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
        DataCopy(gmY_[totalOffset], int8Y, numel);
    }

    __aicore__ inline void CopyInAll(uint64_t offset, uint32_t numel, uint64_t bufOffset)
    {
        AscendC::LocalTensor<T> xIn = fp16BufX_.Get<T>();
        AscendC::LocalTensor<T> resIn = fp16BufResIn_.Get<T>();
        DataCopy(xIn[bufOffset], gmX_[offset], numel);
        DataCopy(resIn[bufOffset], gmResIn_[offset], numel);
    }

    __aicore__ inline void Compute(uint32_t numel, uint64_t bufOffset)
    {
        AscendC::LocalTensor<T> x = fp16BufX_.Get<T>();
        AscendC::LocalTensor<T> fp16X = x[bufOffset];
        AscendC::LocalTensor<T> res = fp16BufResIn_.Get<T>();
        AscendC::LocalTensor<T> fp16ResIn = res[bufOffset];
        AscendC::LocalTensor<T> g = fp16BufGamma_.Get<T>();
        
        AscendC::LocalTensor<int8_t> int8Y = int8BufY_.Get<int8_t>();
        AscendC::LocalTensor<T> fp16Y = fp16BufResOut_.Get<T>();

        AscendC::LocalTensor<float> fp32Xy = fp32BufXy_.Get<float>();
        AscendC::LocalTensor<float> fp32Tmp = calcBuf_.Get<float>();
        AscendC::LocalTensor<half> fp16Tmp = calcBuf_.Get<half>();

        CastFrom16To32(fp32Xy, fp16X, numel);
        CastFrom16To32(fp32Tmp, fp16ResIn, numel);
        ComputeResidualAdd(fp32Xy, fp32Xy, fp32Tmp, numel);

        float squareSum = ComputeSliceSquareSum(fp32Xy, fp32Tmp, fp32Tmp, numel);
        float rms = sqrt(squareSum * avgFactor_ + epsilon_);
        float reciprocalOfRms = 1 / rms;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);  // Synchronised EventId between ping and pong

        ComputeRmsNorm(fp32Xy, fp32Xy, reciprocalOfRms, g, fp32Tmp, fp16Y, numel);
        ComputeHighPrecisionFp32ToI8Quant(int8Y, fp32Xy, fp16Tmp, inputScale_, inputOffset_, quantMin_, numel);
    }

    __aicore__ inline void CopyOutAll(uint64_t offset, uint32_t numel)
    {
        AscendC::LocalTensor<T> resOut = fp16BufResOut_.Get<T>();
        AscendC::LocalTensor<int8_t> int8Y = int8BufY_.Get<int8_t>();
        DataCopy(gmResOut_[offset], resOut, numel);
        DataCopy(gmY_[offset], int8Y, numel);
    }

private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16BufX_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16BufResIn_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16BufGamma_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> int8BufY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16BufResOut_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32BufXy_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf_;
    AscendC::GlobalTensor<T> gmX_;
    AscendC::GlobalTensor<T> gmResIn_;
    AscendC::GlobalTensor<T> gmGamma_;
    AscendC::GlobalTensor<int8_t> gmY_;
    AscendC::GlobalTensor<T> gmResOut_;
    uint32_t numCore_{0};     // 一共激活多少AICORE
    uint32_t numCol_{0};      // 输入的列数
    uint32_t rowWork_{0};     // 每个核处理的行数
    uint64_t gmOffset_{0};    // GM数据起始位置偏移量
    float avgFactor_{1.0};    // numCol_的倒数
    float inputScale_{1.0};   // 非对称量化系数
    float inputOffset_{0};    // 非对称量化偏移适配高精度
    float epsilon_{1e-12f};    // norm平滑参数
    half quantMin_ = -128;
    uint32_t sliceSize_{0};  // 每一行切分的大小
    int32_t numSlice_{0};
    int32_t tailSize_{0};
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *pTilingdata,
                                      AsdOps::RmsNormQuantCommonTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->numCore = (*(const __gm__ uint32_t *)(pTilingdata + 0));
    tilingdata->numCol = (*(const __gm__ uint32_t *)(pTilingdata + 4));
    tilingdata->numRow = (*(const __gm__ uint32_t *)(pTilingdata + 8));
    tilingdata->avgFactor = (*(const __gm__ uint32_t *)(pTilingdata + 12));
    tilingdata->epsilon = (*(const __gm__ uint32_t *)(pTilingdata + 16));
    tilingdata->quantMin = (*(const __gm__ float *)(pTilingdata + 20));
    tilingdata->sliceSize = (*(const __gm__ uint32_t *)(pTilingdata + 24));
    tilingdata->sliceNum = (*(const __gm__ uint32_t *)(pTilingdata + 28));
    tilingdata->tailSliceSize = (*(const __gm__ uint32_t *)(pTilingdata + 32));
    tilingdata->maxCoreNum = (*(const __gm__ uint32_t *)(pTilingdata + 36));
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdataInUb = nullptr;
    CopyGmTilingToUb(tilingdataInUb, pTilingdata, sizeof(AsdOps::RmsNormQuantCommonTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->numCore = (*(__ubuf__ uint32_t *)(tilingdataInUb + 0));
    tilingdata->numCol = (*(__ubuf__ uint32_t *)(tilingdataInUb + 4));
    tilingdata->numRow = (*(__ubuf__ uint32_t *)(tilingdataInUb + 8));
    tilingdata->avgFactor = (*(__ubuf__ uint32_t *)(tilingdataInUb + 12));
    tilingdata->epsilon = (*(__ubuf__ uint32_t *)(tilingdataInUb + 16));
    tilingdata->quantMin = (*(__ubuf__ float *)(tilingdataInUb + 20));
    tilingdata->sliceSize = (*(__ubuf__ uint32_t *)(tilingdataInUb + 24));
    tilingdata->sliceNum = (*(__ubuf__ uint32_t *)(tilingdataInUb + 28));
    tilingdata->tailSliceSize = (*(__ubuf__ uint32_t *)(tilingdataInUb + 32));
    tilingdata->maxCoreNum = (*(__ubuf__ uint32_t *)(tilingdataInUb + 36));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

#define GET_TILING_DATA(tilingData, tilingArg)                                                                       \
    AsdOps::RmsNormQuantCommonTilingData tilingData;                                                                  \
    InitTilingData((tilingArg), &(tilingData))

extern "C" __global__ __aicore__ void rms_post_norm_quant(GM_ADDR x, GM_ADDR r, GM_ADDR g, GM_ADDR scale,
                                                          GM_ADDR offset, GM_ADDR y, GM_ADDR res, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(100)) { // fp16, scale, offset, no slice
        RmsPostNormQuant<half, true> kernel(&pipe, x, r, g, scale, offset, y, res, tilingData);
        kernel.Launch();
    }
    if (TILING_KEY_IS(101)) { // fp16, scale, offset, use slice
        RmsPostNormQuant<half, false> kernel(&pipe, x, r, g, scale, offset, y, res, tilingData);
        kernel.Launch();
    }
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if (TILING_KEY_IS(110)) { // bf16, scale, offset, no slice
        RmsPostNormQuant<bfloat16_t, true> kernel(&pipe, x, r, g, scale, offset, y, res, tilingData);
        kernel.Launch();
    }
    if (TILING_KEY_IS(111)) { // bf16, scale, offset, use slice
        RmsPostNormQuant<bfloat16_t, false> kernel(&pipe, x, r, g, scale, offset, y, res, tilingData);
        kernel.Launch();
    }
#endif
}
