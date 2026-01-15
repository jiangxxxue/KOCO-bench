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

static constexpr uint32_t BUFFER_NUM = 1;       // split the UB to 2 equal part to enable ping-pong techniques.
static constexpr uint32_t DOUBLE_BUFFER = 2;     // split the UB to 2 equal part to enable ping-pong techniques.
static constexpr uint32_t BUF_FACTOR = 2;       // 1(g) + 1(workspace) = 2
static constexpr uint32_t OFFSET_GAMMA = 0;     // the offset of gamma is 0
static constexpr uint32_t OFFSET_WORKSPACE = 1; // the offset of workspace is 1
static constexpr uint32_t GEMMA_ON = 1;         // 开启gemmamode
static constexpr uint32_t GEMMA_OFF = 0;        // 关闭gemmamode
static constexpr uint32_t PRE_ON = 0;           // 开启precisionmode
static constexpr uint32_t PRE_OFF = 1;          // 关闭precisionmode，开始performancemode
static constexpr uint32_t NCT_MAX_SIZE = 8;     // NCT 相关数组的最大长度
static constexpr uint32_t ALIGN_SIZE = 16;      // datacopy搬运对齐数

using AscendC::HardEvent;

template<typename T, uint32_t gemmaMode, uint32_t precisionMode>
class RmsNormShort {
public:
    __aicore__ inline RmsNormShort(__gm__ uint8_t *x, __gm__ uint8_t *g, __gm__ uint8_t *y,
                                   AsdOps::RmsNormCommonTilingData &tilingData)
    {
        xDimNum_ = tilingData.xDimNum;
        for (size_t i = 0; i < xDimNum_; ++ i) {
            xStrides_[i] = tilingData.xStrides[i];
        }
        xOffset_ = tilingData.xOffset;
        
        avgFactor_ = tilingData.avgFactor;
        numCol_ = tilingData.numCol;
        numCore_ = tilingData.numCore;
        precisionMode_ = tilingData.precisionMode;
        epsilon_ = tilingData.epsilon;
        gemmaMode_ = tilingData.gemmaMode;
        sliceSize_ = tilingData.sliceSize;
        
        uint32_t numRow = tilingData.numRow;
        uint32_t rowWork = (numRow + numCore_ - 1) / numCore_;
        if (AscendC::GetBlockIdx() != numCore_ - 1) {
            rowWork_ = rowWork;
        } else {
            rowWork_ = numRow - (numCore_ - 1) * rowWork;
        }

        if (xDimNum_ > 0) {
            isNCT_ = true;
        }

        actNumCol_ = (numCol_ + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE;

        // GM数据起始位置偏移量
        uint64_t gmOffset = static_cast<uint64_t>(rowWork) * actNumCol_;

        if (isNCT_) {
            // 以 xStrides_[xDimNum_ - 2] 为一行的长度
            gmOffset = static_cast<uint64_t>(rowWork) * xStrides_[xDimNum_ - 2];
            gmX_.SetGlobalBuffer((__gm__ T *)x + xOffset_ + (AscendC::GetBlockIdx() * gmOffset));
            gmY_.SetGlobalBuffer((__gm__ T *)y + xOffset_ + (AscendC::GetBlockIdx() * gmOffset));
        } else {
            gmX_.SetGlobalBuffer((__gm__ T *)x + AscendC::GetBlockIdx() * gmOffset);
            gmY_.SetGlobalBuffer((__gm__ T *)y + AscendC::GetBlockIdx() * gmOffset);
        }
        gmG_.SetGlobalBuffer((__gm__ T *)g);
        pipe.InitBuffer(fp16QueX_, DOUBLE_BUFFER, actNumCol_ * sizeof(T));
        pipe.InitBuffer(fp16QueG_, BUFFER_NUM, actNumCol_ * sizeof(T));
        pipe.InitBuffer(fp16QueY_, BUFFER_NUM, actNumCol_ * sizeof(T));
        pipe.InitBuffer(fp32BufG_, actNumCol_ * sizeof(float));
        pipe.InitBuffer(fp32BufXy_, actNumCol_ * sizeof(float));
        pipe.InitBuffer(calcBuf_, BUF_FACTOR * actNumCol_ * sizeof(float));
#if __CCE_AICORE__ == 300
        pipe.InitBuffer(workLocalBuf_, GetReduceSumWorkLocalSize<float>(actNumCol_) * sizeof(float));
#endif
    }

    __aicore__ inline void Launch()
    {
        AscendC::LocalTensor<float> fp32Xy = fp32BufXy_.Get<float>();
        CopyInAndCastF32(fp32Xy, gmX_, fp16QueX_, 0, actNumCol_);
        CopyIn(gmG_, fp16QueG_, 0, actNumCol_);
        AscendC::LocalTensor<float> buf = calcBuf_.Get<float>();
        AscendC::LocalTensor<float> work = buf[OFFSET_WORKSPACE * actNumCol_];
#if __CCE_AICORE__ == 300
        AscendC::LocalTensor<float> workLocal = workLocalBuf_.Get<float>();
        float rms = sqrt(ComputeSliceSquareSum(fp32Xy, work, workLocal, numCol_) * avgFactor_ + epsilon_);
#else
        float rms = sqrt(ComputeSliceSquareSum(fp32Xy, work, work, numCol_) * avgFactor_ + epsilon_);
#endif
        AscendC::LocalTensor<T> gamma = fp16QueG_.DeQue<T>();
        AscendC::LocalTensor<float> fp32G = fp32BufG_.Get<float>();
        CastGAndIsGemmaMode<T, gemmaMode>(fp32G, gamma, numCol_);
        AscendC::LocalTensor<T> fp16Y = fp16QueY_.AllocTensor<T>();
        ComputeRmsNormFast<T, precisionMode>(fp16Y, fp32Xy, rms, gamma, numCol_, work, fp32G);
        fp16QueY_.EnQue(fp16Y);
        
        CopyOut(gmY_, fp16QueY_, 0, actNumCol_);
        
        for (uint64_t pid = 1; pid < rowWork_; pid++) {
            uint64_t offset = pid * numCol_;
            if (isNCT_) {
                // 以 xStrides_[xDimNum_ - 2] 为一行的长度
                offset = pid * xStrides_[xDimNum_ - 2];  // 计算GM实际偏移量
            }
            Compute(offset, gamma, fp32G);
        }
        fp16QueG_.FreeTensor(gamma);
    }

private:
    __aicore__ inline void Compute(uint64_t offset, const AscendC::LocalTensor<T> &gamma,
        const AscendC::LocalTensor<float> &fp32G)
    {
        AscendC::LocalTensor<float> fp32Xy = fp32BufXy_.Get<float>();
        CopyInAndCastF32(fp32Xy, gmX_, fp16QueX_, offset, actNumCol_);
        
        AscendC::LocalTensor<float> buf = calcBuf_.Get<float>();
        AscendC::LocalTensor<float> work = buf[OFFSET_WORKSPACE * actNumCol_];
#if __CCE_AICORE__ == 300
        AscendC::LocalTensor<float> workLocal = workLocalBuf_.Get<float>();
        float squareSum = ComputeSliceSquareSum(fp32Xy, work, workLocal, numCol_);
#else
        float squareSum = ComputeSliceSquareSum(fp32Xy, work, work, numCol_);
#endif
        float rms = sqrt(squareSum * avgFactor_ + epsilon_);
        AscendC::LocalTensor<T> fp16Y = fp16QueY_.AllocTensor<T>();
        ComputeRmsNormFast<T, precisionMode>(fp16Y, fp32Xy, rms, gamma, numCol_, work, fp32G);
        fp16QueY_.EnQue(fp16Y);
 
        CopyOut(gmY_, fp16QueY_, offset, actNumCol_);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, DOUBLE_BUFFER> fp16QueX_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> fp16QueG_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> fp16QueY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32BufXy_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32BufG_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf_;
#if __CCE_AICORE__ == 300
    AscendC::TBuf<AscendC::TPosition::VECCALC> workLocalBuf_;
#endif
    AscendC::GlobalTensor<T> gmX_;
    AscendC::GlobalTensor<T> gmG_;
    AscendC::GlobalTensor<T> gmY_;
    uint32_t numCore_{0};  // 一共激活多少AICORE
    uint32_t numCol_{0};   // 输入的列数
    uint32_t actNumCol_{0}; // 存在元素个数非16对齐，需要向上取整
    uint32_t rowWork_{0};  // 需要计算多少行
    uint32_t sliceSize_{0};  // 每一行切分的大小
    float avgFactor_{1.0f}; // numCol_的倒数
    float epsilon_{1e-12f}; // norm平滑参数
    uint32_t precisionMode_{0};
    uint32_t gemmaMode_{0};
    bool isNCT_{false};
    uint32_t xStrides_[NCT_MAX_SIZE];     // 非连续tensor x步长
    uint32_t xDimNum_{0};                 // 非连续tensor x维度数
    uint32_t xOffset_{0};                 // 非连续tensor x偏移量
};

template<typename T>
class RmsNormLong {
public:
    __aicore__ inline RmsNormLong(__gm__ uint8_t *x, __gm__ uint8_t *g, __gm__ uint8_t *y,
                                  AsdOps::RmsNormCommonTilingData &tilingData)
    {
        xDimNum_ = tilingData.xDimNum;
        for (size_t i = 0; i < xDimNum_; ++ i) {
            xStrides_[i] = tilingData.xStrides[i];
        }
        xOffset_ = tilingData.xOffset;

        avgFactor_ = tilingData.avgFactor;
        numCol_ = tilingData.numCol;
        numCore_ = tilingData.numCore;
        precisionMode_ = tilingData.precisionMode;
        epsilon_ = tilingData.epsilon;
        gemmaMode_ = tilingData.gemmaMode;
        sliceSize_ = tilingData.sliceSize;

        uint32_t numRow = tilingData.numRow;
        uint32_t rowWork = (numRow + numCore_ - 1) / numCore_;
        if (AscendC::GetBlockIdx() != numCore_ - 1) {
            rowWork_ = rowWork;
        } else {
            rowWork_ = numRow - (numCore_ - 1) * rowWork;
        }

        if (xDimNum_ > 0) {
            isNCT_ = true;
        }

        uint64_t gmOffset = static_cast<uint64_t>(rowWork) * numCol_;   // GM数据起始位置偏移量
        if (isNCT_) {
            // 以 xStrides_[xDimNum_ - 2] 为一行的长度
            gmOffset = static_cast<uint64_t>(rowWork) * xStrides_[xDimNum_ - 2];
        }

        numSlice_ = (numCol_ + sliceSize_ - 1) / sliceSize_;
        tailSize_ = numCol_ - (numSlice_ - 1) * sliceSize_;
        
        if (isNCT_) {
            gmX_.SetGlobalBuffer((__gm__ T *)x + AscendC::GetBlockIdx() * gmOffset + xOffset_);
            gmY_.SetGlobalBuffer((__gm__ T *)y + AscendC::GetBlockIdx() * gmOffset + xOffset_);
        } else {
            gmX_.SetGlobalBuffer((__gm__ T *)x + AscendC::GetBlockIdx() * gmOffset);
            gmY_.SetGlobalBuffer((__gm__ T *)y + AscendC::GetBlockIdx() * gmOffset);
        }
        gmG_.SetGlobalBuffer((__gm__ T *)g);
        pipe.InitBuffer(fp16QueX_, BUFFER_NUM, sizeof(T) * sliceSize_);
        pipe.InitBuffer(fp16QueG_, BUFFER_NUM, sliceSize_ * sizeof(T));
        pipe.InitBuffer(fp16QueY_, BUFFER_NUM, sizeof(T) * sliceSize_);
        pipe.InitBuffer(fp32BufXy_, sizeof(float) * sliceSize_);
        pipe.InitBuffer(calcBuf_, BUF_FACTOR * sizeof(float) * sliceSize_);
#if __CCE_AICORE__ == 300
        pipe.InitBuffer(workLocalBuf_, GetReduceSumWorkLocalSize<float>(sliceSize_) * sizeof(float));
#endif
    }

    __aicore__ inline void Launch()
    {
        for (uint64_t pid = 0; pid < rowWork_; pid++) {
            uint64_t rowOffset = pid * numCol_;
            float squareSum = 0.0f;
            if (isNCT_) {
                // 以 xStrides_[xDimNum_ - 2] 为一行的长度
                rowOffset = pid * xStrides_[xDimNum_ - 2];
            }

            for (uint32_t sid = 0; sid < numSlice_; sid++) {
                uint64_t colOffset = rowOffset + sid * sliceSize_;
                uint32_t eleNum = (sid == (numSlice_ - 1)) ? tailSize_ : sliceSize_;
                squareSum += ComputeSquareSum(colOffset, eleNum);
            }

            float rms = sqrt(avgFactor_ * squareSum + epsilon_);
            for (uint64_t sid = 0; sid < numSlice_; sid++) {
                uint64_t sliceOffset = sid * sliceSize_;
                uint64_t colOffset = rowOffset + sliceOffset;
                uint32_t eleNum = (sid == (numSlice_ - 1)) ? tailSize_ : sliceSize_;
                ComputeNorm(rms, colOffset, sliceOffset, eleNum);
            }
        }
    }

private:
    __aicore__ inline float ComputeSquareSum(uint64_t offset, uint32_t numel)
    {
        AscendC::LocalTensor<float> fp32Xy = fp32BufXy_.Get<float>();
        AscendC::LocalTensor<float> buf = calcBuf_.Get<float>();
        AscendC::LocalTensor<float> work = buf[OFFSET_WORKSPACE * sliceSize_];
        
        uint32_t alignNumel = (numel + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE;
        CopyInAndCastF32(fp32Xy, gmX_, fp16QueX_, offset, alignNumel);

#if __CCE_AICORE__ == 300
        AscendC::LocalTensor<float> workLocal = workLocalBuf_.Get<float>();
        return ComputeSliceSquareSum(fp32Xy, work, workLocal, numel);
#else
        return ComputeSliceSquareSum(fp32Xy, work, work, numel);
#endif
    }

    __aicore__ inline void ComputeNorm(float rms, uint64_t totalOffset, uint64_t sliceOffset, uint32_t numel)
    {
        AscendC::LocalTensor<float> fp32Xy = fp32BufXy_.Get<float>();
        AscendC::LocalTensor<float> buf = calcBuf_.Get<float>();
        AscendC::LocalTensor<float> work = buf[OFFSET_WORKSPACE * sliceSize_];
        AscendC::LocalTensor<T> fp16Y = fp16QueY_.AllocTensor<T>();
        
        uint32_t alignNumel = (numel + ALIGN_SIZE - 1) / ALIGN_SIZE * ALIGN_SIZE;
        CopyInAndCastF32(fp32Xy, gmX_, fp16QueX_, totalOffset, alignNumel);
        CopyIn(gmG_, fp16QueG_, sliceOffset, alignNumel);

        AscendC::LocalTensor<T> gamma = fp16QueG_.DeQue<T>();

        ComputeRmsNorm(fp16Y, fp32Xy, rms, gamma, numel, precisionMode_, gemmaMode_, work);

        fp16QueG_.FreeTensor(gamma);

        fp16QueY_.EnQue(fp16Y);
        CopyOut(gmY_, fp16QueY_, totalOffset, alignNumel);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> fp16QueX_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> fp16QueG_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> fp16QueY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32BufXy_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf_;
#if __CCE_AICORE__ == 300
    AscendC::TBuf<AscendC::TPosition::VECCALC> workLocalBuf_;
#endif
    AscendC::GlobalTensor<T> gmX_;
    AscendC::GlobalTensor<T> gmG_;
    AscendC::GlobalTensor<T> gmY_;
    uint32_t numCore_{0};  // 一共激活多少AICORE
    uint32_t numCol_{0};   // 输入的列数
    uint32_t rowWork_{0};  // 需要计算多少
    uint32_t sliceSize_{0};  // 每一行切分的大小
    int32_t numSlice_{0};
    int32_t tailSize_{0};
    float avgFactor_{1.0f}; // numCol_的倒数
    float epsilon_{1e-12f}; // norm平滑参数
    uint32_t precisionMode_{0};
    uint32_t gemmaMode_{0};
    bool isNCT_{false};
    uint32_t xStrides_[NCT_MAX_SIZE];     // 非连续tensor x步长
    uint32_t xDimNum_{0};                 // 非连续tensor x维度数
    uint32_t xOffset_{0};                 // 非连续tensor x偏移量
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AsdOps::RmsNormCommonTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220) || (__CCE_AICORE__ == 300)
    tilingdata->numCore = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->numCol = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->numRow = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
    tilingdata->avgFactor = (*(const __gm__ float *)(p_tilingdata + 12));
    tilingdata->epsilon = (*(const __gm__ float *)(p_tilingdata + 16));
    tilingdata->sliceSize = (*(const __gm__ uint32_t *)(p_tilingdata + 20));
    tilingdata->mode = (*(const __gm__ uint32_t *)(p_tilingdata + 24));
    tilingdata->quantMin = (*(const __gm__ float *)(p_tilingdata + 28));
    tilingdata->precisionMode = (*(const __gm__ uint32_t *)(p_tilingdata + 32));
    tilingdata->gemmaMode = (*(const __gm__ uint32_t *)(p_tilingdata + 36));
    for (size_t i = 0; i < NCT_MAX_SIZE; ++ i) {
        tilingdata->xStrides[i] = (*(const __gm__ uint32_t *)(4 * i + p_tilingdata + 40));
    }
    tilingdata->xOffset = (*(const __gm__ uint32_t *)(p_tilingdata + 72));
    tilingdata->xDimNum = (*(const __gm__ uint32_t *)(p_tilingdata + 76));
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AsdOps::RmsNormCommonTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->numCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->numCol = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    tilingdata->numRow = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 8));
    tilingdata->avgFactor = (*(__ubuf__ float *)(tilingdata_in_ub + 12));
    tilingdata->epsilon = (*(__ubuf__ float *)(tilingdata_in_ub + 16));
    tilingdata->sliceSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 20));
    tilingdata->mode = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 24));
    tilingdata->quantMin = (*(__ubuf__ float *)(tilingdata_in_ub + 28));
    tilingdata->precisionMode = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 32));
    tilingdata->gemmaMode = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 36));
    for (size_t i = 0; i < NCT_MAX_SIZE; ++ i) {
        tilingdata->xStrides[i] = (*(__ubuf__ uint32_t *)(4 * i + tilingdata_in_ub + 40));
    }
    tilingdata->xOffset = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 72));
    tilingdata->xDimNum = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 76));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

#define GET_TILING_DATA(tilingData, tilingArg)                                                                       \
    AsdOps::RmsNormCommonTilingData tilingData;                                                                       \
    InitTilingData(tilingArg, &(tilingData))

extern "C" __global__ __aicore__ void rms_norm(GM_ADDR x, GM_ADDR g, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(11100)) {  // fp16,gemmamode = 1,precisionmode = 0
        RmsNormShort<half, GEMMA_ON, PRE_ON> kernel(x, g, y, tilingData);
        kernel.Launch();
    }
    if (TILING_KEY_IS(10100)){  // fp16,gemmode = 0,precisionmode = 0
        RmsNormShort<half, GEMMA_OFF, PRE_ON> kernel(x, g, y, tilingData);
        kernel.Launch();
    }
    if (TILING_KEY_IS(10000)){  // fp16,gemmode = 0,precisionmode = 1, 启动高性能模式，高性能模式不支持BF16
        RmsNormShort<half, GEMMA_OFF, PRE_OFF> kernel(x, g, y, tilingData);
        kernel.Launch();
    }
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if (TILING_KEY_IS(11110)){  // BF16,gemmamode = 1,precisionmode = 0
        RmsNormShort<bfloat16_t, GEMMA_ON, PRE_ON> kernel(x, g, y, tilingData);
        kernel.Launch();
    }
    if (TILING_KEY_IS(10110)){  // BF16,gemmamode = 0,precisionmode = 0
        RmsNormShort<bfloat16_t, GEMMA_OFF, PRE_ON> kernel(x, g, y, tilingData);
        kernel.Launch();
    }
#endif
    // Long tail
    if (TILING_KEY_IS(110000)) {
        RmsNormLong<half> kernel(x, g, y, tilingData);
        kernel.Launch();
    }
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if (TILING_KEY_IS(110010)) {
        RmsNormLong<bfloat16_t> kernel(x, g, y, tilingData);
        kernel.Launch();
    }
#endif
}
