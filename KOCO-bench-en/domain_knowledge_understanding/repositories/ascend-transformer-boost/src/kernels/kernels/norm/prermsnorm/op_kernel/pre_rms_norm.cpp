/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "kernels/utils/kernel/kernel_utils.h"
#include "kernels/norm/postrmsnorm/tiling/post_rms_norm_tiling_data.h"
#include "kernels/norm/rmsnormforward/op_kernel/rms_norm_base.h"
#include "kernels/norm/common/common_pre_post/comm_pre_post.h"

using AscendC::HardEvent;

static constexpr uint32_t DOUBLE_BUFFER_NUM = 2;       // split the UB to 2 equal part to enable ping-pong techniques.

template<typename T, bool HAS_BIAS>
class PreRmsNormDoubleBuffer {
public:
    __aicore__ inline PreRmsNormDoubleBuffer(AscendC::TPipe *pipe, __gm__ uint8_t *x, __gm__ uint8_t *bias,
        __gm__ uint8_t *resIn, __gm__ uint8_t *g, __gm__ uint8_t *y, __gm__ uint8_t *resOut,
        AsdOps::PostRmsNormTilingData &tilingData)
    {
        uint32_t numRow = tilingData.numRow;
        numCore_ = tilingData.numCore;
        numCol_ = tilingData.numCol;
        avgFactor_ = *reinterpret_cast<float *>(&tilingData.avgFactor);
        epsilon_ = *reinterpret_cast<float *>(&tilingData.epsilon);
        sliceSize_ = tilingData.sliceSize;
        precisionMode_ = tilingData.precisionMode;
        uint32_t rowWork = (numRow + numCore_ - 1) / numCore_;

        if (AscendC::GetBlockIdx() != numCore_ - 1) {
            rowWork_ = rowWork;
        } else {
            rowWork_ = numRow - (numCore_ - 1) * rowWork;
        }
        gmOffset_ = static_cast<uint64_t>(rowWork) * numCol_;

        numColAlignFp32_ = (numCol_ + FP32_PER_REPEAT - 1) / FP32_PER_REPEAT * FP32_PER_REPEAT;
        numColAlignFp16_ = (numCol_ + FP16_PER_REPEAT - 1) / FP16_PER_REPEAT * FP16_PER_REPEAT;

        repeatTimes_ = numColAlignFp32_ / FP32_PER_REPEAT;
        alignRepeatOffset_ = repeatTimes_ * FP32_PER_REPEAT;
        tailRepeatNum_ = numColAlignFp32_ - alignRepeatOffset_;

        gmX_.SetGlobalBuffer((__gm__ T *)x + AscendC::GetBlockIdx() * gmOffset_);
        if constexpr (HAS_BIAS) {
            gmBias_.SetGlobalBuffer((__gm__ T *)bias);
        }
        gmResIn_.SetGlobalBuffer((__gm__ T *)resIn + AscendC::GetBlockIdx() * gmOffset_);
        gmGamma_.SetGlobalBuffer((__gm__ T *)g);
        gmY_.SetGlobalBuffer((__gm__ T *)y + AscendC::GetBlockIdx() * gmOffset_);
        gmResOut_.SetGlobalBuffer((__gm__ T *)resOut + AscendC::GetBlockIdx() * gmOffset_);

        pipe->InitBuffer(fp16BufX_, DOUBLE_BUFFER_NUM * sliceSize_ * sizeof(T));
        if constexpr (HAS_BIAS) {
            pipe->InitBuffer(fp16BufBias_, sliceSize_ * sizeof(T));
        }
        pipe->InitBuffer(fp16BufResIn_, DOUBLE_BUFFER_NUM * sliceSize_ * sizeof(T));
        pipe->InitBuffer(fp16BufGamma_, sliceSize_ * sizeof(T));
        pipe->InitBuffer(fp16BufY_, sliceSize_ * sizeof(T));
        pipe->InitBuffer(fp16BufResOut_, sliceSize_ * sizeof(T));
        pipe->InitBuffer(fp32BufXy_, sliceSize_ * sizeof(float));
        pipe->InitBuffer(calcBuf_, sliceSize_ * sizeof(float));
        pipe->InitBuffer(brcbBuf_, AsdOps::BRCB_BYTE);
    }

    __aicore__ inline void Launch()
    {
        event_t eventCur;
        event_t eventCur1;
        AscendC::LocalTensor<T> g = fp16BufGamma_.Get<T>();
        DataCopyCustom<T>(g, gmGamma_[0], numColAlignFp16_);
        if constexpr (HAS_BIAS) {
            AscendC::LocalTensor<T> fp16Bias = fp16BufBias_.Get<T>();
            DataCopyCustom<T>(fp16Bias, gmBias_[0], numColAlignFp16_);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);  // Synchronised EventId between ping and pong
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);  // Synchronised EventId between ping and pong

        for (uint64_t pid = 0; pid < rowWork_; pid++) {
            eventCur = pid % DOUBLE_BUFFER_NUM == 0 ? EVENT_ID0: EVENT_ID1;
            eventCur1 = pid % DOUBLE_BUFFER_NUM == 0 ? EVENT_ID4: EVENT_ID5;
            uint64_t offset = pid * numCol_;
            uint64_t bufOffset = (pid % DOUBLE_BUFFER_NUM) * sliceSize_;

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventCur);
            CopyInAll(offset, numColAlignFp16_, bufOffset);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventCur);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventCur);
            Compute(numCol_, bufOffset, offset, eventCur1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventCur);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventCur);
            CopyOutY(offset, numCol_);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventCur);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
    }

private:
    __aicore__ inline  void Compute(uint32_t numel, uint64_t bufOffset, uint64_t offset, event_t eventCur1)
    {
        AscendC::LocalTensor<T> x = fp16BufX_.Get<T>();
        AscendC::LocalTensor<T> fp16X = x[bufOffset];
        AscendC::LocalTensor<T> resIn = fp16BufResIn_.Get<T>();
        AscendC::LocalTensor<T> fp16ResIn = resIn[bufOffset];
        AscendC::LocalTensor<T> g = fp16BufGamma_.Get<T>();

        AscendC::LocalTensor<T> fp16Y = fp16BufY_.Get<T>();
        AscendC::LocalTensor<T> resOut = fp16BufResOut_.Get<T>();

        AscendC::LocalTensor<float> fp32Xy = fp32BufXy_.Get<float>();
        AscendC::LocalTensor<float> fp32Tmp = calcBuf_.Get<float>();
        AscendC::LocalTensor<float> brcbBuf = brcbBuf_.Get<float>();

        if constexpr (HAS_BIAS) {
            AscendC::LocalTensor<T> fp16Bias = fp16BufBias_.Get<T>();
            AddResAndBias<T>(fp16X, fp16ResIn, fp16Bias, fp32Xy, fp32Tmp, numel);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
            CastFrom32To16(resOut, fp32Xy, numel);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventCur1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventCur1);

            CopyOutRes(offset, numel);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        } else {
            AddResAndCast<T>(fp16X, fp16ResIn, fp32Xy, fp32Tmp, numel);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
            CastFrom32To16(resOut, fp32Xy, numel);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventCur1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventCur1);

            CopyOutRes(offset, numel);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        }

        FigureOutNorm(fp32Tmp, brcbBuf, fp32Xy, avgFactor_, epsilon_, numel, numColAlignFp32_, repeatTimes_,
            alignRepeatOffset_, tailRepeatNum_);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);

        MultiplyGamma(g, fp32Tmp, fp32Xy, fp16Y, numel, numColAlignFp32_, numColAlignFp16_, precisionMode_);
    }

    __aicore__ inline void CopyInAll(uint64_t offset, uint32_t numel, uint64_t bufOffset)
    {
        AscendC::LocalTensor<T> xIn = fp16BufX_.Get<T>();
        AscendC::LocalTensor<T> resIn = fp16BufResIn_.Get<T>();
        DataCopy(xIn[bufOffset], gmX_[offset], numel);
        DataCopy(resIn[bufOffset], gmResIn_[offset], numel);
    }

    __aicore__ inline void CopyOutY(uint64_t offset, uint32_t numel)
    {
        AscendC::LocalTensor<T> fp16Y = fp16BufY_.Get<T>();
        DataCopyCustom<T>(gmY_[offset], fp16Y, numel);
    }

    __aicore__ inline void CopyOutRes(uint64_t offset, uint32_t numel)
    {
        AscendC::LocalTensor<T> resOut = fp16BufResOut_.Get<T>();
        DataCopyCustom<T>(gmResOut_[offset], resOut, numel);
    }

private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16BufX_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16BufBias_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16BufResIn_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16BufGamma_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16BufY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16BufResOut_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32BufXy_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> brcbBuf_;  // for brcb
    AscendC::GlobalTensor<T> gmX_;
    AscendC::GlobalTensor<T> gmBias_;
    AscendC::GlobalTensor<T> gmResIn_;
    AscendC::GlobalTensor<T> gmGamma_;
    AscendC::GlobalTensor<T> gmY_;
    AscendC::GlobalTensor<T> gmResOut_;
    uint32_t numCore_{0};
    uint32_t numCol_{0};
    uint32_t rowWork_{0};
    uint64_t gmOffset_{0};
    uint32_t sliceSize_{0};
    float avgFactor_{1.0f};
    float epsilon_{1e-12f};
    uint32_t numColAlignFp32_{64};
    uint32_t numColAlignFp16_{128};
    uint32_t repeatTimes_{1};
    uint32_t alignRepeatOffset_{0};
    uint32_t tailRepeatNum_{0};
    uint32_t precisionMode_{0};
};

template<typename T, bool HAS_BIAS>
class PreRmsNormShort {
public:
    __aicore__ inline PreRmsNormShort(AscendC::TPipe *pipe, __gm__ uint8_t *x, __gm__ uint8_t *bias,
        __gm__ uint8_t *resIn, __gm__ uint8_t *g, __gm__ uint8_t *y, __gm__ uint8_t *resOut,
        AsdOps::PostRmsNormTilingData &tilingData)
    {
        uint32_t numRow = tilingData.numRow;
        numCore_ = tilingData.numCore;
        numCol_ = tilingData.numCol;
        avgFactor_ = *reinterpret_cast<float *>(&tilingData.avgFactor);
        epsilon_ = *reinterpret_cast<float *>(&tilingData.epsilon);
        sliceSize_ = tilingData.sliceSize;
        precisionMode_ = tilingData.precisionMode;
        uint32_t rowWork = (numRow + numCore_ - 1) / numCore_;

        if (AscendC::GetBlockIdx() != numCore_ - 1) {
            rowWork_ = rowWork;
        } else {
            rowWork_ = numRow - (numCore_ - 1) * rowWork;
        }
        gmOffset_ = static_cast<uint64_t>(rowWork) * numCol_;

        numColAlignFp32_ = (numCol_ + FP32_PER_REPEAT - 1) / FP32_PER_REPEAT * FP32_PER_REPEAT;
        numColAlignFp16_ = (numCol_ + FP16_PER_REPEAT - 1) / FP16_PER_REPEAT * FP16_PER_REPEAT;

        gmX_.SetGlobalBuffer((__gm__ T *)x + AscendC::GetBlockIdx() * gmOffset_);
        if constexpr (HAS_BIAS) {
            gmBias_.SetGlobalBuffer((__gm__ T *)bias);
            pipe->InitBuffer(calcBuf_, NUM_TWO * sliceSize_ * sizeof(float));
            pipe->InitBuffer(fp16BufX_, NUM_FOUR * sliceSize_ * sizeof(T)); // x,resIn,gamma,bias
        } else {
            pipe->InitBuffer(calcBuf_, 1 * sliceSize_ * sizeof(float));
            pipe->InitBuffer(fp16BufX_, NUM_THREE * sliceSize_ * sizeof(T));
        }
        gmGamma_.SetGlobalBuffer((__gm__ T *)g);
        gmResIn_.SetGlobalBuffer((__gm__ T *)resIn + AscendC::GetBlockIdx() * gmOffset_);
        gmY_.SetGlobalBuffer((__gm__ T *)y + AscendC::GetBlockIdx() * gmOffset_);
        gmResOut_.SetGlobalBuffer((__gm__ T *)resOut + AscendC::GetBlockIdx() * gmOffset_);

        pipe->InitBuffer(fp32BufXy_, sliceSize_ * sizeof(float));
        pipe->InitBuffer(fp16BufOut_, sliceSize_ * sizeof(T));

        AscendC::LocalTensor<T> fp16X = fp16BufX_.Get<T>();
        DataCopyCustom<T>(fp16X, gmX_[0], numColAlignFp16_);
        DataCopyCustom<T>(fp16X[sliceSize_], gmResIn_[0], numColAlignFp16_);
        DataCopy(fp16X[sliceSize_ * NUM_TWO], gmGamma_, numColAlignFp16_);
    }

    __aicore__ inline void Launch()
    {
        AscendC::LocalTensor<float> buf = calcBuf_.Get<float>();
        AscendC::LocalTensor<T> fp16X = fp16BufX_.Get<T>();
        AscendC::LocalTensor<T> fp16ResIn = fp16X[sliceSize_];

        if constexpr (HAS_BIAS) {
            AscendC::LocalTensor<T> fp16Bias = fp16X[sliceSize_ * NUM_THREE];
            AscendC::LocalTensor<float> fp32Bias = buf[sliceSize_];
            BiasIn(fp16X, fp16Bias, fp32Bias, gmBias_, numColAlignFp16_);
        }

        uint64_t pid = 0;
        while (pid < rowWork_) {
            uint64_t offset = pid * numCol_;
            if (pid != 0) {
                CopyInXResIn(fp16X, fp16ResIn, gmX_, gmResIn_, offset, numColAlignFp16_);
            }
            AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            Compute(offset);

            AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

            CopyOut(offset, numCol_);

            AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);

            ++pid;
        }
    }

private:
    __aicore__ inline  void Compute(uint32_t offset)
    {
        AscendC::LocalTensor<T> fp16X = fp16BufX_.Get<T>();
        AscendC::LocalTensor<T> fp16ResIn = fp16X[sliceSize_];
        AscendC::LocalTensor<T> fp16Gamma = fp16X[sliceSize_ * NUM_TWO];
        AscendC::LocalTensor<float> fp32ReduceWorkspace = fp16BufOut_.Get<float>();
        AscendC::LocalTensor<float> fp32Xy = fp32BufXy_.Get<float>();
        AscendC::LocalTensor<float> buf = calcBuf_.Get<float>();
        AscendC::LocalTensor<T> outBuf = fp16BufOut_.Get<T>();
        AscendC::LocalTensor<float> sqx = buf[0];

        if constexpr (HAS_BIAS) {
            AscendC::LocalTensor<T> fp16Bias = fp16X[sliceSize_ * NUM_THREE];
            AscendC::LocalTensor<float> bufBias = buf[sliceSize_];
            AddResBiasAndCast<T>(fp16X, fp16ResIn, fp16Bias, fp32Xy, buf, bufBias, numCol_);
        } else {
            AddResAndCast<T>(fp16X, fp16ResIn, fp32Xy, buf, numCol_);
        }

        CastFrom32To16(fp16X, fp32Xy, numCol_);

        AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);

        DataCopyCustom<T>(gmResOut_[offset], fp16X, numCol_);

        AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);

        FigureOutNorm(sqx, fp32Xy, fp32ReduceWorkspace, avgFactor_, epsilon_, numCol_, numColAlignFp32_);

        MultiplyGamma(fp16Gamma, sqx, fp32Xy, outBuf, numCol_, numColAlignFp32_, numColAlignFp16_, precisionMode_);
    }

    __aicore__ inline  void CopyOut(uint32_t offset, uint32_t numel)
    {
        AscendC::LocalTensor<T> outBuf = fp16BufOut_.Get<T>();
        DataCopyCustom<T>(gmY_[offset], outBuf, numel);
    }

private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16BufX_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32BufXy_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16BufOut_;
    AscendC::GlobalTensor<T> gmX_;
    AscendC::GlobalTensor<T> gmBias_;
    AscendC::GlobalTensor<T> gmResIn_;
    AscendC::GlobalTensor<T> gmGamma_;
    AscendC::GlobalTensor<T> gmY_;
    AscendC::GlobalTensor<T> gmResOut_;
    uint32_t numCore_{0};  // 一共激活多少AICORE
    uint32_t numCol_{0};   // 输入的列数
    uint32_t rowWork_{0};  // 需要计算多少行
    uint64_t gmOffset_{0}; // GM数据起始位置偏移量
    uint32_t sliceSize_{0};  // 每一行切分的大小
    float avgFactor_{1.0f}; // num_col_的倒数
    float epsilon_{1e-12f}; // norm平滑参数
    uint32_t numColAlignFp32_{64};
    uint32_t numColAlignFp16_{128};
    uint32_t precisionMode_{0};
};

template<typename T, bool HAS_BIAS>
class PreRmsNormLong {
public:
    __aicore__ inline PreRmsNormLong(AscendC::TPipe *pipe, __gm__ uint8_t *x, __gm__ uint8_t *bias,
        __gm__ uint8_t *resIn, __gm__ uint8_t *g, __gm__ uint8_t *y, __gm__ uint8_t *resOut,
        AsdOps::PostRmsNormTilingData &tilingData)
    {
        uint32_t numRow = tilingData.numRow;
        numCore_ = tilingData.numCore;
        numCol_ = tilingData.numCol;
        precisionMode_ = tilingData.precisionMode;
        avgFactor_ = *reinterpret_cast<float *>(&tilingData.avgFactor);
        epsilon_ = *reinterpret_cast<float *>(&tilingData.epsilon);
        sliceSize_ = tilingData.sliceSize;
        uint32_t rowWork = (numRow + numCore_ - 1) / numCore_;
        if (AscendC::GetBlockIdx() != numCore_ - 1) {
            rowWork_ = rowWork;
        } else {
            rowWork_ = numRow - (numCore_ - 1) * rowWork;
        }
#if __CCE_AICORE__ != 220
        if ((numCol_ % sliceSize_) * sizeof(T) < BLOCK_BYTE && (numCol_ % sliceSize_) != 0) {
            sliceSizeTmp_ = sliceSize_ - ((BLOCK_BYTE / sizeof(T)) - (numCol_ % sliceSize_));
        } else {
            sliceSizeTmp_ = sliceSize_;
        }
#else
        sliceSizeTmp_ = sliceSize_;
#endif
        numSlice_ = (numCol_ + sliceSizeTmp_ - 1) / sliceSizeTmp_;
        tailSize_ = numCol_ - (numSlice_ - 1) * sliceSizeTmp_;
        gmOffset_ = static_cast<uint64_t>(rowWork) * numCol_;
        gmX_.SetGlobalBuffer((__gm__ T *)x + AscendC::GetBlockIdx() * gmOffset_);
        if constexpr (HAS_BIAS) {
            gmBias_.SetGlobalBuffer((__gm__ T *)bias);
            pipe->InitBuffer(calcBuf_, NUM_TWO * sliceSize_ * sizeof(float));
            pipe->InitBuffer(fp16BufX_, NUM_FOUR * sliceSize_ * sizeof(T));

        } else {
            pipe->InitBuffer(calcBuf_, sliceSize_ * sizeof(float));
            pipe->InitBuffer(fp16BufX_, NUM_THREE * sliceSize_ * sizeof(T));
        }
        gmResIn_.SetGlobalBuffer((__gm__ T *)resIn + AscendC::GetBlockIdx() * gmOffset_);
        gmGamma_.SetGlobalBuffer((__gm__ T *)g);
        gmY_.SetGlobalBuffer((__gm__ T *)y + AscendC::GetBlockIdx() * gmOffset_);
        gmResOut_.SetGlobalBuffer((__gm__ T *)resOut + AscendC::GetBlockIdx() * gmOffset_);

        pipe->InitBuffer(fp32BufXy_, sliceSize_ * sizeof(float));
        pipe->InitBuffer(fp16BufOut_, sliceSize_ * sizeof(T));
    }

    __aicore__ inline void Launch()
    {
        AscendC::LocalTensor<T> fp16X = fp16BufX_.Get<T>();
        AscendC::LocalTensor<float> buf = calcBuf_.Get<float>();
        AscendC::LocalTensor<T> fp16ResIn = fp16X[sliceSize_];
        AscendC::LocalTensor<T> fp16Gamma = fp16X[NUM_TWO * sliceSize_];
        uint64_t pid = 0;
        while (pid < rowWork_) {
            uint64_t rowOffset = pid * numCol_;
            uint32_t numEle = sliceSizeTmp_;
            squareSum_ = 0.0f;
            for (uint64_t sid = 0; sid < numSlice_; sid++) {
                uint64_t colOffset = rowOffset + sid * sliceSizeTmp_;
                if ((sid == (numSlice_ - 1)) && (tailSize_ != 0)) {
                    numEle = tailSize_;
                }
                numelAlignFp32_ = (numEle + FP32_PER_REPEAT - 1) / FP32_PER_REPEAT * FP32_PER_REPEAT;
                numelAlignFp16_ = (numEle + FP16_PER_REPEAT - 1) / FP16_PER_REPEAT * FP16_PER_REPEAT;

                CopyInXResIn(fp16X, fp16ResIn, gmX_, gmResIn_, colOffset, numelAlignFp16_);
                if constexpr (HAS_BIAS) {
                    CopyInBias(sid * sliceSizeTmp_);
                }
                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                squareSum_ += ComputeSquareSum(numEle, sid, colOffset);
            }
            numEle = sliceSizeTmp_;
            float factor = avgFactor_ * squareSum_ + epsilon_;
            for (uint64_t sid = 0; sid < numSlice_; sid++) {
                uint64_t colOffset = rowOffset + sid * sliceSizeTmp_;
                if ((sid == (numSlice_ - 1)) && (tailSize_ != 0)) {
                    numEle = tailSize_;
                }
                numelAlignFp32_ = (numEle + FP32_PER_REPEAT - 1) / FP32_PER_REPEAT * FP32_PER_REPEAT;
                numelAlignFp16_ = (numEle + FP16_PER_REPEAT - 1) / FP16_PER_REPEAT * FP16_PER_REPEAT;

                AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);

                CopyInXResIn(fp16X, fp16ResIn, gmX_, gmResIn_, colOffset, numelAlignFp16_);
                CopyInG(fp16Gamma, gmGamma_, sid * sliceSizeTmp_, numelAlignFp16_);
                if constexpr (HAS_BIAS) {
                    CopyInBias(sid * sliceSizeTmp_);
                }

                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

                ComputeNorm(factor, numEle);

                AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

                CopyOut(colOffset, numEle);

                AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
            }
            pid++;
        }
    }

private:

    __aicore__ inline void CopyInBias(uint64_t offset)
    {
        AscendC::LocalTensor<T> fp16X = fp16BufX_.Get<T>();
        AscendC::LocalTensor<float> buf = calcBuf_.Get<float>();
        AscendC::LocalTensor<float> fp32Bias = buf[sliceSize_];
        DataCopy(fp16X[sliceSize_ * NUM_THREE], gmBias_[offset], numelAlignFp16_);
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        Cast(fp32Bias, fp16X[sliceSize_ * NUM_THREE], AscendC::RoundMode::CAST_NONE, numelAlignFp16_);
    }

    __aicore__ inline float ComputeSquareSum(uint32_t numel, uint32_t sid, uint64_t colOffset)
    {
        AscendC::LocalTensor<T> fp16X = fp16BufX_.Get<T>();
        AscendC::LocalTensor<T> fp16ResIn = fp16X[sliceSize_];
        AscendC::LocalTensor<float> fp32ReduceWorkspace = fp16BufOut_.Get<float>();
        AscendC::LocalTensor<float> fp32Xy = fp32BufXy_.Get<float>();
        AscendC::LocalTensor<float> buf = calcBuf_.Get<float>();
        AscendC::LocalTensor<float> sqx = buf[0];
        AscendC::LocalTensor<float> bias = buf[sliceSize_];

        if constexpr (HAS_BIAS) {
            AscendC::LocalTensor<T> fp16Bias = fp16X[sliceSize_ * NUM_THREE];
            AscendC::LocalTensor<float> bufBias = buf[sliceSize_];
            AddResBiasAndCast<T>(fp16X, fp16ResIn, fp16Bias, fp32Xy, buf, bufBias, numel);
        } else {
            AddResAndCast<T>(fp16X, fp16ResIn, fp32Xy, buf, numel);
        }

        // fp32Xy = x + resIn
        CastFrom32To16(fp16X, fp32Xy, numel);

        AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);

        DataCopyCustom<T>(gmResOut_[colOffset], fp16X, numel);

        AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);

        AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);

        Mul(sqx, fp32Xy, fp32Xy, numelAlignFp32_);

        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);

        ReduceSumCustom(sqx, sqx, fp32ReduceWorkspace, numel);

        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);

        return sqx.GetValue(0);
    }

    __aicore__ void ComputeNorm(float sqs, uint32_t numel)
    {
        AscendC::LocalTensor<T> fp16X = fp16BufX_.Get<T>();
        AscendC::LocalTensor<T> fp16ResIn = fp16X[sliceSize_];
        AscendC::LocalTensor<float> fp32Xy = fp32BufXy_.Get<float>();
        AscendC::LocalTensor<float> buf = calcBuf_.Get<float>();
        AscendC::LocalTensor<T> outBuf = fp16BufOut_.Get<T>();
        AscendC::LocalTensor<float> sqx = buf[0];
        AscendC::LocalTensor<T> fp16Gamma = fp16X[NUM_TWO * sliceSize_];

        if constexpr (HAS_BIAS) {
            AscendC::LocalTensor<T> fp16Bias = fp16X[sliceSize_ * NUM_THREE];
            AscendC::LocalTensor<float> bufBias = buf[sliceSize_];
            AddResBiasAndCast<T>(fp16X, fp16ResIn, fp16Bias, fp32Xy, buf, bufBias, numel);
        } else {
            AddResAndCast<T>(fp16X, fp16ResIn, fp32Xy, buf, numel);
        }

        float factor = 1 / sqs;

        AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);

        Duplicate(sqx, factor, AscendC::DEFAULT_REPEAT_STRIDE);
        AscendC::PipeBarrier<PIPE_V>();

        Sqrt(sqx, sqx, AscendC::DEFAULT_REPEAT_STRIDE);

        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);

        factor = sqx.GetValue(0);
        Muls(fp32Xy, fp32Xy, factor, numelAlignFp32_);
        AscendC::PipeBarrier<PIPE_V>();

        MultiplyGamma(fp16Gamma, sqx, fp32Xy, outBuf, numel, numelAlignFp32_, numelAlignFp16_, precisionMode_);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t numel)
    {
        AscendC::LocalTensor<T> outBuf = fp16BufOut_.Get<T>();
        DataCopyCustom<T>(gmY_[offset], outBuf, numel);
    }

private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16BufX_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32BufXy_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16BufOut_;
    AscendC::GlobalTensor<T> gmX_;
    AscendC::GlobalTensor<T> gmBias_;
    AscendC::GlobalTensor<T> gmGamma_;
    AscendC::GlobalTensor<T> gmResIn_;
    AscendC::GlobalTensor<T> gmY_;
    AscendC::GlobalTensor<T> gmResOut_;
    uint32_t numCore_{0};  // 一共激活多少AICORE
    uint32_t numCol_{0};   // 输入的列数
    uint32_t rowWork_{0};  // 需要计算多少行
    uint64_t gmOffset_{0}; // GM数据起始位置偏移量
    uint32_t sliceSize_{0};  // 每一行切分的大小
    uint32_t sliceSizeTmp_{0};  // 每一行切分的大小
    float epsilon_{1e-12f}; // norm平滑参数
    uint32_t numSlice_{0};
    uint32_t tailSize_{0};
    float avgFactor_{1.0f}; // num_col_的倒数
    float squareSum_{0.0f};
    uint32_t numelAlignFp32_{64};
    uint32_t numelAlignFp16_{32};
    uint32_t precisionMode_{0};
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *pTilingdata, AsdOps::PostRmsNormTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->numCore = (*(const __gm__ uint32_t *)(pTilingdata + 0));
    tilingdata->numCol = (*(const __gm__ uint32_t *)(pTilingdata + 4));
    tilingdata->numRow = (*(const __gm__ uint32_t *)(pTilingdata + 8));
    tilingdata->avgFactor = (*(const __gm__ uint32_t *)(pTilingdata + 12));
    tilingdata->epsilon = (*(const __gm__ uint32_t *)(pTilingdata + 16));
    tilingdata->sliceSize = (*(const __gm__ uint32_t *)(pTilingdata + 20));
    tilingdata->precisionMode = (*(const __gm__ uint32_t *)(pTilingdata + 24));
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdataInUb = nullptr;
    CopyGmTilingToUb(tilingdataInUb, pTilingdata, sizeof(AsdOps::PostRmsNormTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->numCore = (*(__ubuf__ uint32_t *)(tilingdataInUb + 0));
    tilingdata->numCol = (*(__ubuf__ uint32_t *)(tilingdataInUb + 4));
    tilingdata->numRow = (*(__ubuf__ uint32_t *)(tilingdataInUb + 8));
    tilingdata->avgFactor = (*(__ubuf__ uint32_t *)(tilingdataInUb + 12));
    tilingdata->epsilon =  (*(__ubuf__ uint32_t *)(tilingdataInUb + 16));
    tilingdata->sliceSize = (*(__ubuf__ uint32_t *)(tilingdataInUb + 20));
    tilingdata->precisionMode = (*(__ubuf__ uint32_t *)(tilingdataInUb + 24));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

#define GET_TILING_DATA(tilingData, tilingArg)  \
    AsdOps::PostRmsNormTilingData tilingData;    \
    InitTilingData(tilingArg, &(tilingData))

extern "C" __global__ __aicore__ void pre_rms_norm(GM_ADDR x, GM_ADDR bias, GM_ADDR resIn, GM_ADDR g, GM_ADDR y,
    GM_ADDR resOut, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(0)) { // 000
        PreRmsNormShort<half, true> kernel(&pipe, x, bias, resIn, g, y, resOut, tilingData);
        kernel.Launch();
    } else if (TILING_KEY_IS(2)) { // 010
        PreRmsNormShort<half, false> kernel(&pipe, x, bias, resIn, g, y, resOut, tilingData);
        kernel.Launch();
    } else if (TILING_KEY_IS(4)) { // 100
        PreRmsNormLong<half, true> kernel(&pipe, x, bias, resIn, g, y, resOut, tilingData);
        kernel.Launch();
    } else if (TILING_KEY_IS(6)) { // 110
        PreRmsNormLong<half, false> kernel(&pipe, x, bias, resIn, g, y, resOut, tilingData);
        kernel.Launch();
    } else if (TILING_KEY_IS(1000)) { // 000 + double buffer
        PreRmsNormDoubleBuffer<half, true> kernel(&pipe, x, bias, resIn, g, y, resOut, tilingData);
        kernel.Launch();
    } else if (TILING_KEY_IS(1002)) { // 010 + double buffer
        PreRmsNormDoubleBuffer<half, false> kernel(&pipe, x, bias, resIn, g, y, resOut, tilingData);
        kernel.Launch();
    }
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if (TILING_KEY_IS(1)) { // 001
        PreRmsNormShort<bfloat16_t, true> kernel(&pipe, x, bias, resIn, g, y, resOut, tilingData);
        kernel.Launch();
    } else if (TILING_KEY_IS(3)) { // 011
        PreRmsNormShort<bfloat16_t, false> kernel(&pipe, x, bias, resIn, g, y, resOut, tilingData);
        kernel.Launch();
    } else if (TILING_KEY_IS(5)) { // 101
        PreRmsNormLong<bfloat16_t, true> kernel(&pipe, x, bias, resIn, g, y, resOut, tilingData);
        kernel.Launch();
    } else if (TILING_KEY_IS(7)) { // 111
        PreRmsNormLong<bfloat16_t, false> kernel(&pipe, x, bias, resIn, g, y, resOut, tilingData);
        kernel.Launch();
    } else if (TILING_KEY_IS(1001)) { // 001 + double buffer
        PreRmsNormDoubleBuffer<bfloat16_t, true> kernel(&pipe, x, bias, resIn, g, y, resOut, tilingData);
        kernel.Launch();
    } else if (TILING_KEY_IS(1003)) { // 011 + double buffer
        PreRmsNormDoubleBuffer<bfloat16_t, false> kernel(&pipe, x, bias, resIn, g, y, resOut, tilingData);
        kernel.Launch();
    }
#endif
}