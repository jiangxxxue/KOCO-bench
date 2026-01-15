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
#include "kernels/norm/coherelayernorm/tiling/tiling_data.h"

static constexpr uint32_t g_singleRowMove = 0;
static constexpr uint32_t g_multipleRowMove = 1;
static constexpr uint32_t g_minBufferSize = 32; // 32 bytes for storing the result of ReduceSum
static constexpr uint32_t g_bufferNum = 1;
static constexpr uint64_t g_alignment = 31;

template <typename T, uint32_t TilingMode = 0>
class CohereLayerNorm {
public:
    __aicore__ inline CohereLayerNorm(__gm__ uint8_t *x, __gm__ uint8_t *gamma, __gm__ uint8_t *y,
                                      const AsdOps::CohereLayerNormTilingData &tilingDataPtr)
        : numCore_(static_cast<uint64_t>(tilingDataPtr.numCore)),
          numCoreRows_(static_cast<uint64_t>(tilingDataPtr.numCoreRows)),
          coreRowStrides_(static_cast<uint64_t>(tilingDataPtr.coreRowStrides)),
          coreRowRepeats_(static_cast<uint64_t>(tilingDataPtr.coreRowRepeats)),
          coreRowTailStrides_(static_cast<uint64_t>(tilingDataPtr.coreRowTailStrides)),
          coreRowTailRepeats_(static_cast<uint64_t>(tilingDataPtr.coreRowTailRepeats)),
          residualCoreRowStrides_(static_cast<uint64_t>(tilingDataPtr.residualCoreRowStrides)),
          residualCoreRowRepeats_(static_cast<uint64_t>(tilingDataPtr.residualCoreRowRepeats)),
          residualCoreRowTailStrides_(static_cast<uint64_t>(tilingDataPtr.residualCoreRowTailStrides)),
          residualCoreRowTailRepeats_(static_cast<uint64_t>(tilingDataPtr.residualCoreRowTailRepeats)),
          numColumns_(static_cast<uint64_t>(tilingDataPtr.numColumns)),
          columnStrides_(static_cast<uint64_t>(tilingDataPtr.columnStrides)),
          columnRepeats_(static_cast<uint64_t>(tilingDataPtr.columnRepeats)),
          residualColumnStrides_(static_cast<uint64_t>(tilingDataPtr.residualColumnStrides)),
          residualColumnRepeats_(static_cast<uint64_t>(tilingDataPtr.residualColumnRepeats)),
          numHeads_(static_cast<uint64_t>(tilingDataPtr.numHeads)),
          epsilon_(tilingDataPtr.epsilon),
          averageFactor_(tilingDataPtr.averageFactor)
    {
        gmOffset_ = static_cast<uint64_t>(AscendC::GetBlockIdx() * numCoreRows_ * numColumns_);
        coreRowStartIndex_ = AscendC::GetBlockIdx() * numCoreRows_;
        xGM_.SetGlobalBuffer((__gm__ T *)x + gmOffset_);
        gammaGM_.SetGlobalBuffer((__gm__ T *)gamma);
        yGM_.SetGlobalBuffer((__gm__ T *)y + gmOffset_);

        bool condition = AscendC::GetBlockIdx() != numCore_ - 1;
        rowTailExist_ = condition ? (coreRowTailRepeats_ > 0) : (residualCoreRowTailRepeats_ > 0);
        colTailExist_ = residualColumnRepeats_ > 0;
        rowStrides_ = condition ? coreRowStrides_ : residualCoreRowStrides_;
        rowRepeats_ = condition ? (coreRowRepeats_ + coreRowTailRepeats_)
                                : (residualCoreRowRepeats_ + residualCoreRowTailRepeats_);
        rowTailStrides_ = condition ? coreRowTailStrides_ : residualCoreRowTailStrides_;
        colRepeats_ = columnRepeats_ + residualColumnRepeats_;

        uint32_t bufferLen = rowStrides_ * columnStrides_;

        pipe.InitBuffer(xQue_, g_bufferNum, AlignedTo32(bufferLen * sizeof(T)));
        pipe.InitBuffer(gammaQue_, g_bufferNum, AlignedTo32(columnStrides_ * sizeof(T)));
        pipe.InitBuffer(yQue_, g_bufferNum, AlignedTo32(bufferLen * sizeof(T)));
        pipe.InitBuffer(sumBufferFP32_, g_minBufferSize);
        pipe.InitBuffer(xBufferFP32_, AlignedTo32(columnStrides_ * sizeof(float)));
        pipe.InitBuffer(gammaBufferFP32_, AlignedTo32(columnStrides_ * sizeof(float)));
        pipe.InitBuffer(yBufferFP32_, AlignedTo32(columnStrides_ * sizeof(float)));
    }

    __aicore__ inline void Launch()
    {
        if constexpr (TilingMode == g_multipleRowMove) {
            ProcessMultipleRows();
        } else {
            ProcessSingleRows();
        }
    }

private:
    __aicore__ inline uint64_t AlignedTo32(uint64_t value)
    {
        return (value + g_alignment) & ~g_alignment;
    }

    __aicore__ inline void ProcessMultipleRows()
    {
        for (uint64_t rowRepeatIndex = 0; rowRepeatIndex < rowRepeats_; rowRepeatIndex++) {
            uint64_t count = rowStrides_ * columnStrides_;
            uint64_t offset = rowRepeatIndex * count;
            if (rowTailExist_ && (rowRepeatIndex == (rowRepeats_ - 1))) {
                count = rowTailStrides_ * columnStrides_;
                CopyInX(offset, count);
                AscendC::PipeBarrier<PIPE_V>();
                Compute(rowRepeatIndex, rowTailStrides_);
                AscendC::PipeBarrier<PIPE_V>();
                CopyOutY(offset, count);
            } else {
                CopyInX(offset, count);
                AscendC::PipeBarrier<PIPE_V>();
                Compute(rowRepeatIndex, rowStrides_);
                AscendC::PipeBarrier<PIPE_V>();
                CopyOutY(offset, count);
            }
        }
    }

    __aicore__ inline void ProcessSingleRows()
    {
        for (uint64_t rowRepeatIndex = 0; rowRepeatIndex < rowRepeats_; rowRepeatIndex++) {
            uint64_t rowOffset = rowRepeatIndex * numColumns_;
            float mean = ComputeMean(rowOffset);
            float variance = ComputeSquareSum(rowOffset, mean);
            Compute(rowOffset, mean, variance, rowRepeatIndex);
        }
    }

    __aicore__ inline void CopyInX(uint64_t offset, uint64_t count)
    {
        AscendC::LocalTensor<T> xLocal = xQue_.AllocTensor<T>();
        DataCopy(xLocal, xGM_[offset], count);
        xQue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInGamma(uint64_t offset, uint64_t count)
    {
        AscendC::LocalTensor<T> gammaLocal = gammaQue_.AllocTensor<T>();
        DataCopy(gammaLocal, gammaGM_[offset], count);
        gammaQue_.EnQue(gammaLocal);
    }

    __aicore__ inline void CopyOutY(uint64_t offset, uint64_t count)
    {
        AscendC::LocalTensor<T> yLocal = yQue_.DeQue<T>();
        DataCopy(yGM_[offset], yLocal, count);
        yQue_.FreeTensor(yLocal);
    }

    __aicore__ inline void Compute(uint64_t rowRepeatIndex, uint64_t rowStrides)
    {
        AscendC::LocalTensor<T> xLocal = xQue_.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = yQue_.AllocTensor<T>();
        AscendC::LocalTensor<float> sumLocal = sumBufferFP32_.Get<float>();
        AscendC::LocalTensor<float> xLocalFP32 = xBufferFP32_.Get<float>();
        AscendC::LocalTensor<float> gammaLocalFP32 = gammaBufferFP32_.Get<float>();
        AscendC::LocalTensor<float> yLocalFP32 = yBufferFP32_.Get<float>();

        for (uint64_t rowStrideIndex = 0; rowStrideIndex < rowStrides; rowStrideIndex++) {
            CastFrom16To32(xLocalFP32, xLocal[rowStrideIndex * columnStrides_], columnStrides_);
            // compute the sum of x, reuse gammaLocalFP32 as workspace for ReduceSum
            ReduceSum(sumLocal, xLocalFP32, gammaLocalFP32, columnStrides_);
            float mean = sumLocal.GetValue(0) * averageFactor_; // compute the E(x)
            Duplicate(gammaLocalFP32, mean, columnStrides_); // reuse gammaLocal to store mean vector
            AscendC::PipeBarrier<PIPE_V>();
            // compute x - E(x), store it in xLocalFP32
            Sub(xLocalFP32, xLocalFP32, gammaLocalFP32, columnStrides_);
            AscendC::PipeBarrier<PIPE_V>();
            // compute (x - E(x))^2, store it in gammaLocalFP32
            Mul(gammaLocalFP32, xLocalFP32, xLocalFP32, columnStrides_);
            AscendC::PipeBarrier<PIPE_V>();
            // compute sum((x - E(x))^2), store it in sumLocal
            ReduceSum(sumLocal, gammaLocalFP32, yLocalFP32, columnStrides_);
            float variance = sumLocal.GetValue(0) * averageFactor_ + epsilon_; // compute the Var(x)
            // reuse yLocalFP32 to store variance vector
            Duplicate(yLocalFP32, variance, columnStrides_);
            AscendC::PipeBarrier<PIPE_V>();
            // compute sqrt(Var(x) + epsilon), store it in yLocalFP32
            Sqrt(yLocalFP32, yLocalFP32, columnStrides_);
            AscendC::PipeBarrier<PIPE_V>();
            // compute (x - E(x)) / sqrt(Var(x) + epsilon), store it in yLocalFP32
            Div(yLocalFP32, xLocalFP32, yLocalFP32, columnStrides_);
            AscendC::PipeBarrier<PIPE_V>();
            // compute the head index corresponding to current x
            uint64_t headIndex = (coreRowStartIndex_ + rowRepeatIndex * rowStrides_ + rowStrideIndex) % numHeads_;
            uint64_t gammaOffset = headIndex * columnStrides_;
            AscendC::PipeBarrier<PIPE_V>();
            CopyInGamma(gammaOffset, columnStrides_);
            AscendC::LocalTensor<T> gammaLocal = gammaQue_.DeQue<T>();
            CastFrom16To32(gammaLocalFP32, gammaLocal, columnStrides_);
            // compute (x - E(x)) / sqrt(Var(x) + epsilon) * gamma, store it in yLocalFP32
            Mul(yLocalFP32, yLocalFP32, gammaLocalFP32, columnStrides_);
            AscendC::PipeBarrier<PIPE_V>();
            CastFrom32To16(yLocal[rowStrideIndex * columnStrides_], yLocalFP32, columnStrides_);
            gammaQue_.FreeTensor(gammaLocal);
        }

        xQue_.FreeTensor(xLocal);
        yQue_.EnQue(yLocal);
    }

    __aicore__ float ComputeMean(uint64_t rowOffset)
    {
        AscendC::LocalTensor<float> sumLocal = sumBufferFP32_.Get<float>();
        AscendC::LocalTensor<float> xLocalFP32 = xBufferFP32_.Get<float>();
        AscendC::LocalTensor<float> gammaLocalFP32 = gammaBufferFP32_.Get<float>();
        uint64_t count = columnStrides_;
        uint64_t columnOffset = 0;
        float accumulatedSum = 0;
        for (uint64_t colRepeatIndex = 0; colRepeatIndex < colRepeats_; colRepeatIndex++) {
            if (colTailExist_ && (colRepeatIndex == colRepeats_ - 1)) {
                count = residualColumnStrides_;
            }
            columnOffset = rowOffset + colRepeatIndex * columnStrides_;
            // copy in x partially
            CopyInX(columnOffset, count);
            AscendC::LocalTensor<T> xLocal = xQue_.DeQue<T>();
            CastFrom16To32(xLocalFP32, xLocal, count);
            // compute the sum of parts of x, reuse gammaLocalFP32 as workspace for ReduceSum
            ReduceSum(sumLocal, xLocalFP32, gammaLocalFP32, count);
            AscendC::PipeBarrier<PIPE_V>();
            accumulatedSum += sumLocal.GetValue(0); // accumulated the sum of x of each part
            xQue_.FreeTensor(xLocal);
        }
        return (accumulatedSum * averageFactor_);
    }

    __aicore__ float ComputeSquareSum(uint64_t rowOffset, float mean)
    {
        AscendC::LocalTensor<float> sumLocal = sumBufferFP32_.Get<float>();
        AscendC::LocalTensor<float> xLocalFP32 = xBufferFP32_.Get<float>();
        AscendC::LocalTensor<float> gammaLocalFP32 = gammaBufferFP32_.Get<float>();
        uint64_t count = columnStrides_;
        uint64_t columnOffset = 0;
        float accumulatedSum = 0;
        for (uint64_t colRepeatIndex = 0; colRepeatIndex < colRepeats_; colRepeatIndex++) {
            if (colTailExist_ && (colRepeatIndex == colRepeats_ - 1)) {
                count = residualColumnStrides_;
            }
            columnOffset = rowOffset + colRepeatIndex * columnStrides_;
            // copy in x partially
            CopyInX(columnOffset, count);
            AscendC::LocalTensor<T> xLocal = xQue_.DeQue<T>();
            CastFrom16To32(xLocalFP32, xLocal, count);
            Duplicate(gammaLocalFP32, mean, count); // reuse gammaLocal to store mean vector
            AscendC::PipeBarrier<PIPE_V>();
            // compute (x - E(x)) partially, store it in xLocalFP32
            Sub(xLocalFP32, xLocalFP32, gammaLocalFP32, count);
            AscendC::PipeBarrier<PIPE_V>();
            // compute (x - E(x))^2 partially, store it in xLocalFP32
            Mul(xLocalFP32, xLocalFP32, xLocalFP32, count);
            AscendC::PipeBarrier<PIPE_V>();
            // compute the sum of (x - E(x))^2 partially, reuse gammaLocalFP32 as workspace for ReduceSum
            ReduceSum(sumLocal, xLocalFP32, gammaLocalFP32, count);
            AscendC::PipeBarrier<PIPE_V>();
            accumulatedSum += sumLocal.GetValue(0); // accumulated the sum of (x - E(x))^2 of each part
            xQue_.FreeTensor(xLocal);
        }
        return (accumulatedSum * averageFactor_ + epsilon_);
    }

    __aicore__ void Compute(uint64_t rowOffset, float mean, float squareSum, uint64_t rowRepeatIndex)
    {
        AscendC::LocalTensor<float> sumLocal = sumBufferFP32_.Get<float>();
        AscendC::LocalTensor<float> xLocalFP32 = xBufferFP32_.Get<float>();
        AscendC::LocalTensor<float> gammaLocalFP32 = gammaBufferFP32_.Get<float>();
        AscendC::LocalTensor<float> yLocalFP32 = yBufferFP32_.Get<float>();
        uint64_t count = columnStrides_;
        uint64_t columnOffset = 0;
        for (uint64_t colRepeatIndex = 0; colRepeatIndex < colRepeats_; colRepeatIndex++) {
            if (colTailExist_ && (colRepeatIndex == colRepeats_ - 1)) {
                count = residualColumnStrides_;
            }
            columnOffset = rowOffset + colRepeatIndex * columnStrides_;
            // copy in x partially
            CopyInX(columnOffset, count);
            AscendC::LocalTensor<T> xLocal = xQue_.DeQue<T>();
            CastFrom16To32(xLocalFP32, xLocal, count);
            // reuse gammaLocal to store mean vector
            Duplicate(gammaLocalFP32, mean, count);
            AscendC::PipeBarrier<PIPE_V>();
            // compute x - E(x) partially, store it in xLocalFP32
            Sub(xLocalFP32, xLocalFP32, gammaLocalFP32, count);
            AscendC::PipeBarrier<PIPE_V>();
            // reuse gammaLocal to store square sum vector
            Duplicate(gammaLocalFP32, squareSum, count);
            AscendC::PipeBarrier<PIPE_V>();
            // compute sqrt(Var(x) + epsilon) partially, store it in gammaLocalFP32
            Sqrt(gammaLocalFP32, gammaLocalFP32, count);
            AscendC::PipeBarrier<PIPE_V>();
            // compute (x - E(x)) / sqrt(Var(x) + epsilon) partially, store it in yLocalFP32
            Div(xLocalFP32, xLocalFP32, gammaLocalFP32, count);
            AscendC::PipeBarrier<PIPE_V>();
            // compute the head index corresponding to current x
            uint64_t headIndex = (coreRowStartIndex_ + rowRepeatIndex) % numHeads_;
            uint64_t gammaOffset = headIndex * numColumns_ + colRepeatIndex * columnStrides_;
            // copy in gamma partially
            CopyInGamma(gammaOffset, count);
            AscendC::LocalTensor<T> gammaLocal = gammaQue_.DeQue<T>();
            CastFrom16To32(gammaLocalFP32, gammaLocal, columnStrides_);
            AscendC::PipeBarrier<PIPE_V>();
            // compute (x - E(x)) / sqrt(Var(x) + epsilon) * gamma, store it in yLocalFP32
            Mul(yLocalFP32, xLocalFP32, gammaLocalFP32, columnStrides_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<T> yLocal = yQue_.AllocTensor<T>();
            CastFrom32To16(yLocal, yLocalFP32, count);
            xQue_.FreeTensor(xLocal);
            gammaQue_.FreeTensor(gammaLocal);
            yQue_.EnQue(yLocal);
            // copy out the result partially
            CopyOutY(columnOffset, count);
        }
    }

private:
    uint64_t numCore_{0};
    uint64_t numCoreRows_{0};
    uint64_t coreRowStrides_{0};
    uint64_t coreRowRepeats_{0};
    uint64_t coreRowTailStrides_{0};
    uint64_t coreRowTailRepeats_{0};
    uint64_t residualCoreRowStrides_{0};
    uint64_t residualCoreRowRepeats_{0};
    uint64_t residualCoreRowTailStrides_{0};
    uint64_t residualCoreRowTailRepeats_{0};
    uint64_t numColumns_{0};
    uint64_t columnStrides_{0};
    uint64_t columnRepeats_{0};
    uint64_t residualColumnStrides_{0};
    uint64_t residualColumnRepeats_{0};
    uint64_t numHeads_{0};
    float epsilon_{0.0};
    float averageFactor_{0.0};
    uint64_t gmOffset_{0};
    uint64_t coreRowStartIndex_{0};
    uint64_t rowStrides_{0};
    uint64_t rowRepeats_{0};
    uint64_t rowTailStrides_{0};
    bool rowTailExist_{false};
    bool colTailExist_{false};
    uint64_t colRepeats_{0};
    uint64_t count_{0};
    uint64_t offset_{0};
    
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, g_bufferNum> xQue_;
    AscendC::TQue<AscendC::TPosition::VECIN, g_bufferNum> gammaQue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, g_bufferNum> yQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBufferFP32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> varianceBufferFP32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xBufferFP32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaBufferFP32_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yBufferFP32_;
    AscendC::GlobalTensor<T> xGM_, gammaGM_;
    AscendC::GlobalTensor<T> yGM_;
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AsdOps::CohereLayerNormTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->numCore = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->numCoreRows = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->coreRowStrides = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
    tilingdata->coreRowRepeats = (*(const __gm__ uint32_t *)(p_tilingdata + 12));
    tilingdata->coreRowTailStrides = (*(const __gm__ uint32_t *)(p_tilingdata + 16));
    tilingdata->coreRowTailRepeats = (*(const __gm__ uint32_t *)(p_tilingdata + 20));
    tilingdata->residualCoreRowStrides = (*(const __gm__ uint32_t *)(p_tilingdata + 24));
    tilingdata->residualCoreRowRepeats = (*(const __gm__ uint32_t *)(p_tilingdata + 28));
    tilingdata->residualCoreRowTailStrides = (*(const __gm__ uint32_t *)(p_tilingdata + 32));
    tilingdata->residualCoreRowTailRepeats = (*(const __gm__ uint32_t *)(p_tilingdata + 36));
    tilingdata->numColumns = (*(const __gm__ uint32_t *)(p_tilingdata + 40));
    tilingdata->columnStrides = (*(const __gm__ uint32_t *)(p_tilingdata + 44));
    tilingdata->columnRepeats = (*(const __gm__ uint32_t *)(p_tilingdata + 48));
    tilingdata->residualColumnStrides = (*(const __gm__ uint32_t *)(p_tilingdata + 52));
    tilingdata->residualColumnRepeats = (*(const __gm__ uint32_t *)(p_tilingdata + 56));
    tilingdata->numHeads = (*(const __gm__ uint32_t *)(p_tilingdata + 60));
    tilingdata->epsilon = (*(const __gm__ float *)(p_tilingdata + 64));
    tilingdata->averageFactor = (*(const __gm__ float *)(p_tilingdata + 68));
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AsdOps::CohereLayerNormTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->numCore = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->numCoreRows = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->coreRowStrides = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
    tilingdata->coreRowRepeats = (*(const __gm__ uint32_t *)(p_tilingdata + 12));
    tilingdata->coreRowTailStrides = (*(const __gm__ uint32_t *)(p_tilingdata + 16));
    tilingdata->coreRowTailRepeats = (*(const __gm__ uint32_t *)(p_tilingdata + 20));
    tilingdata->residualCoreRowStrides = (*(const __gm__ uint32_t *)(p_tilingdata + 24));
    tilingdata->residualCoreRowRepeats = (*(const __gm__ uint32_t *)(p_tilingdata + 28));
    tilingdata->residualCoreRowTailStrides = (*(const __gm__ uint32_t *)(p_tilingdata + 32));
    tilingdata->residualCoreRowTailRepeats = (*(const __gm__ uint32_t *)(p_tilingdata + 36));
    tilingdata->numColumns = (*(const __gm__ uint32_t *)(p_tilingdata + 40));
    tilingdata->columnStrides = (*(const __gm__ uint32_t *)(p_tilingdata + 44));
    tilingdata->columnRepeats = (*(const __gm__ uint32_t *)(p_tilingdata + 48));
    tilingdata->residualColumnStrides = (*(const __gm__ uint32_t *)(p_tilingdata + 52));
    tilingdata->residualColumnRepeats = (*(const __gm__ uint32_t *)(p_tilingdata + 56));
    tilingdata->numHeads = (*(const __gm__ uint32_t *)(p_tilingdata + 60));
    tilingdata->epsilon = (*(const __gm__ float *)(p_tilingdata + 64));
    tilingdata->averageFactor = (*(const __gm__ float *)(p_tilingdata + 68));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

#define GET_TILING_DATA(tilingArg, tilingData)                                                                         \
    AsdOps::CohereLayerNormTilingData tilingData;                                                                      \
    InitTilingData(tilingArg, &(tilingData))
 
extern "C" __global__ __aicore__ void cohere_layer_norm(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling, tilingData);
    if (TILING_KEY_IS(2000000000)) { // half & multiple row move
        CohereLayerNorm<half, g_multipleRowMove> op(x, gamma, y, tilingData);
        op.Launch();
    }
    if (TILING_KEY_IS(2010000000)) { // half & single row mov
        CohereLayerNorm<half, g_singleRowMove> op(x, gamma, y, tilingData);
        op.Launch();
    }
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if (TILING_KEY_IS(2100000000)) { // bf16 & multiple row move
        CohereLayerNorm<bfloat16_t, g_multipleRowMove> op(x, gamma, y, tilingData);
        op.Launch();
    }
    if (TILING_KEY_IS(2110000000)) { // bf16 & single row mov
        CohereLayerNorm<bfloat16_t, g_singleRowMove> op(x, gamma, y, tilingData);
        op.Launch();
    }
#endif
}