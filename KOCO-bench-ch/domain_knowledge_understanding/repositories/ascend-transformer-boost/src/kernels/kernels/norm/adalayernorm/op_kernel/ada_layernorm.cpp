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
#include "kernels/norm/adalayernorm/tiling/tiling_data.h"

constexpr int32_t DOUBLE_BUFFER_NUM = 2;

using AscendC::HardEvent;

template<bool FastComputeMode, typename InDtype, typename InnerDtype>
class AdaLayerNorm {
public:
    __aicore__ inline AdaLayerNorm() {}

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *gamma, __gm__ uint8_t *beta,
                                __gm__ uint8_t *z, AsdOps::AdaLayerNormTilingData &tilingData, AscendC::TPipe* pipe)
    {
        coreIdx_ = AscendC::GetBlockIdx();
        eps_ = tilingData.epsStr;
        aveNum_ = tilingData.aveStr;
        coreNum_ = tilingData.numCore;  // 核数
        coreLoop_ = tilingData.coreLoop;
        numFirstDim_ = tilingData.numFirstDim;  // X第一维的大小
        numRowDim_ = tilingData.numRowDim;  // 中间维的大小
        numLastDim_ = tilingData.numLastDim;  // 最后一维的大小
        gammaFirstDim_ = tilingData.gammaFirstDim;
        gammaRowDim_ = tilingData.gammaRowDim;
        rowLoop_ = tilingData.rowLoop;
        rowSize_ = tilingData.rowSize;
        tailRowSize_ = tilingData.tailRowSize;
        sliceNum_ = tilingData.sliceNum;
        sliceSize_ = tilingData.sliceSize;
        tailSliceSize_ = tilingData.tailSliceSize;

        xGm.SetGlobalBuffer((__gm__ InDtype *)x);
        gammaGm.SetGlobalBuffer((__gm__ InDtype *)gamma);
        betaGm.SetGlobalBuffer((__gm__ InDtype *)beta);
        zGm.SetGlobalBuffer((__gm__ InDtype *)z);
        pipe->InitBuffer(xQue, DOUBLE_BUFFER_NUM, rowSize_ * RoundUp(sliceSize_) * sizeof(InDtype));
        pipe->InitBuffer(betaQue, DOUBLE_BUFFER_NUM, RoundUp(sliceSize_) * sizeof(InDtype));
        pipe->InitBuffer(gammaQue, DOUBLE_BUFFER_NUM, RoundUp(sliceSize_) * sizeof(InDtype));
        pipe->InitBuffer(zQue, DOUBLE_BUFFER_NUM, rowSize_ * RoundUp(sliceSize_) * sizeof(InDtype));
        pipe->InitBuffer(xBufFp32, RoundUp(sliceSize_) * sizeof(InnerDtype));
        pipe->InitBuffer(yBufFp32, RoundUp(sliceSize_) * sizeof(InnerDtype));
        yLocalFp32 = yBufFp32.Get<InnerDtype>();
        xLocalFp32 = xBufFp32.Get<InnerDtype>();
    }

    __aicore__ inline void Process()
    {
        if constexpr (FastComputeMode) {
            FastCompute();
        } else {
            SliceCompute();
        }
    }

private:
    __aicore__ inline void FastCompute()
    {
        for (uint64_t loopIdx = coreIdx_; loopIdx < coreLoop_; loopIdx += coreNum_) {
            // 1.计算偏移
            uint64_t batchIdx = loopIdx / rowLoop_;  // 计算batchIdx
            // rowidx、rowSize
            uint64_t rowIdx = loopIdx % rowLoop_;
            uint64_t rowSize = (rowIdx == rowLoop_-1) ? tailRowSize_ : rowSize_;
            uint64_t sliceIdx = 0;  // 对于fastcompute，sliceIdx=0
            uint64_t sliceSize = sliceSize_;
            // offsetX
            uint64_t offsetX = (uint64_t)batchIdx * numRowDim_ * numLastDim_ + rowIdx * rowSize_ * numLastDim_;
            uint64_t gammaBatchIdx = (batchIdx * gammaFirstDim_ / numFirstDim_);
            if (gammaRowDim_ == 1) {
                // 2. 搬入
                FastCopyIn(offsetX, gammaBatchIdx, rowSize);
                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                // 3. 计算
                FastPrecision2dCompute(rowSize);
                AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                // 4. 搬出
                CopyOut(zGm, zQue, offsetX, rowSize * numLastDim_);
            } else {
                 // 2. 搬入
                CopyIn(xGm, xQue, offsetX, rowSize * numLastDim_);
                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                // 3. 计算
                FastPrecision3dCompute(gammaBatchIdx, rowIdx, rowSize);
                AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                // 4. 搬出
                CopyOut(zGm, zQue, offsetX, rowSize * numLastDim_);
            }
        }
    }

    __aicore__ inline void FastCopyIn(uint64_t offset, uint32_t gammabatchIdx, uint64_t rowSize)
    {
        CopyIn(xGm, xQue, offset, rowSize * numLastDim_);
        CopyIn(gammaGm, gammaQue, gammabatchIdx  * numLastDim_, numLastDim_);
        CopyIn(betaGm, betaQue, gammabatchIdx * numLastDim_, numLastDim_);
    }

    __aicore__ inline void FastPrecision2dCompute(uint64_t rowsize)
    {
        AscendC::LocalTensor<InDtype> xLocal = xQue.DeQue<InDtype>();
        AscendC::LocalTensor<InDtype> zLocal = zQue.AllocTensor<InDtype>();
        AscendC::LocalTensor<InDtype> beta = betaQue.DeQue<InDtype>();
        AscendC::LocalTensor<InDtype> gamma = gammaQue.DeQue<InDtype>();
        for (int64_t rid = 0; rid < rowsize; ++rid) {
            const AscendC::LocalTensor<InDtype> &zLocalInner = zLocal[rid * numLastDim_];
            CastFrom16To32(xLocalFp32, xLocal[rid * numLastDim_], numLastDim_);
            ComputeRowLayernorm(gamma, beta);
            CastFrom32To16(zLocalInner, yLocalFp32, numLastDim_);
            AscendC::PipeBarrier<PIPE_V>();
        }
        xQue.FreeTensor(xLocal);
        betaQue.FreeTensor(beta);
        gammaQue.FreeTensor(gamma);
        zQue.EnQue(zLocal);
    }

    __aicore__ inline void FastPrecision3dCompute(uint32_t gammaBatchIdx, uint32_t rowIdx, uint64_t rowsize)
    {
        AscendC::LocalTensor<InDtype> xLocal = xQue.DeQue<InDtype>();
        AscendC::LocalTensor<InDtype> zLocal = zQue.AllocTensor<InDtype>();
        for (int64_t rid = 0; rid < rowsize; ++rid) {
            uint64_t gammaRowIdx = ((rowIdx * rowSize_ + rid) * gammaRowDim_ / numRowDim_);
            uint64_t offsetGamma = (gammaBatchIdx * gammaRowDim_+ gammaRowIdx) * numLastDim_;
            CopyIn(gammaGm, gammaQue, offsetGamma, numLastDim_);
            CopyIn(betaGm, betaQue, offsetGamma, numLastDim_);
            AscendC::LocalTensor<InDtype> beta = betaQue.DeQue<InDtype>();
            AscendC::LocalTensor<InDtype> gamma = gammaQue.DeQue<InDtype>();
            const AscendC::LocalTensor<InDtype> &zLocalInner = zLocal[rid * numLastDim_];
            CastFrom16To32(xLocalFp32, xLocal[rid * numLastDim_], numLastDim_);
            ComputeRowLayernorm(gamma, beta);
            CastFrom32To16(zLocalInner, yLocalFp32, numLastDim_);
            AscendC::PipeBarrier<PIPE_V>();
            gammaQue.FreeTensor(gamma);
            betaQue.FreeTensor(beta);
        }
        xQue.FreeTensor(xLocal);
        zQue.EnQue(zLocal);
    }

    __aicore__ inline void SliceCompute()
    {
        // 分成一行一行计算，一行数据超过UBSize需要slice
        for (uint64_t loopIdx = coreIdx_; loopIdx < coreLoop_; loopIdx += coreNum_) {
            // 1.计算偏移
            uint64_t batchIdx = loopIdx / rowLoop_;  // 对应的batch
            // rowIdx
            uint64_t rowIdx = loopIdx % rowLoop_;
            // 2. 计算一行的均值和方差
            float mean = ComputeRowMean(batchIdx, rowIdx);
            float std = ComputeRowStd(batchIdx, rowIdx, mean);
            // 3. 计算一行的layernorm
            for (uint64_t sliceIdx = 0; sliceIdx < sliceNum_; sliceIdx++) {
                uint64_t size = 0;
                uint64_t offset = 0;
                uint64_t scaleOffset = 0;
                GetSizeAndOffset(batchIdx, rowIdx, sliceIdx, size, offset);
                GetScaleOffset(batchIdx, rowIdx, sliceIdx, scaleOffset);
                ComputeSliceLayernorm(offset, scaleOffset, size, mean, std);
            }
        }
    }

private:
    __aicore__ inline void ComputeRowLayernorm(const AscendC::LocalTensor<InDtype> gamma,
        const AscendC::LocalTensor<InDtype> beta)
    {
        // 计算xLocalFp32的均值保存到yLocalFp32中;shape为行的大小，值为mean
        ComputeMean(yLocalFp32, xLocalFp32, aveNum_, numLastDim_);
        // 计算layernorm，保存到yLocalFp32中
        ComputeAdaNorm(yLocalFp32, xLocalFp32, yLocalFp32, eps_, aveNum_, gamma, beta, numLastDim_);
    }

    __aicore__ inline void ComputeAdaNorm(const AscendC::LocalTensor<float> &out, const AscendC::LocalTensor<float> &in,
        const AscendC::LocalTensor<float> &mean, float eps, float aveNum, const AscendC::LocalTensor<InDtype> &gamma,
        const AscendC::LocalTensor<InDtype> &beta, uint32_t count)
    {
        Sub(in, in, mean, count);
        AscendC::PipeBarrier<PIPE_V>();
        Mul(out, in, in, count);
        AscendC::PipeBarrier<PIPE_V>();
        Muls(out, out, aveNum, count);
        AscendC::PipeBarrier<PIPE_V>();
        ReduceSum(out, out, out, count);
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float var = out.GetValue(0);
        AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Duplicate(out, var, count);
        AscendC::PipeBarrier<PIPE_V>();
        Adds(out, out, eps, count);
        AscendC::PipeBarrier<PIPE_V>();
        Sqrt(out, out, count);
        AscendC::PipeBarrier<PIPE_V>();
        Div(out, in, out, count);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(in, gamma, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
        Adds(in, in, (float)1.0, count);
        AscendC::PipeBarrier<PIPE_V>();
        Mul(out, out, in, count);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(in, beta, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
        Add(out, out, in, count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeSliceLayernorm(uint64_t offset, uint64_t scaleOffset, uint32_t size,
        InnerDtype mean, InnerDtype std)
    {
        CopyInAndCastF32(xLocalFp32, xGm, xQue, offset, size);
        AscendC::PipeBarrier<PIPE_V>();
        Adds(yLocalFp32, xLocalFp32, -mean, size);  // y = x - E(x)
        AscendC::PipeBarrier<PIPE_V>();

        Muls(yLocalFp32, yLocalFp32, (InnerDtype)1.0 / std, size);  // y = y / sqrt(V(x) + eps)
        AscendC::PipeBarrier<PIPE_V>();
        CopyInAndCastF32(xLocalFp32, gammaGm, gammaQue, scaleOffset, size);
        Adds(xLocalFp32, xLocalFp32, float(1.0), size);  // gamma = gamma + 1
        AscendC::PipeBarrier<PIPE_V>();

        Mul(yLocalFp32, xLocalFp32, yLocalFp32, size); // y = y * gamma
        AscendC::PipeBarrier<PIPE_V>();
        CopyInAndCastF32(xLocalFp32, betaGm, betaQue, scaleOffset, size);

        Add(yLocalFp32, xLocalFp32, yLocalFp32, size); // y = y + beta
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<InDtype> z_local = zQue.AllocTensor<InDtype>();
        CastFrom32To16(z_local, yLocalFp32, size);
        AscendC::PipeBarrier<PIPE_V>();
        zQue.EnQue(z_local);
        AscendC::LocalTensor<InDtype> z = zQue.DeQue<InDtype>();
        DataCopy(zGm[offset], z, size);
        zQue.FreeTensor(z);
    }

    __aicore__ inline void GetSizeAndOffset(uint64_t batchIdx, uint64_t rowIdx, uint64_t sliceIdx,
        uint64_t &size, uint64_t &offset)
    {
        // 获取X数据的offset和size
        size = (sliceIdx != sliceNum_ - 1) ? sliceSize_ : tailSliceSize_;
        offset = (batchIdx * numRowDim_ + rowIdx) * numLastDim_ + sliceIdx * sliceSize_;
    }

    __aicore__ inline void GetScaleOffset(uint64_t batchIdx, uint64_t rowIdx, uint64_t sliceIdx, uint64_t &scaleOffset)
    {
        // 获取scale数据的offset
        uint64_t gammaBatchIdx = (batchIdx * gammaFirstDim_ / numFirstDim_);
        uint64_t gammaRowIdx = (rowIdx * gammaRowDim_ / numRowDim_);
        scaleOffset = (gammaBatchIdx * gammaRowDim_+ gammaRowIdx) * numLastDim_ + sliceIdx * sliceSize_;
    }

    __aicore__ inline InnerDtype ComputeRowMean(uint64_t batchIdx, uint64_t rowIdx)
    {
        // 计算rowIdx行的mean
        InnerDtype sum = 0;
        for (uint64_t sliceIdx = 0; sliceIdx < sliceNum_; sliceIdx++) {
            uint64_t size = 0;
            uint64_t offset = 0;
            GetSizeAndOffset(batchIdx, rowIdx, sliceIdx, size, offset);
            CopyInAndCastF32(xLocalFp32, xGm, xQue, offset, size);  // 从inputX拷贝到xLocalFp32
            AscendC::PipeBarrier<PIPE_V>();
            sum += ComputeSum(xLocalFp32, yLocalFp32, xLocalFp32, size);    // 计算SUM
            AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
        }
        return sum * aveNum_;
    }

        __aicore__ inline InnerDtype ComputeRowStd(uint64_t batchIdx, uint64_t rowIdx, InnerDtype mean)
    {
        // 计算rowIdx行的std
        InnerDtype squareSum = 0;
        for (uint64_t sliceIdx = 0; sliceIdx < sliceNum_; sliceIdx++) {
            uint64_t size = 0;
            uint64_t offset = 0;
            GetSizeAndOffset(batchIdx, rowIdx, sliceIdx, size, offset);
            CopyInAndCastF32(xLocalFp32, xGm, xQue, offset, size);
            AscendC::PipeBarrier<PIPE_V>();
            Adds(yLocalFp32, xLocalFp32, -mean, size);
            AscendC::PipeBarrier<PIPE_V>();
            squareSum += ComputeSliceSquareSum(yLocalFp32, yLocalFp32, yLocalFp32, size);
            AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
        }
        InnerDtype var = squareSum * aveNum_ + eps_;
        return sqrt(var);
    }

private:
    AscendC::TPipe* pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> xQue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> gammaQue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> betaQue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> zQue;

    AscendC::TBuf<AscendC::TPosition::VECCALC> xBufFp32;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yBufFp32;
    AscendC::LocalTensor<InnerDtype> yLocalFp32;  // 相加操作的，临时LocalTensor
    AscendC::LocalTensor<InnerDtype> xLocalFp32;
    AscendC::GlobalTensor<InDtype> xGm, gammaGm, betaGm, zGm;

    uint32_t coreIdx_{0};
    uint32_t coreNum_{0};
    uint32_t coreLoop_{0};
    uint32_t numFirstDim_{0};
    uint32_t numRowDim_{0};
    uint32_t numLastDim_{0};
    uint32_t gammaFirstDim_{0};
    uint32_t gammaRowDim_{0};
    InnerDtype aveNum_{0};
    InnerDtype eps_{0};
    uint32_t rowLoop_{0};
    uint32_t rowSize_{0};
    uint32_t tailRowSize_{0};
    uint32_t sliceNum_{0};
    uint32_t sliceSize_{0};
    uint32_t tailSliceSize_{0};
    uint64_t gmOffset_{0};
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AsdOps::AdaLayerNormTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->numCore = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->epsStr = (*(const __gm__ float *)(p_tilingdata + 4));
    tilingdata->aveStr = (*(const __gm__ float *)(p_tilingdata + 8));
    tilingdata->normMode = (*(const __gm__ uint32_t *)(p_tilingdata + 12));
    tilingdata->numFirstDim = (*(const __gm__ uint32_t *)(p_tilingdata + 16));
    tilingdata->numRowDim = (*(const __gm__ uint32_t *)(p_tilingdata + 20));
    tilingdata->numLastDim = (*(const __gm__ uint32_t *)(p_tilingdata + 24));
    tilingdata->gammaFirstDim = (*(const __gm__ uint32_t *)(p_tilingdata + 28));
    tilingdata->gammaRowDim = (*(const __gm__ uint32_t *)(p_tilingdata + 32));
    tilingdata->coreLoop = (*(const __gm__ uint32_t *)(p_tilingdata + 36));
    tilingdata->rowLoop = (*(const __gm__ uint32_t *)(p_tilingdata + 40));
    tilingdata->rowSize = (*(const __gm__ uint32_t *)(p_tilingdata + 44));
    tilingdata->tailRowSize = (*(const __gm__ uint32_t *)(p_tilingdata + 48));
    tilingdata->sliceNum = (*(const __gm__ uint32_t *)(p_tilingdata + 52));
    tilingdata->sliceSize = (*(const __gm__ uint32_t *)(p_tilingdata + 56));
    tilingdata->tailSliceSize = (*(const __gm__ uint32_t *)(p_tilingdata + 60));
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AsdOps::AdaLayerNormTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->numCore = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->epsStr = (*(__ubuf__ float *)(tilingdata_in_ub + 4));
    tilingdata->aveStr = (*(__ubuf__ float *)(tilingdata_in_ub + 8));
    tilingdata->normMode = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 12));
    tilingdata->numFirstDim = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 16));
    tilingdata->numRowDim = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 20));
    tilingdata->numLastDim = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 24));
    tilingdata->gammaFirstDim = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 28));
    tilingdata->gammaRowDim = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 32));
    tilingdata->coreLoop = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 36));
    tilingdata->rowLoop = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 40));
    tilingdata->rowSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 44));
    tilingdata->tailRowSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 48));
    tilingdata->sliceNum = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 52));
    tilingdata->sliceSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 56));
    tilingdata->tailSliceSize = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 60));
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

#define GET_TILING_DATA(tilingArg, tilingData) \
    AsdOps::AdaLayerNormTilingData tilingData; \
    InitTilingData(tilingArg, &(tilingData))

extern "C" __global__ __aicore__ void ada_layer_norm(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR z, GM_ADDR tiling)
{
    AscendC::TPipe tpipe;
    GET_TILING_DATA(tiling, tilingData);
    if (TILING_KEY_IS(2000000000)) { // half & SliceCompute
        AdaLayerNorm<false, half, float> op;
        op.Init(x, gamma, beta, z, tilingData, &tpipe);
        op.Process();
    }
    if (TILING_KEY_IS(2010000000)) { // half & FastCompute
        AdaLayerNorm<true, half, float> op;
        op.Init(x, gamma, beta, z, tilingData, &tpipe);
        op.Process();
    }
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if (TILING_KEY_IS(2100000000)) { // bf16 & SliceCompute
        AdaLayerNorm<false, bfloat16_t, float> op;
        op.Init(x, gamma, beta, z, tilingData, &tpipe);
        op.Process();
    }
    if (TILING_KEY_IS(2110000000)) { // bf16 & FastCompute
        AdaLayerNorm<true, bfloat16_t, float> op;
        op.Init(x, gamma, beta, z, tilingData, &tpipe);
        op.Process();
    }
#endif
}