/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "post_layer_norm_base.h"
#include "kernels/norm/common/op_kernel/common_quant.h"

static constexpr half QUANT_MIN = -128;

template<bool FastComputeMode = false>
class PostLayerNormQuant : public PostLayerNormBase<half> {
public:
    __aicore__ inline PostLayerNormQuant() {}

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *gamma, __gm__ uint8_t *beta, __gm__ uint8_t *y,
                                __gm__ uint8_t *scale, __gm__ uint8_t *offset, __gm__ uint8_t *z, __gm__ uint8_t *res,
                                AsdOps::PostLayerNormTilingData &tilingData)
    {
        InitBase(x, y, gamma, beta, tilingData);
        zGm.SetGlobalBuffer((__gm__ int8_t *)z + gmOffset_);
        resGm.SetGlobalBuffer((__gm__ half *)res + gmOffset_);
        pipe.InitBuffer(zQue, BUFFER_NUM, rowNumPerStep_ * RoundUp(sliceSize_) * sizeof(int8_t));
        pipe.InitBuffer(resQue, BUFFER_NUM, rowNumPerStep_ * RoundUp(sliceSize_) * sizeof(half));

        GetQuantInfo<float, int32_t>(inputScale_, inputOffset_, scale, offset, xBufFp32);
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
    __aicore__ inline void Compute(int32_t nums)
    {
        AscendC::LocalTensor<half> xLocal = xQue.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = yQue.DeQue<half>();
        AscendC::LocalTensor<half> beta = betaQue.DeQue<half>();
        AscendC::LocalTensor<half> gamma = gammaQue.DeQue<half>();
        AscendC::LocalTensor<int8_t> zLocal = zQue.AllocTensor<int8_t>();
        AscendC::LocalTensor<half> resLocal = resQue.AllocTensor<half>();
        for (int32_t rid = 0; rid < nums; ++rid) {
            const AscendC::LocalTensor<int8_t> &zLocalInner = zLocal[rid * colNum_];
            const AscendC::LocalTensor<half> &resInner = resLocal[rid * colNum_];
            ComputeRes(rid, xLocal, yLocal);
            ComputeRowLayernorm(gamma, beta);
            CastFrom32To16(resInner, yLocalFp32, colNum_);
            ComputeFp16ToI8Quant(zLocalInner, resInner, xLocalFp16,
                (half)inputScale_, (half)inputOffset_, QUANT_MIN, colNum_);
        }
        xQue.FreeTensor(xLocal);
        yQue.FreeTensor(yLocal);
        betaQue.FreeTensor(beta);
        gammaQue.FreeTensor(gamma);
        zQue.EnQue(zLocal);
        resQue.EnQue(resLocal);
    }

    __aicore__ inline void CopyOutMultiRow(uint64_t procId, int32_t size)
    {
        uint64_t offset = procId * rowNumPerStep_ * colNum_;
        CopyOut(zGm, zQue, offset, size);
        CopyOut(resGm, resQue, offset, size);
    }

    __aicore__ inline void FastCompute()
    {
        uint32_t StepNum = CeilDiv(rowNumCurrCore_, rowNumPerStep_);
        for (uint64_t i = 0; i < StepNum; ++i) {
            uint32_t rowNum = (i < StepNum - 1) ? rowNumPerStep_ : rowNumLastStep_;
            CopyInMultiRow(i, rowNum * colNum_);
            Compute(rowNum);
            CopyOutMultiRow(i, rowNum * colNum_);
        }
    }

    __aicore__ inline void SliceCompute()
    {
        for (uint64_t rowIdx = 0; rowIdx < rowNumCurrCore_; rowIdx++) {
            float mean = ComputeRowMean(rowIdx);
            float std = ComputeRowStd(rowIdx, mean);
            for (uint64_t sliceIdx = 0; sliceIdx < sliceNum_; sliceIdx++) {
                uint32_t size = (sliceIdx != sliceNum_ - 1) ? sliceSize_ : tailSliceSize_;
                uint64_t offset = rowIdx * colNum_ + sliceIdx * sliceSize_;
                uint64_t sliceOffset = sliceIdx * sliceSize_;
                ComputeSliceLayernorm(offset, sliceOffset, size, mean, std);
                Cast16AndCopyOut(yLocalFp32, resGm, resQue, offset, size);
                AscendC::LocalTensor<int8_t> zLocal = zQue.AllocTensor<int8_t>();
                ComputeFp32ToI8Quant(zLocal, yLocalFp32, xLocalFp16,
                    (half)inputScale_, (half)inputOffset_, QUANT_MIN, size);
                zQue.EnQue(zLocal);
                CopyOut(zGm, zQue, offset, size);
            }
        }
    }

private:
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> zQue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> resQue;
    AscendC::GlobalTensor<half> resGm;
    AscendC::GlobalTensor<int8_t> zGm;
    float inputScale_{0};
    int32_t inputOffset_{0};
};

extern "C" __global__ __aicore__ void post_layer_norm_quant(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR res_in,
                                                            GM_ADDR scale, GM_ADDR offset, GM_ADDR z, GM_ADDR res,
                                                            GM_ADDR tiling)
{
    if (TILING_KEY_IS(0)) {
        PostLayerNormQuant<true> op;
        GET_TILING_DATA(tiling_data, tiling);
        op.Init(x, gamma, beta, res_in, scale, offset, z, res, tiling_data);
        op.Process();
    }
    if (TILING_KEY_IS(1)) {
        PostLayerNormQuant<false> op;
        GET_TILING_DATA(tiling_data, tiling);
        op.Init(x, gamma, beta, res_in, scale, offset, z, res, tiling_data);
        op.Process();
    }
}