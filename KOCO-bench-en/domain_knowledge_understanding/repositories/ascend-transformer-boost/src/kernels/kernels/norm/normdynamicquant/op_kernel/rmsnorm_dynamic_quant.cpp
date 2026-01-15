/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "norm_dynamic_quant_base.h"

template <bool WITH_BETA = true, bool IS_SYMMETRIC = true>
class RmsNormDynamicQuant : public NormDynamicQuantBase {
public:
    __aicore__ inline RmsNormDynamicQuant() {}
    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *g, __gm__ uint8_t *b,
                                __gm__ uint8_t *y, __gm__ uint8_t *scale, __gm__ uint8_t *offset,
                                AsdOps::NormDynamicQuantTilingData &tilingData)
    {
        InitBase(x, g, b, y, scale, offset, tilingData);
    }

    __aicore__ inline void Launch()
    {
        for (uint64_t rowIdx = 0; rowIdx < rowNum_; rowIdx++) {
            uint64_t offset = rowIdx * colNum_;
            CopyInAll(offset, sliceSize_);
            RMDQCompute(rowIdx, sliceSize_);
            CopyOutAll<IS_SYMMETRIC>(rowIdx, offset, sliceSize_);
        }
    }

private:
    __aicore__ inline void RMDQCompute(uint64_t rowIdx, uint32_t count)
    {
        AscendC::LocalTensor<half> gLocal = gQue_.DeQue<half>();
        AscendC::LocalTensor<half> bLocal = bQue_.DeQue<half>();
        AscendC::LocalTensor<int8_t> yLocal = yQue_.AllocTensor<int8_t>();

        float squareSum = ComputeSliceSquareSum(calcFp32_, tmpFp32_, tmpFp32_, count);
        float rms = sqrt(squareSum * avgFactor_ + epsilon_);

        ComputeRmsNorm<WITH_BETA>(calcFp32_, calcFp32_, rms, gLocal, bLocal, tmpFp32_, count);
        CastFrom32To16(calcFp16_, calcFp32_, count);

        if constexpr (IS_SYMMETRIC) {
            SymmetricQuant(rowIdx, yLocal, calcFp16_, tmpFp16_, count);
        } else {
            AsymmetricQuant(rowIdx, yLocal, calcFp16_, tmpFp16_, count);
        }

        yQue_.EnQue(yLocal);
        gQue_.FreeTensor(gLocal);
        bQue_.FreeTensor(bLocal);
    }
};

extern "C" __global__ __aicore__ void rms_norm_dynamic_quant(GM_ADDR x, GM_ADDR g, GM_ADDR b,
    GM_ADDR y, GM_ADDR scale, GM_ADDR offset, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(2000000000)) { // fp16, beta, symmetric
        RmsNormDynamicQuant<true, true> kernel;
        kernel.Init(x, g, b, y, scale, offset, tilingData);
        kernel.Launch();
    }
    if (TILING_KEY_IS(2001000000)) { // fp16, empty beta, symmetric
        RmsNormDynamicQuant<false, true> kernel;
        kernel.Init(x, g, b, y, scale, offset, tilingData);
        kernel.Launch();
    }
    if (TILING_KEY_IS(2010000000)) { // fp16, beta, asymmetric
        RmsNormDynamicQuant<true, false> kernel;
        kernel.Init(x, g, b, y, scale, offset, tilingData);
        kernel.Launch();
    }
    if (TILING_KEY_IS(2011000000)) { // fp16, empty beta, asymmetric
        RmsNormDynamicQuant<false, false> kernel;
        kernel.Init(x, g, b, y, scale, offset, tilingData);
        kernel.Launch();
    }
}