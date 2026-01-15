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

template <bool IS_SYMMETRIC = true>
class LayerNormDynamicQuant : public NormDynamicQuantBase  {
public:
    __aicore__ inline LayerNormDynamicQuant() {}
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
            LNDQCompute(rowIdx, sliceSize_);
            CopyOutAll<IS_SYMMETRIC>(rowIdx, offset, sliceSize_);
        }
    }

private:
    __aicore__ inline void LNDQCompute(uint64_t rowIdx, uint32_t count)
    {
        AscendC::LocalTensor<half> gLocal = gQue_.DeQue<half>();
        AscendC::LocalTensor<half> bLocal = bQue_.DeQue<half>();
        AscendC::LocalTensor<int8_t> yLocal = yQue_.AllocTensor<int8_t>();

        ComputeMean(tmpFp32_, calcFp32_, avgFactor_, count);
        ComputeLayerNorm(tmpFp32_, calcFp32_, tmpFp32_, epsilon_, avgFactor_, gLocal, bLocal, count);
        CastFrom32To16(calcFp16_, tmpFp32_, count);

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

extern "C" __global__ __aicore__ void layer_norm_dynamic_quant(GM_ADDR x, GM_ADDR g, GM_ADDR b,
    GM_ADDR y, GM_ADDR scale, GM_ADDR offset, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(2000000000)) {
        LayerNormDynamicQuant<true> op;
        op.Init(x, g, b, y, scale, offset, tilingData);
        op.Launch();
    }

    if (TILING_KEY_IS(2010000000)) {
        LayerNormDynamicQuant<false> op;
        op.Init(x, g, b, y, scale, offset, tilingData);
        op.Launch();
    }
}
