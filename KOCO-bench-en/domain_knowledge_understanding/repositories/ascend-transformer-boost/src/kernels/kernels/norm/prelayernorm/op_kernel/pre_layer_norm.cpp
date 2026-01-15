/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernels/norm/postlayernorm/op_kernel/post_layer_norm_base.h"

template<typename T, bool FastComputeMode = true>
class PreLayerNorm : public PostLayerNormBase<T> {
public:
    __aicore__ inline PreLayerNorm() {}

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *gamma, __gm__ uint8_t *beta,
                                __gm__ uint8_t *z, __gm__ uint8_t *res_out, AsdOps::PostLayerNormTilingData &tilingData)
    {
        this->InitBase(x, y, gamma, beta, tilingData);
        zGm.SetGlobalBuffer((__gm__ T *)z + this->gmOffset_);
        resOutGm.SetGlobalBuffer((__gm__ T *)res_out + this->gmOffset_);
        this->pipe.InitBuffer(zQue, PostLayerNormBase<T>::BUFFER_NUM,
                      this->rowNumPerStep_* RoundUp(this->sliceSize_) * sizeof(T));
        this->pipe.InitBuffer(resOutQue, PostLayerNormBase<T>::BUFFER_NUM,
                      this->rowNumPerStep_* RoundUp(this->sliceSize_) * sizeof(T));
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
    __aicore__ inline void SliceCompute()
    {
        for (uint32_t rowIdx = 0; rowIdx < this->rowNumCurrCore_; rowIdx++) {
            float mean = this->ComputeRowMean(rowIdx);
            float std = this->ComputeRowStd(rowIdx, mean);
            for (uint64_t sliceIdx = 0; sliceIdx < this->sliceNum_; sliceIdx++) {
                uint32_t size = 0;
                uint64_t offset = 0;
                uint64_t sliceOffset = 0;
                this->GetSizeAndOffset(rowIdx, sliceIdx, size, offset, sliceOffset);
                this->ComputeResidual(offset, size);
                Cast16AndCopyOut(this->xLocalFp32, resOutGm, resOutQue, offset, size);

                this->ComputeSliceLayernorm(offset, sliceOffset, size, mean, std);
                Cast16AndCopyOut(this->yLocalFp32, zGm, zQue, offset, size);
            }
        }
    }

    __aicore__ inline void FastCompute()
    {
        uint32_t StepNum = CeilDiv(this->rowNumCurrCore_, this->rowNumPerStep_);
        for (uint64_t i = 0; i < StepNum; ++i) {
            uint32_t rowNum = (i < StepNum - 1) ? this->rowNumPerStep_ : this->rowNumLastStep_;
            this->CopyInMultiRow(i, rowNum * this->colNum_);
            Compute(rowNum);
            this->CopyOutMultiRow(i, rowNum * this->colNum_);
        }
    }

private:
    __aicore__ inline void Compute(int32_t rowNum)
    {
        AscendC::LocalTensor<T> xLocal = this->xQue.template DeQue<T>();
        AscendC::LocalTensor<T> yLocal = this->yQue.template DeQue<T>();
        AscendC::LocalTensor<T> beta = this->betaQue.template DeQue<T>();
        AscendC::LocalTensor<T> gamma = this->gammaQue.template DeQue<T>();
        AscendC::LocalTensor<T> zLocal = zQue.template AllocTensor<T>();
        AscendC::LocalTensor<T> resOutLocal = resOutQue.template AllocTensor<T>();

        for (int32_t rid = 0; rid < rowNum; ++rid) {
            const AscendC::LocalTensor<T> &zLocalInner = zLocal[rid * this->colNum_];
            const AscendC::LocalTensor<T> &resLocalInner = resOutLocal[rid * this->colNum_];

            this->ComputeRes(rid, xLocal, yLocal);
            CastFrom32To16(resLocalInner, this->xLocalFp32, this->colNum_);
            this->ComputeRowLayernorm(gamma, beta);

            CastFrom32To16(zLocalInner, this->yLocalFp32, this->colNum_);
        }
        this->xQue.FreeTensor(xLocal);
        this->yQue.FreeTensor(yLocal);
        this->betaQue.FreeTensor(beta);
        this->gammaQue.FreeTensor(gamma);
        zQue.EnQue(zLocal);
        resOutQue.EnQue(resOutLocal);
    }

    __aicore__ inline void CopyOutMultiRow(int64_t procId, int32_t size)
    {
        uint64_t offset = procId * this->rowNumPerStep_ * this->colNum_;
        CopyOut(zGm, zQue, offset, size);
        CopyOut(resOutGm, resOutQue, offset, size);
    }

private:
    AscendC::GlobalTensor<T> zGm;
    AscendC::GlobalTensor<T> resOutGm;
    AscendC::TQue<AscendC::QuePosition::VECOUT, PostLayerNormBase<T>::BUFFER_NUM> zQue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, PostLayerNormBase<T>::BUFFER_NUM> resOutQue;
};

extern "C" __global__ __aicore__ void pre_layer_norm(GM_ADDR x, GM_ADDR res_in, GM_ADDR gamma, GM_ADDR beta,
                                                      GM_ADDR z, GM_ADDR res_out, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(0)) { // 00
        PreLayerNorm<half, true> op;
        op.Init(x, res_in, gamma, beta, z, res_out, tiling_data);
        op.Process();
    }
    if (TILING_KEY_IS(1)) { // 01
        PreLayerNorm<half, false> op;
        op.Init(x, res_in, gamma, beta, z, res_out, tiling_data);
        op.Process();
    }
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if (TILING_KEY_IS(2)) { // 10
        PreLayerNorm<bfloat16_t, true> op;
        op.Init(x, res_in, gamma, beta, z, res_out, tiling_data);
        op.Process();
    }
    if (TILING_KEY_IS(3)) { // 11
        PreLayerNorm<bfloat16_t, false> op;
        op.Init(x, res_in, gamma, beta, z, res_out, tiling_data);
        op.Process();
    }
#endif
}