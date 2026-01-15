/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef DYNAMIC_QUANT_UNALIGN_910B_H
#define DYNAMIC_QUANT_UNALIGN_910B_H

#include "kernel_operator.h"
#include "dynamic_quant_align.h"
#include "kernels/elewise/dynamic_quant/dynamic_quant_tiling/dynamic_quant_util.h"


template <typename T>
class DynamicQuantUnalign910B : public DynamicQuantAlign<T> {
public:
    __aicore__ inline DynamicQuantUnalign910B() {};
    __aicore__ inline void InitParams(AsdOps::DynamicQuantTilingData *tilingdata)
    {
        DynamicQuantAlign<T>::InitParams(tilingdata);

        sizeZOut_ = tilingdata->sizeZOut;
        sizeCopyRow_ = tilingdata->sizeCopyRow;

        lenXSizeH_ = this->sizeH_ * AsdOps::DYNAMIC_QUANT_SIZE_OF_HALF;
        lenZSizeH_ = this->sizeH_ * AsdOps::DYNAMIC_QUANT_SIZE_OF_INT8;
        numRightPad_ = this->sizeX_ - this->sizeH_;

        isSizeXPad_ = this->sizeH_ % AsdOps::DYNAMIC_QUANT_ALIGN_NUM_X != 0;
        isSizeZPad_ = this->sizeH_ % AsdOps::DYNAMIC_QUANT_ALIGN_SIZE != 0;
    }

    __aicore__ inline void InitBuffer(__gm__ uint8_t *x, __gm__ uint8_t *z, __gm__ uint8_t *scale,
        __gm__ uint8_t *offset)
    {
        if (this->isTailLoop_) {
            this->inGm_.SetGlobalBuffer((__gm__ T*)x + this->numHeadCore_ * this->lenHead_ + \
                (this->blockIdx_ - this->numHeadCore_) * this->lenTail_, this->lenTail_ + this->lenLastTail_);
            this->outGm_.SetGlobalBuffer((__gm__ int8_t*)z + this->numHeadCore_ * this->lenHead_ + \
                (this->blockIdx_ - this->numHeadCore_) * this->lenTail_, this->lenTail_ + this->lenLastTail_);
            this->scaleGm_.SetGlobalBuffer((__gm__ float*)scale + this->numHeadCore_ * this->numHeadRow_ + \
                (this->blockIdx_ - this->numHeadCore_) * this->numTailRow_, this->numTailRow_ + this->numLastTailRow_);
            if (this->asymmetric_) {
                this->offsetGm_.SetGlobalBuffer((__gm__ float*)offset + this->numHeadCore_ * this->numHeadRow_ + \
                    (this->blockIdx_ - this->numHeadCore_) * this->numTailRow_, this->numTailRow_ +
                    this->numLastTailRow_);
            }
        } else {
            DynamicQuantAlign<T>::InitBuffer(x, z, scale, offset);
        }
    }

    __aicore__ inline void InitQueue()
    {
        this->pipe_.InitBuffer(this->inQueue_, AsdOps::DYNAMIC_QUANT_BUFFER_NUM_TWO,
            this->sizeX_ * this->numCopyRow_ * AsdOps::DYNAMIC_QUANT_SIZE_OF_HALF);
        this->pipe_.InitBuffer(this->outQueue_, AsdOps::DYNAMIC_QUANT_BUFFER_NUM_ONE,
            sizeZOut_ * this->numCopyRow_ * AsdOps::DYNAMIC_QUANT_SIZE_OF_INT8);
        this->pipe_.InitBuffer(this->scaleQueue_, AsdOps::DYNAMIC_QUANT_BUFFER_NUM_ONE,
            sizeCopyRow_ * AsdOps::DYNAMIC_QUANT_SIZE_OF_FLOAT);
        this->pipe_.InitBuffer(this->fp16Buf_, this->sizeX_ * AsdOps::DYNAMIC_QUANT_SIZE_OF_HALF);
        this->pipe_.InitBuffer(this->fp32Buf_, this->sizeX_ * AsdOps::DYNAMIC_QUANT_SIZE_OF_FLOAT);
        if (this->asymmetric_) {
            this->pipe_.InitBuffer(this->offsetQueue_, AsdOps::DYNAMIC_QUANT_BUFFER_NUM_ONE,
                sizeCopyRow_ * AsdOps::DYNAMIC_QUANT_SIZE_OF_FLOAT);
        }
    }

    __aicore__ inline void Process()
    {
        // tiling strategy, pipeline parallel
        bool isCopyRowPad = this->numCopyRow_ % AsdOps::DYNAMIC_QUANT_ALIGN_NUM_X != 0;
        ProcessWithHeadDoublePipLineForAlign(isCopyRowPad);
        ProcessWithTailCopyInForUnalign();
        ProcessWithHeadLastComputeAndCopyOutForAlign(isCopyRowPad);
        ProcessWithTailComputeAndCopyOutForUnalign();
    }

private:
    __aicore__ inline void ProcessWithHeadDoublePipLineForAlign(bool isCopyRowPad)
    {
        if (this->loopCount_ > 0) {
            CopyIn(0, this->numCopyRow_, this->lenCopyRow_);
            for (uint64_t i = 1; i < this->loopCount_; i++) {
                CopyIn(i, this->numCopyRow_, this->lenCopyRow_);
                ExcuteCompute(this->numCopyRow_);
                CopyOut(i - 1, this->numCopyRow_, this->lenCopyRow_, isCopyRowPad);
            }
        }
    }
    __aicore__ inline void ProcessWithTailCopyInForUnalign()
    {
        if (this->isTailLoop_) {
            CopyIn(this->loopCount_, this->numLastTailRow_, this->lenLastTail_);
        }
    }

    __aicore__ inline void ProcessWithHeadLastComputeAndCopyOutForAlign(bool isCopyRowPad)
    {
        if (this->loopCount_ > 0) {
            ExcuteCompute(this->numCopyRow_);
            CopyOut(this->loopCount_ - 1, this->numCopyRow_, this->lenCopyRow_, isCopyRowPad);
        }
    }

    __aicore__ inline void ProcessWithTailComputeAndCopyOutForUnalign()
    {
        if (this->isTailLoop_) {
            bool isLastRowPad = this->numLastTailRow_ % AsdOps::DYNAMIC_QUANT_ALIGN_NUM_X != 0;
            ExcuteCompute(this->numLastTailRow_);
            CopyOut(this->loopCount_, this->numLastTailRow_, this->lenLastTail_, isLastRowPad);
        }
    }

    __aicore__ inline void ExcuteCompute(uint32_t copyRow)
    {
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            DynamicQuantAlign<T>::ComputeBF16(copyRow, sizeZOut_);
            return;
        }
#endif
        if (this->asymmetric_) {
            DynamicQuantAlign<T>::ComputeAsymmetric(copyRow, sizeZOut_);
        } else {
            DynamicQuantAlign<T>::Compute(copyRow, sizeZOut_);
        }
    }

    __aicore__ inline void CopyIn(uint64_t progress, uint32_t calRow, uint32_t calCount)
    {
        // alloc tensor from queue memory
        AscendC::LocalTensor<T> inLocal = this->inQueue_.template AllocTensor<T>();
        if (isSizeXPad_) {
            uint16_t blockCount = calRow;
            uint16_t blockLen = lenXSizeH_;
            uint16_t srcStride = 0;
            uint16_t dstStride = 0;
            AscendC::DataCopyParams dataCopyParams {blockCount, blockLen, srcStride, dstStride};

            uint8_t leftPadding = 0;
            uint8_t rightPadding = 0;
            uint64_t paddingValue = 0;
            AscendC::DataCopyPadParams padParams {true, leftPadding, rightPadding, paddingValue};

#if __CCE_AICORE__ == 220
            DataCopyPad(inLocal, this->inGm_[progress * this->lenCopyRow_], dataCopyParams, padParams);
#endif
        } else {
            DataCopy(inLocal, this->inGm_[progress * this->lenCopyRow_], calCount);
        }
        this->inQueue_.EnQue(inLocal);
    }

    __aicore__ inline void CopyOut(uint64_t progress, uint32_t calRow, uint32_t calCount,
        bool isScalePad)
    {
        // deque output tensor from VECOUT queue
        AscendC::LocalTensor<int8_t> outLocal = this->outQueue_.template DeQue<int8_t>();
        AscendC::LocalTensor<float> scaleLocal = this->scaleQueue_.template DeQue<float>();
        AscendC::LocalTensor<float> offsetLocal;
        if (this->asymmetric_) {
            offsetLocal = this->offsetQueue_.template DeQue<float>();
        }
        // copy progress_th tile from local tensor to global tensor
        if (isSizeZPad_) {
            uint16_t blockCountX = calRow;
            uint16_t blockLenX = lenZSizeH_;
            AscendC::DataCopyParams dataCopyParamsX {blockCountX, blockLenX, 0, 0};
#if __CCE_AICORE__ == 220
            DataCopyPad(this->outGm_[progress * this->lenCopyRow_], outLocal, dataCopyParamsX);
#endif
        } else {
            DataCopy(this->outGm_[progress * this->lenCopyRow_], outLocal, calCount);
        }

        uint16_t countScale = 1;
        uint16_t lenScale = calRow * AsdOps::DYNAMIC_QUANT_SIZE_OF_FLOAT;
        if (isScalePad) {
            AscendC::DataCopyParams paramsScale {countScale, lenScale, 0, 0};
#if __CCE_AICORE__ == 220
            DataCopyPad(this->scaleGm_[progress * this->numCopyRow_], scaleLocal, paramsScale);
#endif
        } else {
            DataCopy(this->scaleGm_[progress * this->numCopyRow_], scaleLocal, calRow);
        }
        if (this->asymmetric_) {
            if (isScalePad) {
                AscendC::DataCopyParams paramsScale {countScale, lenScale, 0, 0};
#if __CCE_AICORE__ == 220
                DataCopyPad(this->offsetGm_[progress * this->numCopyRow_], offsetLocal, paramsScale);
#endif
            } else {
                DataCopy(this->offsetGm_[progress * this->numCopyRow_], offsetLocal, calRow);
            }
            this->offsetQueue_.FreeTensor(offsetLocal);
        }
        // free output tensor for reuse
        this->outQueue_.FreeTensor(outLocal);
        this->scaleQueue_.FreeTensor(scaleLocal);
    }

private:
    uint32_t sizeZOut_;
    uint32_t sizeCopyRow_;

    uint32_t lenXSizeH_;
    uint32_t lenZSizeH_;
    uint32_t numRightPad_;
    bool isSizeXPad_;
    bool isSizeZPad_;
};

#endif // DYNAMIC_QUANT_UNALIGN_910B_H