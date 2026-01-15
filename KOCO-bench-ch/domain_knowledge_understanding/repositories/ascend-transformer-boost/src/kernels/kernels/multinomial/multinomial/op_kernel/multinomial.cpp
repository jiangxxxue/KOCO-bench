
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
#include "kernels/multinomial/multinomial/tiling/tiling_data.h"
#include "kernels/utils/kernel/kernel_utils.h"

namespace {
static constexpr int32_t BUFFER_NUM = 1;
static constexpr uint32_t BLOCK_NUMEL = 16;
static constexpr uint32_t DATA_BYTE = 2;
static constexpr uint32_t MAX_CORE_NUM = 50;

using AscendC::HardEvent;

class KernelMultinomial {
public:
    __aicore__ inline KernelMultinomial() {}
    __aicore__ inline void Init(__gm__ uint8_t *x,
                                __gm__ uint8_t *z,
                                uint32_t realLastDim,
                                uint32_t expandLastDim,
                                uint64_t firstDim,
                                float *randValList,
                                uint32_t numSamples,
                                uint32_t numSamplesMax,
                                uint32_t perCoreRunNum,
                                uint32_t nlElePerCorePerRun,
                                uint32_t lElePerCoreLastRun,
                                uint32_t tempUbEleAligened) {
        realLastDim_ = realLastDim;
        firstDim_ = firstDim;
        expandLastDim_ = expandLastDim;
        tilingUb_ = randValList;
        numSamples_ = numSamples;
        numSamplesMax_ = numSamplesMax;
        perCoreRunNum_ = perCoreRunNum;
        nlElePerCorePerRun_ = nlElePerCorePerRun;
        lElePerCoreLastRun_ = lElePerCoreLastRun;
        tempUbEleAligened_ = tempUbEleAligened;
        xGm_.SetGlobalBuffer((__gm__ half *)x + AscendC::GetBlockIdx() * realLastDim_);
        zGm_.SetGlobalBuffer((__gm__ int32_t *)z + AscendC::GetBlockIdx() * numSamples_);
        pipe_.InitBuffer(xQue_, BUFFER_NUM, nlElePerCorePerRun_ * DATA_BYTE);
        pipe_.InitBuffer(outputBuf_, 64 * DATA_BYTE * 2);
        pipe_.InitBuffer(vecIn_, 1, MAX_CORE_NUM * sizeof(int32_t));
    }
    __aicore__ inline void Process(__gm__ uint8_t *sync)
    {
        for (uint64_t j = 0; j < perCoreRunNum_; j++) {
            if (j == perCoreRunNum_ - 1) {
                CopyIn(j, lElePerCoreLastRun_);
                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Compute(j, lElePerCoreLastRun_);
            } else {
                CopyIn(j, nlElePerCorePerRun_);
                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Compute(j, nlElePerCorePerRun_);
            }
            AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
        // 32 blk size
        syncGm_.SetGlobalBuffer((__gm__ int32_t *)(sync), 32 * MAX_CORE_NUM * firstDim_);
        uint64_t blockIdx = AscendC::GetBlockIdx();
        if (firstDim_ != 1) {
            if (blockIdx == firstDim_ - 1) {
                auto syncBuf = vecIn_.AllocTensor<int32_t>();
                AscendC::IBWait(syncGm_, syncBuf, firstDim_ - 2, 0);
                // 此处最后一个核需前一个等待前一个核的event_id，因此 offset == 2
                CopyOut();
                vecIn_.FreeTensor(syncBuf);
            } else if (blockIdx == 0) {
                auto syncBuf = vecIn_.AllocTensor<int32_t>();
                CopyOut();
                AscendC::IBSet(syncGm_, syncBuf, 0, 0);
                vecIn_.FreeTensor(syncBuf);
            } else {
                auto syncBuf = vecIn_.AllocTensor<int32_t>();
                AscendC::IBWait(syncGm_, syncBuf, blockIdx - 1, 0);
                CopyOut();
                AscendC::IBSet(syncGm_, syncBuf, blockIdx, 0);
                vecIn_.FreeTensor(syncBuf);
            }
        } else {
            CopyOut();
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t loopIdx, uint32_t copyEleNum)
    {
        AscendC::LocalTensor<half> xLocal = xQue_.AllocTensor<half>();
        DataCopy(xLocal, xGm_[loopIdx * tempUbEleAligened_], copyEleNum);
        xQue_.EnQue(xLocal);
    }
    __aicore__ inline void Compute(uint64_t loopIdx, uint32_t copyEleNum)
    {
        AscendC::LocalTensor<half> xLocal = xQue_.DeQue<half>();
        AscendC::LocalTensor<int32_t> output = outputBuf_.Get<int32_t>();
        if (loopIdx == 0) {
            tempValue_ = (float)xLocal.GetValue(0);
        } else {
            tempValue_ = (float)xLocal.GetValue(0) + maxNum_;
            xLocal.SetValue(0, (half)tempValue_);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        if (nlElePerCorePerRun_ >= tempUbEleAligened_) {
            realLastDimTemp_ = copyEleNum;
        } else {
            realLastDimTemp_ = realLastDim_;
        }
        for (size_t i = 1; i < realLastDimTemp_; i++) {
            half tempValue = xLocal.GetValue(i - 1);
            half tempValueOne = xLocal.GetValue(i);
            AscendC::PipeBarrier<PIPE_ALL>();
            float tempSum = (float)tempValue + (float)tempValueOne;
            AscendC::PipeBarrier<PIPE_ALL>();
            xLocal.SetValue(i, (half)tempSum);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        maxNum_ = xLocal.GetValue(realLastDimTemp_ - 1);
        AscendC::PipeBarrier<PIPE_ALL>();

        for (uint32_t i = 0; i < numSamples_; i++) {
            if (flagCount_[i] != 1) {
                float randTemp = *(tilingUb_ + (i % numSamplesMax_));
                AscendC::PipeBarrier<PIPE_ALL>();
                float estimateVal = randTemp;
                AscendC::PipeBarrier<PIPE_ALL>();
                for (uint32_t zz = 0; zz < realLastDimTemp_; zz++) {
                    AscendC::PipeBarrier<PIPE_ALL>();
                    if (float(xLocal.GetValue(zz)) >= estimateVal) {
                        output.SetValue(i, (int32_t)(zz + loopIdx * tempUbEleAligened_));
                        flagCount_[i] = 1;
                        break;
                    }
                }
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        for (uint32_t i = 0; i < numSamples_; i++) {
            if (flagCount_[i] != 1) {
                output.SetValue(i, (int32_t)0);
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        xQue_.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<int32_t> output = outputBuf_.Get<int32_t>();
        DataCopy(zGm_[0], output, 8); // 8 for 32 bytes / sizeof(int32_t)
    }
private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> xQue_, vecIn_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuf_;
    AscendC::GlobalTensor<half> xGm_;
    AscendC::GlobalTensor<int32_t> syncGm_;
    AscendC::GlobalTensor<int32_t> zGm_;
    uint32_t realLastDim_{0};
    uint32_t expandLastDim_{0};
    uint32_t numSamples_{0};
    uint32_t numSamplesMax_{0};
    uint64_t firstDim_{0};
    float maxNum_ = 0;
    float tempValue_ = 0;
    uint32_t perCoreRunNum_;
    uint32_t nlElePerCorePerRun_;
    uint32_t lElePerCoreLastRun_;
    uint32_t tempUbEleAligened_;
    uint32_t flagCount_[64] = {0};
    uint32_t realLastDimTemp_ = 0;
    float* tilingUb_{nullptr};
};
}

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AsdOps::MultinomialTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->realLastDim = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->expandLastDim = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->firstDim = (*(const __gm__ uint64_t *)(p_tilingdata + 8));
    tilingdata->numSamples = (*(const __gm__ uint32_t *)(p_tilingdata + 16));
    tilingdata->numSamplesMax = (*(const __gm__ uint32_t *)(p_tilingdata + 20));
    tilingdata->perCoreRunNum = (*(const __gm__ int32_t *)(p_tilingdata + 24));
    tilingdata->nlElePerCorePerRun = (*(const __gm__ uint32_t *)(p_tilingdata + 28));
    tilingdata->lElePerCoreLastRun = (*(const __gm__ uint32_t *)(p_tilingdata + 32));
    tilingdata->tempUbEleAligened = (*(const __gm__ uint32_t *)(p_tilingdata + 36));
    for (uint32_t i = 0; i < tilingdata->numSamplesMax; i++) {
        tilingdata->randValList[i] = (*(const __gm__ float *)(p_tilingdata + 40 + i *sizeof(float)));
    }
#else
    AscendC::TPipe pipe;
    __ubuf__ uint8_t *tilingdata_in_ub = nullptr;
    CopyGmTilingToUb(tilingdata_in_ub, p_tilingdata, sizeof(AsdOps::MultinomialTilingData), &pipe);
    AscendC::PipeBarrier<PIPE_ALL>();
    tilingdata->realLastDim = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 0));
    tilingdata->expandLastDim = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 4));
    tilingdata->firstDim = (*(__ubuf__ uint64_t *)(tilingdata_in_ub + 8));
    tilingdata->numSamples = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 16));
    tilingdata->numSamplesMax = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 20));
    tilingdata->perCoreRunNum = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 24));
    tilingdata->nlElePerCorePerRun = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 28));
    tilingdata->lElePerCoreLastRun = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 32));
    tilingdata->tempUbEleAligened = (*(__ubuf__ uint32_t *)(tilingdata_in_ub + 36));
    for (uint32_t i = 0; i < tilingdata->numSamplesMax; i++) {
        tilingdata->randValList[i] = (*(__ubuf__ float *)(tilingdata_in_ub + 40 + i * sizeof(float)));
    }
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}


#define GET_TILING_DATA(tiling_data, tiling_arg)    \
    AsdOps::MultinomialTilingData tiling_data;      \
    InitTilingData((tiling_arg), &(tiling_data))

extern "C" __global__ __aicore__ void multinomial(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMultinomial op;
    op.Init(x, z, tiling_data.realLastDim,
            tiling_data.expandLastDim,
            tiling_data.firstDim,
            tiling_data.randValList,
            tiling_data.numSamples,
            tiling_data.numSamplesMax,
            tiling_data.perCoreRunNum,
            tiling_data.nlElePerCorePerRun,
            tiling_data.lElePerCoreLastRun,
            tiling_data.tempUbEleAligened);
    op.Process(workspace);
}