/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ROTARY_POS_EMB_FP16
#define ROTARY_POS_EMB_FP16
#include "rotary_pos_emb_base.h"

using AscendC::HardEvent;

template <typename QK_DTYPE, typename COS_DTYPE, bool IF_COS_BROADCAST>
class RopeFp16 : public RopeBase<QK_DTYPE, COS_DTYPE, IF_COS_BROADCAST> {
public:
    __aicore__ inline RopeFp16(AtbOps::RopeTilingData *tilingData, AscendC::TPipe *pipe)
        :RopeBase<QK_DTYPE, COS_DTYPE, IF_COS_BROADCAST>(tilingData, pipe)
    {
        this->repeatSize_ = 128; // 128 = 256B / sizeof(half)
        this->maxProcessNum_ = this->tilingData_->maxUbSize / sizeof(uint16_t);
        this->repeatTimesQ_ = (this->tilingData_->hiddenSizeQ + this->repeatSize_ - 1) / this->repeatSize_;
        this->repeatTimesK_ = (this->tilingData_->hiddenSizeK + this->repeatSize_ - 1) / this->repeatSize_;
        headDimAlign_ = ((this->tilingData_->headDim + ELE_NUM_FP16 - 1) / ELE_NUM_FP16) * ELE_NUM_FP16;
        this->alignHalfHeadDim_ = (this->rotateStride_ * NUM_TWO) % ELE_NUM_FP16;
        this->hiddenSizeAlign_ = ((this->hiddenSize_ + this->repeatSize_ - 1) /
            this->repeatSize_) * this->repeatSize_;
        sliceSizeTmp_ = (SLICE_SIZE_FP16 / this->tilingData_->headDim) * this->tilingData_->headDim; // 向下取整 12096
        uint32_t sliceSizeUb = (sliceSizeTmp_ + this->repeatSize_ - 1) / this->repeatSize_ * this->repeatSize_;

        this->cosPad_ = 0;
        this->sinPad_ = this->cosPad_ + sliceSizeUb;
        this->negOne_ = this->sinPad_ + sliceSizeUb;
        this->oriPos_ = this->negOne_ + sliceSizeUb;
        this->padBefore_ = this->oriPos_ + sliceSizeUb;
        this->removeBefore_ = this->padBefore_ + sliceSizeUb;
        sinResPos_ = this->removeBefore_ + sliceSizeUb;
        this->repeatTimes_ = sliceSizeUb / this->repeatSize_;

        this->syncOffset_ = (this->tilingData_->headDim % ELE_NUM_FP16 == 0) ?
            sliceSizeUb : this->headNum_ * headDimAlign_;
        this->offsetExtraGm_ = NUM_TWO * this->blockIdx_ * this->syncOffset_;
        this->pipe_->InitBuffer(outQueueCO2_, ((this->maxProcessNum_ - this->batchSize_ * NUM_TWO) * sizeof(QK_DTYPE)));
        this->SliceCalculation(sliceSizeTmp_, sliceTimeQ_, lastSliceSizeQ_, sliceTimeK_, lastSliceSizeK_);
    }

    __aicore__ inline void CalcRotary(AscendC::LocalTensor<QK_DTYPE> commonUbuf_,
                                      uint32_t repeatTimeOnce)
    {
        if (this->alignRotary_ == 0) {
            AscendC::PipeBarrier<PIPE_V>();
            this->CalcRopeAlign(commonUbuf_, repeatTimeOnce,
                this->oriPos_, this->removeBefore_, this->padBefore_);
        } else {
            AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            this->CalcRope(commonUbuf_, repeatTimeOnce,
                           this->oriPos_,
                           this->removeBefore_,
                           this->padBefore_,
                           sinResPos_,
                           this->padBefore_);
        }
    }
    __aicore__ inline void CosFormatJudge(__gm__ uint8_t *extra,
                                          __gm__ uint8_t *sync)
    {
        AscendC::GlobalTensor<uint8_t> extraGm;
        extraGm.SetGlobalBuffer((__gm__ uint8_t *)extra);
        AscendC::LocalTensor<QK_DTYPE> commonUbuf_ = outQueueCO2_.Get<QK_DTYPE>();
        AscendC::GlobalTensor<COS_DTYPE> extraGmCosDtype;
        extraGmCosDtype.SetGlobalBuffer((__gm__ COS_DTYPE *)extra);
#if defined __CCE_AICORE__ == 100
        AscendC::PipeBarrier<PIPE_ALL>();
        this->ExpandCosSin(commonUbuf_, this->cosGm_, extraGmCosDtype);
        this->cosGm_ = extraGmCosDtype;
        AscendC::PipeBarrier<PIPE_ALL>();
        this->ExpandCosSin(commonUbuf_, this->sinGm_,
            extraGmCosDtype[this->tilingData_->ntokens * this->tilingData_->headDim]);
        this->sinGm_ = extraGmCosDtype[this->tilingData_->ntokens * this->tilingData_->headDim];
        extraGm = extraGm[this->tilingData_->ntokens * this->tilingData_->headDim *
                            4]; // sizeof(uint8_t) * 2 = sizeof(half)
        AscendC::PipeBarrier<PIPE_ALL>();
       
#else
       AscendC::GlobalTensor<int32_t> syncGm;
        syncGm.SetGlobalBuffer((__gm__ int32_t *)sync);
        if (block_idx == 0) {
            AscendC::PipeBarrier<PIPE_ALL>();
            this->ExpandCosSin(commonUbuf_, this->cosGm_, extraGmCosDtype);
            this->cosGm_ = extraGmCosDtype;
            AscendC::PipeBarrier<PIPE_ALL>();
            this->ExpandCosSin(commonUbuf_, this->sinGm_,
                extraGmCosDtype[this->tilingData_->ntokens * this->tilingData_->headDim]);
            this->sinGm_ = extraGmCosDtype[this->tilingData_->ntokens * this->tilingData_->headDim];
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        this->cosGm_ = extraGmCosDtype;
        this->sinGm_ = extraGmCosDtype[this->tilingData_->ntokens * this->tilingData_->headDim];
        extraGm = extraGm[this->tilingData_->ntokens * this->tilingData_->headDim *
                            4]; // sizeof(uint8_t) * 2 = sizeof(half)
        AscendC::LocalTensor<int32_t> syncBuf = outQueueCO2_.Get<int32_t>();
        AscendC::SyncAll(syncGm, syncBuf);
        
#endif
    }

    __aicore__ inline void Process(__gm__ uint8_t *extra, __gm__ uint8_t *sync)
    {
        AscendC::GlobalTensor<uint8_t> extraGm;
        extraGm.SetGlobalBuffer((__gm__ uint8_t *)extra);

        AscendC::LocalTensor<QK_DTYPE> commonUbuf_ = outQueueCO2_.Get<QK_DTYPE>();
        if (this->tilingData_->cosFormat == 1) {
            CosFormatJudge(extra, sync);
        }

        uint32_t headNumTempQ = this->tilingData_->hiddenSizeQ > sliceSizeTmp_ ?
            (sliceSizeTmp_ / this->tilingData_->headDim) : this->tilingData_->headNumQ;
        uint32_t dynamicSliceQ = this->tilingData_->hiddenSizeQ > sliceSizeTmp_ ?
            sliceSizeTmp_ : this->tilingData_->hiddenSizeQ;
        uint32_t headNumTempK = this->tilingData_->hiddenSizeK > sliceSizeTmp_ ?
            (sliceSizeTmp_ / this->tilingData_->headDim) : this->tilingData_->headNumK;
        uint32_t dynamicSliceK = this->tilingData_->hiddenSizeK > sliceSizeTmp_ ?
            sliceSizeTmp_ : this->tilingData_->hiddenSizeK;
        uint32_t repeatTemp = (dynamicSliceQ + this->repeatSize_ - 1) / this->repeatSize_;
        this->ExpandNeg(commonUbuf_, sinResPos_, headNumTempQ, repeatTemp); // 根据是否对齐选择1 -1 還是 -1 0
        for (uint32_t zz = 0; zz < this->dynamicRound_; ++zz) {
            this->CosSinBroadcast(extraGm, zz, commonUbuf_, dynamicSliceQ); // cos sin 和 QK 无关
            for (uint32_t perSlice = 0; perSlice < sliceTimeQ_; ++perSlice) { // 核内每块
                uint64_t outQOffset = static_cast<uint64_t>(block_idx) * this->nlCoreRun_ *
                    this->tilingData_->hiddenSizeQ + zz * this->tilingData_->hiddenSizeQ + perSlice * sliceSizeTmp_;

                uint32_t dynamicSliceQTemp = (perSlice == sliceTimeQ_ - 1) ? lastSliceSizeQ_ : sliceSizeTmp_;
                headNumTempQ = dynamicSliceQTemp / this->tilingData_->headDim;
                uint32_t repeatTimeOnce = (dynamicSliceQTemp + this->repeatSize_ - 1) / this->repeatSize_;
                this->QkComm(this->qGm_[outQOffset],
                            extraGm, dynamicSliceQTemp, commonUbuf_, headNumTempQ);
                CalcRotary(commonUbuf_, repeatTimeOnce);
                AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
                DataCopy(this->outQGm_[outQOffset],
                        commonUbuf_[this->padBefore_],
                        {1, static_cast<uint16_t>(dynamicSliceQTemp / ELE_NUM_FP16), 0, 0});

                AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            }

            for (uint32_t perSlice = 0; perSlice < sliceTimeK_; ++perSlice) {
                uint32_t dynamicSliceKTemp = (perSlice == sliceTimeK_ - 1) ? lastSliceSizeK_ : sliceSizeTmp_;
                uint64_t outKOffset = static_cast<uint64_t>(block_idx) * this->nlCoreRun_ *
                    this->tilingData_->hiddenSizeK + zz * this->tilingData_->hiddenSizeK + perSlice * sliceSizeTmp_;
                headNumTempK = dynamicSliceKTemp / this->tilingData_->headDim;
                uint32_t repeatTimeOnce = (dynamicSliceKTemp + this->repeatSize_ - 1) / this->repeatSize_;
                AscendC::PipeBarrier<PIPE_MTE2>();
                this->QkComm(this->kGm_[outKOffset],
                    extraGm,
                    dynamicSliceKTemp,
                    commonUbuf_,
                    headNumTempK);
                CalcRotary(commonUbuf_, repeatTimeOnce);
                AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
                DataCopy(this->outKGm_[outKOffset],
                        commonUbuf_[this->padBefore_],
                        {1, static_cast<uint16_t>(dynamicSliceKTemp / ELE_NUM_FP16), 0, 0});
                AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            }
        }
    }
private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> outQueueCO2_;
    uint32_t headDimAlign_; // 对齐的headDim
    uint32_t sinResPos_{0}; // fp32的buf中0 0 0 1 1 1的位置
    uint32_t sliceTimeQ_; // 切分块的次数
    uint32_t lastSliceSizeQ_; // 最后一块的大小
    uint32_t sliceTimeK_;
    uint32_t lastSliceSizeK_;
    uint32_t sliceSizeTmp_;
    uint32_t ResOut_;
};

#endif