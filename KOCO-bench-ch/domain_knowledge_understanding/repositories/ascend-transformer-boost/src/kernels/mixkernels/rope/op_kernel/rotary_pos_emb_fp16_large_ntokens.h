/*
* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ROTARY_POS_EMB_FP16_LARGE_NTOKENS
#define ROTARY_POS_EMB_FP16_LARGE_NTOKENS
#include "rotary_pos_emb_base.h"

using AscendC::HardEvent;

template <typename QK_DTYPE, typename COS_DTYPE, bool IF_COS_BROADCAST>
class RopeFp16LargeNtokens : public RopeBase<QK_DTYPE, COS_DTYPE, IF_COS_BROADCAST> {
public:
    __aicore__ inline RopeFp16LargeNtokens(AtbOps::RopeTilingData *tilingData, AscendC::TPipe *pipe)
        :RopeBase<QK_DTYPE, COS_DTYPE, IF_COS_BROADCAST>(tilingData, pipe)
    {
        this->repeatSize_ = 128; // 128 = 256B / sizeof(half)
        this->maxProcessNum_ = this->tilingData_->maxUbSize / sizeof(uint16_t);
        this->repeatTimesQ_ = (this->tilingData_->hiddenSizeQ + this->repeatSize_ - 1) / this->repeatSize_;
        this->repeatTimesK_ = (this->tilingData_->hiddenSizeK + this->repeatSize_ - 1) / this->repeatSize_;
        headDimAlign_ = ((this->realHeadDim_ + ELE_NUM_FP16 - 1) / ELE_NUM_FP16) * ELE_NUM_FP16 * this->multiple_;
        this->alignHalfHeadDim_ = (this->rotateStride_ * NUM_TWO) % ELE_NUM_FP16;
        this->hiddenSizeAlign_ = ((this->hiddenSize_ + this->repeatSize_ - 1) /
            this->repeatSize_) * this->repeatSize_;
        this->sliceSizeTmp_ = (SLICE_SIZE_FP16_LARGE_NTOKENS / this->tilingData_->headDim) *
            this->tilingData_->headDim; // 向下取整 12096
        uint32_t sliceSizeUb = (sliceSizeTmp_ + this->repeatSize_ - 1) / this->repeatSize_ * this->repeatSize_;

        this->cosPad_ = 0;
        this->sinPad_ = this->cosPad_ + sliceSizeUb;
        this->negOne_ = this->sinPad_ + sliceSizeUb;
        this->sinResPos_ = this->negOne_ + sliceSizeUb;

        this->oriPos_ = this->sinResPos_ + sliceSizeUb;
        this->padBefore_ = this->oriPos_ + sliceSizeUb;
        this->removeBefore_ = this->padBefore_ + sliceSizeUb;

        this->offsetSubPong_ = sliceSizeUb + sliceSizeUb + sliceSizeUb;
        this->offsetPong_ = this->removeBefore_ + this->offsetSubPong_ + sliceSizeUb;

        this->repeatTimes_ = sliceSizeUb / this->repeatSize_;

        this->syncOffset_ = (this->realHeadDim_ % ELE_NUM_FP16 == 0) ?
            sliceSizeUb : this->headNum_ * headDimAlign_;
        this->offsetExtraGm_ = this->blockIdx_ * NUM_FOUR * this->syncOffset_;
        this->offsetCosExtraGm_ = this->offsetExtraGm_;
        this->offsetSinExtraGm_ = this->offsetCosExtraGm_ + this->syncOffset_;
        this->offsetQKExtraGm_ = this->offsetSinExtraGm_ + this->syncOffset_;
        this->pipe_->InitBuffer(outQueueCO2_, ((this->maxProcessNum_ - this->batchSize_ * NUM_TWO) * sizeof(QK_DTYPE)));
        this->SliceCalculation(sliceSizeTmp_, sliceTimeQ_, lastSliceSizeQ_, sliceTimeK_, lastSliceSizeK_);
    }

    __aicore__ inline void CalcRotary(AscendC::LocalTensor<QK_DTYPE> commonUbuf_,
                                      uint32_t repeatTimeOnce, uint32_t offsetQK = 0)
    {
        if (this->alignRotary_ == 0) {
            AscendC::PipeBarrier<PIPE_V>();
            this->CalcRopeAlign(commonUbuf_, repeatTimeOnce,
                this->oriPos_ + offsetQK, this->removeBefore_ + offsetQK, this->padBefore_ + offsetQK);
        } else {
            AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            this->CalcRope(commonUbuf_, repeatTimeOnce,
                           this->oriPos_ + offsetQK,
                           this->removeBefore_ + offsetQK,
                           this->padBefore_ + offsetQK,
                           sinResPos_,
                           this->padBefore_ + offsetQK);
        }
    }

    struct CalcOutQKParams {
        AscendC::GlobalTensor<uint8_t> extraGm;
        AscendC::LocalTensor<QK_DTYPE> commonUbuf;
        AscendC::GlobalTensor<QK_DTYPE> Gm;
        AscendC::GlobalTensor<QK_DTYPE> outGm;
        uint32_t calQkCommNum;
        uint32_t roundIdx;
        uint32_t sliceTime;
        uint32_t hiddenSize;
        uint32_t lastSliceSize;
        uint32_t offsetPingPong;
    };

    __aicore__ inline void CalcOutQK(CalcOutQKParams &calcOutQKParams)
    {
        for (uint32_t perSlice = 0; perSlice < calcOutQKParams.sliceTime; ++perSlice) {
            uint32_t offsetSubPingPong = calcOutQKParams.calQkCommNum % 2 * this->offsetSubPong_;
            uint64_t outOffset = static_cast<uint64_t>(block_idx) * this->nlCoreRun_ *
                calcOutQKParams.hiddenSize + calcOutQKParams.roundIdx * calcOutQKParams.hiddenSize +
                perSlice * sliceSizeTmp_;

            uint32_t dynamicSliceTemp = (perSlice == calcOutQKParams.sliceTime - 1) ? calcOutQKParams.lastSliceSize :
                sliceSizeTmp_;
            uint32_t headNumTemp = dynamicSliceTemp / this->tilingData_->headDim;
            uint32_t repeatTimeOnce = (dynamicSliceTemp + this->repeatSize_ - 1) / this->repeatSize_;
            AscendC::PipeBarrier<PIPE_MTE2>();

            bool isWait = calcOutQKParams.calQkCommNum % 2 == 0 && calcOutQKParams.calQkCommNum > 0;
            if (isWait) {
                AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
            }

            QkCommLargeNtokensParams qkParams = {dynamicSliceTemp, headNumTemp, offsetSubPingPong};
            this->QkCommLargeNtokens(calcOutQKParams.Gm[outOffset],
                calcOutQKParams.extraGm,
                calcOutQKParams.commonUbuf[calcOutQKParams.offsetPingPong],
                qkParams);

            if (isWait) {
                AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
            }
            CalcRotary(calcOutQKParams.commonUbuf[calcOutQKParams.offsetPingPong], repeatTimeOnce, offsetSubPingPong);

            AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::PipeBarrier<PIPE_MTE3>();

            DataCopy(calcOutQKParams.outGm[outOffset], calcOutQKParams.commonUbuf[calcOutQKParams.offsetPingPong +
                offsetSubPingPong + this->padBefore_],
                {1, static_cast<uint16_t>(dynamicSliceTemp / ELE_NUM_FP16), 0, 0});

            ++calcOutQKParams.calQkCommNum;
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

    __aicore__ inline void RepeatParamCalculation(uint32_t &headNumTempQ, uint32_t &dynamicSliceQ,
                                                  uint32_t &headNumTempK, uint32_t &dynamicSliceK,
                                                  uint32_t &repeatTemp)
    {
        headNumTempQ = this->tilingData_->hiddenSizeQ > sliceSizeTmp_ ?
            (sliceSizeTmp_ / this->tilingData_->headDim) : this->tilingData_->headNumQ;
        dynamicSliceQ = this->tilingData_->hiddenSizeQ > sliceSizeTmp_ ?
            sliceSizeTmp_ : this->tilingData_->hiddenSizeQ;
        headNumTempK = this->tilingData_->hiddenSizeK > sliceSizeTmp_ ?
            (sliceSizeTmp_ / this->tilingData_->headDim) : this->tilingData_->headNumK;
        dynamicSliceK = this->tilingData_->hiddenSizeK > sliceSizeTmp_ ?
            sliceSizeTmp_ : this->tilingData_->hiddenSizeK;
        repeatTemp = (dynamicSliceQ + this->repeatSize_ - 1) / this->repeatSize_;
    }

    __aicore__ inline void Process(__gm__ uint8_t *extra, __gm__ uint8_t *sync)
    {
        AscendC::GlobalTensor<uint8_t> extraGm;
        extraGm.SetGlobalBuffer((__gm__ uint8_t *)extra);

        AscendC::LocalTensor<QK_DTYPE> commonUbuf_ = outQueueCO2_.Get<QK_DTYPE>();
        if (this->tilingData_->cosFormat == 1) {
            CosFormatJudge(extra, sync);
        }

        uint32_t headNumTempQ = 0;
        uint32_t dynamicSliceQ = 0;
        uint32_t headNumTempK = 0;
        uint32_t dynamicSliceK = 0;
        uint32_t repeatTemp = 0;
        RepeatParamCalculation(headNumTempQ, dynamicSliceQ, headNumTempK, dynamicSliceK, repeatTemp);
        this->ExpandNeg(commonUbuf_, sinResPos_, headNumTempQ, repeatTemp); // 根据是否对齐选择1 -1 還是 -1 0
        if (this->dynamicRound_ > 1)
            this->ExpandNeg(commonUbuf_[this->offsetPong_], sinResPos_, headNumTempQ, repeatTemp);

        AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID3);
        event_t pingPongEventID = EVENT_ID3;
        uint32_t isOffset = 1;
        for (uint32_t roundIdx = 0; roundIdx < this->dynamicRound_; ++roundIdx) {
            pingPongEventID = (event_t)(pingPongEventID ^ 1);
            isOffset = 1 - isOffset;
            uint32_t offsetPingPong = isOffset * this->offsetPong_;
            AscendC::WaitFlag<HardEvent::V_MTE2>(pingPongEventID);
            this->CosSinBroadcast(extraGm, roundIdx, commonUbuf_[offsetPingPong], dynamicSliceQ); // cos sin 和 QK 无关
            AscendC::WaitFlag<HardEvent::MTE3_V>(pingPongEventID);

            // 计算Q
            CalcOutQKParams calcOutQKParams = {
                extraGm, commonUbuf_, this->qGm_, this->outQGm_, 0, roundIdx, sliceTimeQ_,
                this->tilingData_->hiddenSizeQ, lastSliceSizeQ_, offsetPingPong};
            CalcOutQK(calcOutQKParams);

            // 计算K
            if (this->multiple_ > 1) {
                AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                this->CosSinBroadcast(extraGm, roundIdx, commonUbuf_[offsetPingPong], dynamicSliceK);
            }
            calcOutQKParams = {
                extraGm, commonUbuf_, this->kGm_, this->outKGm_, calcOutQKParams.calQkCommNum, roundIdx, sliceTimeK_,
                this->tilingData_->hiddenSizeK, lastSliceSizeK_, offsetPingPong};
            CalcOutQK(calcOutQKParams);

            AscendC::SetFlag<HardEvent::V_MTE2>(pingPongEventID);
            AscendC::SetFlag<HardEvent::MTE3_V>(pingPongEventID);
        }
        AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID3);
    }
private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> outQueueCO2_;
    uint32_t headDimAlign_; // 对齐的headDim
    uint32_t sinResPos_{0}; // fp32的buf中0 0 0 1 1 1的位置
    uint32_t offsetSubPong_{0};   // QK Pong向量空间的偏移量
    uint32_t offsetPong_{0}; // 第二块Pong空间的偏移量, 0~offsetPong_是Ping的空间, offsetPong_~2*offsetPong_是Pong的空间
    uint32_t sliceTimeQ_; // 切分块的次数
    uint32_t lastSliceSizeQ_; // 最后一块的大小
    uint32_t sliceTimeK_;
    uint32_t lastSliceSizeK_;
    uint32_t sliceSizeTmp_;
    uint32_t ResOut_;
};

#endif