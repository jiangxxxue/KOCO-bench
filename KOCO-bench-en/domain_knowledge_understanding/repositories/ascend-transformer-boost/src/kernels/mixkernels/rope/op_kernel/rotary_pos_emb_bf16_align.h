/*
* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ROTARY_POS_EMB_BF16_ALIGN
#define ROTARY_POS_EMB_BF16_ALIGN
#include "rotary_pos_emb_base.h"

using AscendC::HardEvent;
using AscendC::Cast;

template <typename QK_DTYPE, typename COS_DTYPE, bool IF_COS_BROADCAST>
class RopeBf16Align : public RopeBase<QK_DTYPE, COS_DTYPE, IF_COS_BROADCAST> {
public:
    __aicore__ inline RopeBf16Align(AtbOps::RopeTilingData *tilingData, AscendC::TPipe *pipe)
        :RopeBase<QK_DTYPE, COS_DTYPE, IF_COS_BROADCAST>(tilingData, pipe)
    {
        this->repeatSize_ = 64; // 64 = 256B / sizeof(float)
        // 3 is fp16 space needed; maxProcessNum_ = 3 * MAX_LEN_FP16 / 2 * NUM_TWO;
        this->maxProcessNum_ = 3 * MAX_LEN_FP16;
        this->repeatTimesQ_ = (this->tilingData_->hiddenSizeQ + this->repeatSize_ - 1) / this->repeatSize_;
        this->repeatTimesK_ = (this->tilingData_->hiddenSizeK + this->repeatSize_ - 1) / this->repeatSize_;
        headDimAlign_ = ((this->tilingData_->headDim + ELE_NUM_FP16 - 1) / ELE_NUM_FP16) * ELE_NUM_FP16;
        this->hiddenSizeAlign_ = ((this->hiddenSize_ +  this->repeatSize_ - 1) / this->repeatSize_) * this->repeatSize_;
        this->syncOffset_ = (this->tilingData_->headDim % ELE_NUM_FP16 == 0) ?
            this->hiddenSizeAlign_ : this->headNum_ * headDimAlign_;
        this->offsetExtraGm_ = NUM_TWO * this->blockIdx_ * this->syncOffset_;
        this->alignHalfHeadDim_ = (this->rotateStride_ * NUM_TWO) % ELE_NUM_FP32;
        // 使用pinpong， SLICE_SIZE / NUM_TWO
        sliceSizeTmp_ = (SLICE_SIZE / NUM_TWO / this->tilingData_->headDim) * this->tilingData_->headDim; // 向下取整
        uint32_t sliceSizeUb = (sliceSizeTmp_ + this->repeatSize_ - 1) / this->repeatSize_ * this->repeatSize_;
        // fp16
        this->oriPos_ = 0;
        this->removeBefore_ = this->oriPos_ + sliceSizeUb;
        this->padBefore_ = this->removeBefore_ + sliceSizeUb;
        this->QKpong_ = this->padBefore_ + sliceSizeUb;
        // fp32
        this->cosPad_ = 0;
        this->sinPad_ = this->cosPad_ + sliceSizeUb;
        this->negOne_ = this->sinPad_ + sliceSizeUb;
        this->posOneF32_ = this->negOne_+ sliceSizeUb;
        oriPosF32_ = this->posOneF32_ + sliceSizeUb;
        PadBeforeF32_ = oriPosF32_ + sliceSizeUb;
        removeBeforeF32_ = PadBeforeF32_ + sliceSizeUb;
        this->offsetPongF32_ = this->removeBeforeF32_ + sliceSizeUb;
        this->pipe_->InitBuffer(qkfp32QueueCO2_, (this->tilingData_->maxUbSize -
            (this->batchSize_  + this->maxProcessNum_) * sizeof(half))); // 留给fp32的
        this->pipe_->InitBuffer(outQueueCO2_, ((this->maxProcessNum_) * sizeof(half)));
        this->SliceCalculation(sliceSizeTmp_, sliceTimeQ_, lastSliceSizeQ_, sliceTimeK_, lastSliceSizeK_);
    }

    __aicore__ inline void ConvertCos(const AscendC::LocalTensor<float> &qkfp32Ubuf_,
                                      const AscendC::LocalTensor<QK_DTYPE> &commonUbuf_, uint32_t repeatTimes)
    {
        conv_v<ArchType::ASCEND_V220, QK_DTYPE, float>(qkfp32Ubuf_[this->cosPad_], commonUbuf_[this->cosPad_],
                                                       repeatTimes, 1, 1, DEFAULT_REPEAT_STRIDE,
                                                       DEFAULT_REPEAT_STRIDE / NUM_TWO);
        conv_v<ArchType::ASCEND_V220, QK_DTYPE, float>(qkfp32Ubuf_[this->sinPad_], commonUbuf_[this->sinPad_],
                                                       repeatTimes, 1, 1, DEFAULT_REPEAT_STRIDE,
                                                       DEFAULT_REPEAT_STRIDE / NUM_TWO);
    }

    __aicore__ inline void CastB162F32(const AscendC::LocalTensor<float> &qkfp32Ubuf_,
                                       const AscendC::LocalTensor<QK_DTYPE> &commonUbuf_, uint32_t repeatTimes1)
    {
        conv_v<ArchType::ASCEND_V220, QK_DTYPE, float>(qkfp32Ubuf_[oriPosF32_], commonUbuf_[this->oriPos_],
                                                       repeatTimes1, 1, 1, DEFAULT_REPEAT_STRIDE,
                                                       DEFAULT_REPEAT_STRIDE / NUM_TWO);
        conv_v<ArchType::ASCEND_V220, QK_DTYPE, float>(qkfp32Ubuf_[removeBeforeF32_], commonUbuf_[this->removeBefore_],
                                                       repeatTimes1, 1, 1, DEFAULT_REPEAT_STRIDE,
                                                       DEFAULT_REPEAT_STRIDE / NUM_TWO);
        conv_v<ArchType::ASCEND_V220, QK_DTYPE, float>(qkfp32Ubuf_[PadBeforeF32_], commonUbuf_[this->padBefore_],
                                                       repeatTimes1, 1, 1, DEFAULT_REPEAT_STRIDE,
                                                       DEFAULT_REPEAT_STRIDE / NUM_TWO);
    }

    __aicore__ inline void CastF322B16(const AscendC::GlobalTensor<QK_DTYPE> &dst,
        const AscendC::LocalTensor<QK_DTYPE> &src1,
        const AscendC::LocalTensor<float> &src,
        uint32_t repeatTimes1,
        uint32_t hiddenSize1)
    {
        convr_v<ArchType::ASCEND_V220, float, QK_DTYPE>(src1, src, repeatTimes1, 1, 1, DEFAULT_REPEAT_STRIDE / NUM_TWO,
                                                        DEFAULT_REPEAT_STRIDE);
        AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(dst, src1, {1, static_cast<uint16_t>(hiddenSize1 / ELE_NUM_FP16), 0, 0});
    } // S: To Cast

    __aicore__ inline void Process(__gm__ uint8_t *extra)
    {
        AscendC::GlobalTensor<uint8_t> extraGm;
        extraGm.SetGlobalBuffer((__gm__ uint8_t *)extra);
        AscendC::LocalTensor<QK_DTYPE> commonUbuf_ = outQueueCO2_.Get<QK_DTYPE>();
        AscendC::LocalTensor<float> qkfp32Ubuf_ = qkfp32QueueCO2_.Get<float>();
        if (this->tilingData_->cosFormat == 1) {
            AscendC::GlobalTensor<COS_DTYPE> extraGmCosDtype;
            extraGmCosDtype.SetGlobalBuffer((__gm__ COS_DTYPE *)extra);
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
        }

        uint32_t dynamicSliceQ = this->tilingData_->hiddenSizeQ > sliceSizeTmp_ ?
            sliceSizeTmp_ : this->tilingData_->hiddenSizeQ;
        uint32_t headNumTempQ = dynamicSliceQ / this->tilingData_->headDim;
        uint32_t dynamicSliceK = this->tilingData_->hiddenSizeK > sliceSizeTmp_ ?
            sliceSizeTmp_ : this->tilingData_->hiddenSizeK;
        uint32_t headNumTempK = dynamicSliceK / this->tilingData_->headDim;
        uint32_t repeatTemp = (dynamicSliceQ + this->repeatSize_ - 1) / this->repeatSize_;
        this->ExpandNeg(qkfp32Ubuf_, posOneF32_, headNumTempQ, repeatTemp);
        if (this->dynamicRound_ > 1) {
            this->ExpandNeg(qkfp32Ubuf_[this->offsetPongF32_], posOneF32_, headNumTempQ, repeatTemp);
        }

        AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID3);
        event_t pingPongEventID = EVENT_ID3;
        uint32_t isOffset = 1;
        for (uint32_t zz = 0; zz < this->dynamicRound_; ++zz) {
            pingPongEventID = (event_t)(pingPongEventID ^ 1);
            isOffset = 1 - isOffset;
            uint32_t offsetPingPong = isOffset * this->QKpong_; // 用于commonUbuf_
            uint32_t offsetPingPongF32 = isOffset * this->offsetPongF32_;   // 用于qkfp32Ubuf_
            AscendC::WaitFlag<HardEvent::V_MTE2>(pingPongEventID);
            this->CosSinBroadcast(extraGm, zz, commonUbuf_[offsetPingPong], dynamicSliceQ);
            AscendC::WaitFlag<HardEvent::MTE3_V>(pingPongEventID);
            if (this->tilingData_->headDim % ELE_NUM_FP16 == 0) {
                AscendC::PipeBarrier<PIPE_V>();
                ConvertCos(qkfp32Ubuf_[offsetPingPongF32], commonUbuf_[offsetPingPong], repeatTemp);
            } else {
                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                ConvertCos(qkfp32Ubuf_[offsetPingPongF32], commonUbuf_[offsetPingPong], repeatTemp);
            }

            for (uint32_t perSlice = 0; perSlice < sliceTimeQ_; ++perSlice) { // 核内每块
                uint32_t dynamicSliceQTemp = (perSlice == sliceTimeQ_ - 1) ? lastSliceSizeQ_ : sliceSizeTmp_;
                headNumTempQ = dynamicSliceQTemp / this->tilingData_->headDim;
                uint32_t repeatTimeOnce = (dynamicSliceQTemp + this->repeatSize_ - 1) / this->repeatSize_;
                AscendC::PipeBarrier<PIPE_MTE2>();
                AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                this->QkCommBF16(this->qGm_[this->blockIdx_ * this->nlCoreRun_ * this->tilingData_->hiddenSizeQ +
                                        zz * this->tilingData_->hiddenSizeQ + perSlice * sliceSizeTmp_],
                             extraGm, dynamicSliceQTemp, commonUbuf_[offsetPingPong], headNumTempQ);

                AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
                if (this->alignRotary_ == 0) {
                    AscendC::PipeBarrier<PIPE_V>();
                    CastB162F32(qkfp32Ubuf_[offsetPingPongF32], commonUbuf_[offsetPingPong], repeatTimeOnce);
                    AscendC::PipeBarrier<PIPE_V>();
                    this->CalcRopeAlign(qkfp32Ubuf_[offsetPingPongF32], repeatTimeOnce,
                                        oriPosF32_, removeBeforeF32_, PadBeforeF32_);
                } else {
                    AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                    AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                    CastB162F32(qkfp32Ubuf_[offsetPingPongF32], commonUbuf_[offsetPingPong], repeatTimeOnce);
                    AscendC::PipeBarrier<PIPE_V>();
                    this->CalcRope(qkfp32Ubuf_[offsetPingPongF32], repeatTimeOnce, oriPosF32_, removeBeforeF32_,
                        PadBeforeF32_, posOneF32_, PadBeforeF32_);
                }

                AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
                CastF322B16(this->outQGm_[this->blockIdx_ * this->nlCoreRun_ * this->tilingData_->hiddenSizeQ +
                                          zz * this->tilingData_->hiddenSizeQ + perSlice * sliceSizeTmp_],
                            commonUbuf_[offsetPingPong + this->padBefore_],
                            qkfp32Ubuf_[offsetPingPongF32 + this->PadBeforeF32_],
                            repeatTimeOnce,
                            dynamicSliceQTemp);
            }

            for (uint32_t perSlice = 0; perSlice < sliceTimeK_; ++perSlice) { // 核内每块
                uint32_t dynamicSliceKTemp = (perSlice == sliceTimeK_ - 1) ? lastSliceSizeK_ : sliceSizeTmp_;
                headNumTempK = dynamicSliceKTemp / this->tilingData_->headDim;
                uint32_t repeatTimeOnce = (dynamicSliceKTemp + this->repeatSize_ - 1) / this->repeatSize_;
                AscendC::PipeBarrier<PIPE_MTE2>();
                AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                this->QkCommBF16(this->kGm_[this->blockIdx_ * this->nlCoreRun_ * this->tilingData_->hiddenSizeK +
                    zz * this->tilingData_->hiddenSizeK + perSlice * sliceSizeTmp_],
                    extraGm,
                    dynamicSliceKTemp,
                    commonUbuf_[offsetPingPong],
                    headNumTempK);

                AscendC::SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
                if (this->alignRotary_ == 0) {
                    AscendC::PipeBarrier<PIPE_V>();
                    CastB162F32(qkfp32Ubuf_[offsetPingPongF32], commonUbuf_[offsetPingPong], repeatTimeOnce);
                    AscendC::PipeBarrier<PIPE_V>();
                    this->CalcRopeAlign(qkfp32Ubuf_[offsetPingPongF32], repeatTimeOnce,
                                        oriPosF32_, removeBeforeF32_, PadBeforeF32_);
                } else {
                    AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                    AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
                    CastB162F32(qkfp32Ubuf_[offsetPingPongF32], commonUbuf_[offsetPingPong], repeatTimeOnce);
                    AscendC::PipeBarrier<PIPE_V>();
                    this->CalcRope(qkfp32Ubuf_[offsetPingPongF32], repeatTimeOnce, oriPosF32_, removeBeforeF32_,
                        PadBeforeF32_, posOneF32_, PadBeforeF32_);
                }

                AscendC::SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
                AscendC::WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
                CastF322B16(this->outKGm_[this->blockIdx_ * this->nlCoreRun_ * this->tilingData_->hiddenSizeK +
                                          zz * this->tilingData_->hiddenSizeK + perSlice * sliceSizeTmp_],
                            commonUbuf_[offsetPingPong + this->padBefore_],
                            qkfp32Ubuf_[offsetPingPongF32 + PadBeforeF32_],
                            repeatTimeOnce,
                            dynamicSliceKTemp);
            }
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> qkfp32QueueCO2_;
    uint32_t oriPosF32_{0}; // fp32的buf中qk的位置
    uint32_t PadBeforeF32_{0}; // fp32的buf中保存qk[-x : hiddensize - x]
    uint32_t removeBeforeF32_{0}; // fp32的buf中保存qk[x : hiddensize + x]
    uint32_t posOneF32_{0}; // fp32的buf中0 0 0 1 1 1的位置
    uint32_t offsetPongF32_{0};
    uint32_t headDimAlign_; // 对齐的headDim
    uint32_t sliceTimeQ_; // 切分块的次数
    uint32_t lastSliceSizeQ_; // 最后一块的大小
    uint32_t sliceTimeK_;
    uint32_t lastSliceSizeK_;
    uint32_t sliceSizeTmp_;
    uint32_t QKpong_{0};
};

#endif