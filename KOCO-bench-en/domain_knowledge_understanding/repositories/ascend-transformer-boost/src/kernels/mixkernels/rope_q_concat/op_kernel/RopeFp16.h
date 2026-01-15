/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ROPR_Q_CONCAT_FP16
#define ROPR_Q_CONCAT_FP16

#include "RopeBase.h"

using AscendC::HardEvent;

template <typename QK_DTYPE, typename COS_DTYPE, bool IF_COS_BROADCAST>
class RopeFp16 : public RopeBase<QK_DTYPE, COS_DTYPE, IF_COS_BROADCAST> {
public:
    __aicore__ inline RopeFp16(const AtbOps::RopeQConcatTilingData *tilingData, AscendC::TPipe *pipe)
        : RopeBase<QK_DTYPE, COS_DTYPE, IF_COS_BROADCAST>(tilingData, pipe)
    {
        this->repeatSize_ = REPEAT_SIZE_FP32;
        this->rotateStride_ = this->tilingData_->headDim / this->tilingData_->rotaryCoeff;
        headBlockLen = static_cast<uint16_t>(this->tilingData_->headDim / ELE_NUM_FP16);
        headBlockLenFP32 = static_cast<uint16_t>(this->tilingData_->headDim / ELE_NUM_FP32);
        rotaryLen = static_cast<uint16_t>(this->rotateStride_ / ELE_NUM_FP32);
        concatBlockLen = static_cast<uint16_t>(this->tilingData_->concatSize / ELE_NUM_FP16);
        outLineOffset = this->tilingData_->headDim + this->tilingData_->concatSize;
        uint64_t dataNum = this->tilingData_->headDim * this->tilingData_->maxNPerLoopForUb;
        uint64_t dataSizeFp16 = dataNum * sizeof(QK_DTYPE);
        uint64_t dataSizeFp32 = dataNum * sizeof(float);
        uint64_t concatDataSize =
            this->tilingData_->concatSize * sizeof(QK_DTYPE) * this->tilingData_->maxNPerLoopForUb;

        this->pipe_->InitBuffer(inputQQueue, dataSizeFp16);
        this->pipe_->InitBuffer(inputQCastQueue, dataSizeFp32);
        this->pipe_->InitBuffer(reverseQQueue, dataSizeFp32);
        this->pipe_->InitBuffer(inputCosQueue, dataSizeFp16);
        this->pipe_->InitBuffer(inputSinQueue, dataSizeFp16);
        this->pipe_->InitBuffer(inputCosCastQueue, dataSizeFp32);
        this->pipe_->InitBuffer(inputSinCastQueue, dataSizeFp32);
        this->pipe_->InitBuffer(negQueue, dataSizeFp32);
        this->pipe_->InitBuffer(inputConcatQueue, dataSizeFp16);
    }

    __aicore__ inline void Process()
    {
        if (this->blockIdx_ > this->tilingData_->realCore - 1) {
            return;
        }
        uint64_t startCoreLineIndex =
            static_cast<uint64_t>(this->blockIdx_) * this->tilingData_->nlCoreRun; // 当前核处理head起始位置
        // 生成 [maxNPerLoopForUb,head_dim] 的 neg
        AscendC::LocalTensor<float> negLocal = negQueue.Get<float>();
        ExpandNeg(negLocal, this->tilingData_->maxNPerLoopForUb);
        // 遍历处理每轮数据
        for (uint64_t zz = 0; zz < this->loopTime; ++zz) {
            uint16_t loopN = (zz == this->loopTime - 1) ? this->lastLoopN : this->tilingData_->maxNPerLoopForUb;
            uint64_t startHead = startCoreLineIndex + zz * this->tilingData_->maxNPerLoopForUb;
            uint64_t endHead = startHead + loopN;
            // 搬入数据Q
            AscendC::LocalTensor<QK_DTYPE> inputQ = inputQQueue.Get<QK_DTYPE>();
            AscendC::LocalTensor<float> inputQCastFP32 = inputQCastQueue.Get<float>();
            AscendC::LocalTensor<float> reverseQ = reverseQQueue.Get<float>();
            uint64_t qOffset = startHead * this->tilingData_->headDim;
            CopyQGenReverseQ(inputQ, inputQCastFP32, reverseQ, qOffset, loopN);
            // 搬入数据cos/sin
            AscendC::LocalTensor<QK_DTYPE> inputCos = inputCosQueue.Get<QK_DTYPE>();
            AscendC::LocalTensor<QK_DTYPE> inputSin = inputSinQueue.Get<QK_DTYPE>();
            GenerateCosSin(inputCos, inputSin, startHead, loopN);
            AscendC::LocalTensor<float> inputCosCastFP32 = inputCosCastQueue.Get<float>();
            AscendC::LocalTensor<float> inputSinCastFP32 = inputSinCastQueue.Get<float>();
            Cast(inputCosCastFP32, inputCos, AscendC::RoundMode::CAST_NONE, loopN * this->tilingData_->headDim);
            Cast(inputSinCastFP32, inputSin, AscendC::RoundMode::CAST_NONE, loopN * this->tilingData_->headDim);
            pipe_barrier(PIPE_ALL);
            // 计算rope结果
            uint32_t repeatTime = this->tilingData_->headDim * loopN;
            AscendC::Mul(inputQCastFP32, inputCosCastFP32, inputQCastFP32, repeatTime);
            AscendC::Mul(reverseQ, negLocal, reverseQ, repeatTime);
            pipe_barrier(PIPE_ALL);
            AscendC::Mul(reverseQ, inputSinCastFP32, reverseQ, repeatTime);
            pipe_barrier(PIPE_ALL);
            AscendC::Add(inputQCastFP32, reverseQ, inputQCastFP32, repeatTime);
            pipe_barrier(PIPE_ALL);
            // 搬出rope结果
            CopyOutRopeQ(inputQCastFP32, startHead, loopN);
            // 搬入ConcatInput
            AscendC::LocalTensor<QK_DTYPE> inputConcat = inputConcatQueue.Get<QK_DTYPE>();
            uint64_t concatOffset = startHead * this->tilingData_->concatSize;
            uint64_t concatOutOffset = startHead * outLineOffset + this->tilingData_->headDim;
            AscendC::SetFlag<HardEvent::S_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<HardEvent::S_MTE2>(EVENT_ID1);
            DataCopy(inputConcat, this->concatInputGm_[concatOffset], {loopN, concatBlockLen, 0, 0});
            pipe_barrier(PIPE_ALL);
            // 搬出ConcatInput
            DataCopy(this->outRopeConcatGm_[concatOutOffset], inputConcat, {loopN, concatBlockLen, 0, headBlockLen});
            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline void CopyOutRopeQ(const AscendC::LocalTensor<float> &tempBufQ, uint64_t startHead, uint16_t loopN)
    {
        AscendC::LocalTensor<QK_DTYPE> inputQ = inputQQueue.Get<QK_DTYPE>();
        Cast(inputQ, tempBufQ, AscendC::RoundMode::CAST_RINT, loopN * this->tilingData_->headDim);
        pipe_barrier(PIPE_ALL);
        uint64_t outQOffset = startHead * outLineOffset;
        AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
        AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
        DataCopy(this->outRopeConcatGm_[outQOffset], inputQ, {loopN, headBlockLen, 0, concatBlockLen});
        pipe_barrier(PIPE_ALL);
    }

    template <typename BUF_TYPE>
    __aicore__ inline void GenerateCosSin(const AscendC::LocalTensor<BUF_TYPE> &tempBufCos,
                                          const AscendC::LocalTensor<BUF_TYPE> &tempBufSin, uint64_t startHead,
                                          uint16_t loopN)
    {
        uint64_t endHead = startHead + loopN;
        uint64_t startSinCosHeadIndex = startHead;
        uint64_t headRemain = startHead % this->tilingData_->headNumQ;
        uint64_t localStartAddr = 0;
        if (headRemain != 0) { // 需要前处理
            uint64_t preProcessHeadNum = this->tilingData_->headNumQ - headRemain;
            uint64_t needToProcesHead = preProcessHeadNum > loopN ? loopN : preProcessHeadNum;
            CopyCosSin(tempBufCos, tempBufSin, localStartAddr,
                       (startSinCosHeadIndex / this->tilingData_->headNumQ) * this->tilingData_->headDim,
                       needToProcesHead);
            startSinCosHeadIndex += needToProcesHead;
            localStartAddr += needToProcesHead * this->tilingData_->headDim;
        }
        // 循环迭代处理剩余数据
        if (startSinCosHeadIndex < endHead) {
            uint64_t startSinCosIndex = startSinCosHeadIndex / this->tilingData_->headNumQ;
            uint64_t endSinCosIndex = (endHead + this->tilingData_->headNumQ - 1) / this->tilingData_->headNumQ;
            for (uint64_t index = startSinCosIndex; index < endSinCosIndex; ++index) {
                // 尾数处理
                uint32_t repeatNum = index == endSinCosIndex - 1 ? endHead - index * this->tilingData_->headNumQ
                                                                 : this->tilingData_->headNumQ;
                CopyCosSin(tempBufCos, tempBufSin, localStartAddr, index * this->tilingData_->headDim, repeatNum);
                localStartAddr += this->tilingData_->headDim * this->tilingData_->headNumQ;
            }
        }
    }

    template <typename BUF_TYPE>
    __aicore__ inline void CopyPad(const AscendC::LocalTensor<BUF_TYPE> &tempBuf, uint16_t dstRepeatSize,
                                   uint32_t repeatTime)
    {
        uint32_t loopTime = repeatTime / MAX_REPEAT_TIME;
        for (uint32_t i = 0; i < loopTime; ++i) {
            Copy(tempBuf[this->tilingData_->headDim], tempBuf, this->tilingData_->headDim, MAX_REPEAT_TIME,
                 {1, 1, dstRepeatSize, 0});
        }
        uint32_t copyFinishCount = loopTime * MAX_REPEAT_TIME;
        uint32_t lastRepeatTime = repeatTime - copyFinishCount;
        if (lastRepeatTime > 0) {
            Copy(tempBuf[this->tilingData_->headDim + copyFinishCount * this->tilingData_->headDim], tempBuf,
                 this->tilingData_->headDim, lastRepeatTime, {1, 1, dstRepeatSize, 0});
        }
    }

    // 构建tensor -1 -1 -1 1 1 1
    template <typename BUF_TYPE>
    __aicore__ inline void ExpandNeg(const AscendC::LocalTensor<BUF_TYPE> &tempBuf, uint32_t headNumTemp)
    {
        for (uint32_t i = 0; i < this->rotateStride_; ++i) {
            tempBuf.SetValue(i, (BUF_TYPE)-1);
            tempBuf.SetValue(i + this->rotateStride_, (BUF_TYPE)1);
        }
        AscendC::SetFlag<HardEvent::S_V>(EVENT_ID1);
        AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID1);
        CopyPad(tempBuf, headBlockLenFP32, headNumTemp - 1);
        pipe_barrier(PIPE_ALL);
    }

    template <typename BUF_TYPE>
    __aicore__ inline void
    CopyQGenReverseQ(const AscendC::LocalTensor<BUF_TYPE> &tempBufQ, const AscendC::LocalTensor<float> &tempBufQCast,
                     const AscendC::LocalTensor<float> &tempBufRverseQ, uint64_t qOffset, uint16_t loopN)
    {
        AscendC::SetFlag<HardEvent::S_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<HardEvent::S_MTE2>(EVENT_ID1);
        // 搬入数据Q
        DataCopy(tempBufQ, this->qGm_[qOffset], {loopN, headBlockLen, 0, 0});
        pipe_barrier(PIPE_ALL);
        // cast fp32
        Cast(tempBufQCast, tempBufQ, AscendC::RoundMode::CAST_NONE, loopN * this->tilingData_->headDim);
        pipe_barrier(PIPE_ALL);
        // 搬入数据reverseQ
        DataCopy(tempBufRverseQ, tempBufQCast[this->rotateStride_], {loopN, rotaryLen, rotaryLen, rotaryLen});
        DataCopy(tempBufRverseQ[this->rotateStride_], tempBufQCast, {loopN, rotaryLen, rotaryLen, rotaryLen});
        pipe_barrier(PIPE_ALL);
    }

    template <typename BUF_TYPE>
    __aicore__ inline void CopyCosSin(const AscendC::LocalTensor<BUF_TYPE> &tempBufCos,
                                      const AscendC::LocalTensor<BUF_TYPE> &tempBufSin, uint64_t localStartAddr,
                                      uint64_t gmStartAddr, uint64_t repeatNum)
    {
        AscendC::SetFlag<HardEvent::S_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<HardEvent::S_MTE2>(EVENT_ID1);
        DataCopy(tempBufCos[localStartAddr], this->cosGm_[gmStartAddr], {1, headBlockLen, 0, 0});
        DataCopy(tempBufSin[localStartAddr], this->sinGm_[gmStartAddr], {1, headBlockLen, 0, 0});
        pipe_barrier(PIPE_ALL);
        CopyPad(tempBufCos[localStartAddr], headBlockLen, repeatNum - 1);
        CopyPad(tempBufSin[localStartAddr], headBlockLen, repeatNum - 1);
        pipe_barrier(PIPE_ALL);
    }

private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputQQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputQCastQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputCosQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputSinQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputCosCastQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputSinCastQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputConcatQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> negQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reverseQQueue;

    uint16_t headBlockLen{0};
    uint16_t headBlockLenFP32{0};
    uint16_t rotaryLen{0};
    uint16_t concatBlockLen{0};
    uint64_t outLineOffset{0};
};

#endif