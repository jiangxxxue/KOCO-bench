/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ASCEND_RESHAPE_AND_CACHE_BASE_H
#define ASCEND_RESHAPE_AND_CACHE_BASE_H

#include "kernel_operator.h"
#include "mixkernels/utils/common/kernel/kernel_utils.h"
#include "mixkernels/reshape_and_cache/tiling/reshape_and_cache_tiling_dependency.h"

using namespace AtbOps;

constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t SCALAR_INPUT_PARAMETERS = 4;  // 算子输入input3~7是四个int32类型的参数
constexpr uint32_t COPY_NUM = 512; // 用于设置计算rope压缩均值前的搬运大小
constexpr uint32_t MAX_UB_SIZE = 192 * 1024; // UB size
constexpr int16_t MAX_MASK_NUM = 64;    // count模式下ADD计算最多处理256B，对应float元素为64个
constexpr int16_t MIN_MASK_NUM = 32;    // 当前约束headsize为32的整数倍，设置32为了约束范围内尽可能提高计算带宽。
constexpr int32_t MAX_FLOAT_NUM = 30 * 1024;   // 考虑充分利用UB空间，localtensor可申请的最大浮点数个数
constexpr int32_t PER_FLOAT_NUM = 15 * 1024;    // 开启doublebuffer后，一个buffer内的浮点数个数
constexpr uint32_t TOTAL_BUFFER_SIZE = 180 * 1024;    // 留出一部分空间给标量

class ReshapeAndCacheBase {
public:
    __aicore__ inline ReshapeAndCacheBase() {}

    template <typename T = ReshapeAndCacheTilingData, bool NCT = false>
    __aicore__ inline void Init(AscendC::TPipe *pipe, T *tilingData)
    {
        pipe_ = pipe;
        // init tiling

        numTokens_ = tilingData->numTokens;
        numHeads_ = tilingData->numHeads;
        headSizeK_ = tilingData->headSizeK;
        headSizeV_ = tilingData->headSizeV;
        numBatchs_ = tilingData->numBatchs;
        if constexpr (NCT) {
            offsetK_ = tilingData->offsetK;
            offsetV_ = tilingData->offsetV;
            stride_ = tilingData->stride;
        }
    }

    template <typename T>
    __aicore__ inline void InitGlobalTensor(AscendC::GlobalTensor<T> &gm, GM_ADDR addr)
    {
        gm.SetGlobalBuffer((__gm__ T *)addr);
    }

    template <typename T, typename B>
    __aicore__ inline void InitTBuf(B &buf, uint32_t len)
    {
        pipe_->InitBuffer(buf, RoundUp(len * sizeof(T), BLOCK_SIZE));
    }

    template <typename T, typename B>
    __aicore__ inline void InitScalarTBuf(B &buf, uint32_t len)
    {
        pipe_->InitBuffer(buf, SCALAR_INPUT_PARAMETERS * RoundUp(len * sizeof(T), BLOCK_SIZE));
    }

    __aicore__ inline void AllocateTask()
    {
        coreNums_ = static_cast<uint32_t>(AscendC::GetBlockNum());
        perCoreTaskNum_ = numTokens_ / coreNums_;
        tailTaskNum_ = numTokens_ % coreNums_;
        blockId_ = static_cast<uint32_t>(AscendC::GetBlockIdx());
        startTaskId_ = blockId_ * perCoreTaskNum_;

        if (blockId_ < tailTaskNum_) {
            perCoreTaskNum_++;
            startTaskId_ += blockId_;
        } else {
            startTaskId_ += tailTaskNum_;
        }
    }

        __aicore__ inline void AllocateTaskRope()
    {
        coreNums_ = static_cast<uint32_t>(AscendC::GetBlockNum());
        perCoreTaskNum_ = numTokens_ * TASK_MULTIPLE / coreNums_;
        tailTaskNum_ = numTokens_ * TASK_MULTIPLE % coreNums_;
        blockId_ = static_cast<uint32_t>(AscendC::GetBlockIdx());
        startTaskId_ = blockId_ * perCoreTaskNum_;

        if (blockId_ < tailTaskNum_) {
            perCoreTaskNum_++;
            startTaskId_ += blockId_;
        } else {
            startTaskId_ += tailTaskNum_;
        }
    }

    template <typename T>
    __aicore__ inline void CopyKvCache(
        AscendC::GlobalTensor<T>& src,
        AscendC::LocalTensor<T>& ubAddr,
        AscendC::GlobalTensor<T>& dst,
        uint64_t start,
        uint64_t cacheStart,
        AscendC::DataCopyParams& copyParamsIn,
        AscendC::DataCopyParams& copyParamsOut)
    {
        DataCopy(ubAddr, src[start], copyParamsIn);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
        DataCopy(dst[cacheStart], ubAddr, copyParamsOut);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    template <typename T>
    __aicore__ inline void CopyKvCacheDoubleBuf(
        AscendC::GlobalTensor<T>& src,
        AscendC::LocalTensor<T>& ubAddr,
        AscendC::GlobalTensor<T>& dst,
        event_t eventID,
        uint64_t start,
        uint64_t cacheStart,
        AscendC::DataCopyParams& copyParamsIn,
        AscendC::DataCopyParams& copyParamsOut)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventID);
        DataCopy(ubAddr, src[start], copyParamsIn);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventID);
        DataCopy(dst[cacheStart], ubAddr, copyParamsOut);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventID);
    }

    template <typename T1, bool NCT = false>
    __aicore__ inline void CopyToCache(
        uint32_t index, uint32_t tokenSize,
        uint32_t loop, uint32_t tail,
        AscendC::GlobalTensor<T1>& src,
        AscendC::GlobalTensor<T1>& dst,
        AscendC::GlobalTensor<int32_t>& slotInput, const int32_t &maxUbUsed,
        AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, BUFFER_NUM>& queBind)
    {
        int64_t slotValue = (int64_t)(slotInput.GetValue(index + startTaskId_));
        if (slotValue < 0) {
            return;
        }
        uint64_t cacheStart = static_cast<uint64_t>(slotValue) * tokenSize;
        AscendC::DataCopyParams copyParams = {1, static_cast<uint16_t>(maxUbUsed / BLOCK_SIZE), 0, 0};
        uint64_t start = static_cast<uint64_t>(tokenSize) * (index + startTaskId_);
        if (NCT) {
            start = static_cast<uint64_t>(stride_[0]) * (index + startTaskId_);
        }
        for (uint32_t j = 0; j < loop; j++) {
            auto bindLocal = queBind.AllocTensor<T1>();
            DataCopy(bindLocal, src[start], copyParams);
            queBind.EnQue(bindLocal);
            bindLocal = queBind.DeQue<T1>();
            DataCopy(dst[cacheStart], bindLocal, copyParams);
            queBind.FreeTensor(bindLocal);
            start += (maxUbUsed / sizeof(T1));
            cacheStart += static_cast<uint64_t>(maxUbUsed / sizeof(T1));
        }
        if (tail > 0) {
            AscendC::DataCopyExtParams copyParam = {1, tail, 0, 0, 0};
            AscendC::DataCopyPadExtParams<T1> padParams = {false, 0, 0, 0};
            auto bindLocal = queBind.AllocTensor<T1>();
            DataCopyPad(bindLocal, src[start], copyParam, padParams);
            queBind.EnQue(bindLocal);
            bindLocal = queBind.DeQue<T1>();
            DataCopyPad(dst[cacheStart], bindLocal, copyParam);
            queBind.FreeTensor(bindLocal);
        }
    }

    template <typename T>
    __aicore__ inline void PrepareCopy(
        uint32_t tokenSize, uint32_t &loop,
        uint32_t &tail, const int32_t &maxUbUsed,
        AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, BUFFER_NUM>& queBind)
    {
        loop = (tokenSize * sizeof(T)) / maxUbUsed;
        tail = (tokenSize * sizeof(T)) % maxUbUsed;
        pipe_->InitBuffer(queBind, BUFFER_NUM, maxUbUsed);
    }

    template <typename T>
    __aicore__ inline void RopeInitscalarBuf()
    {
        uint32_t scalarBufStart = 0;
        tokenSize_ = 1 * static_cast<uint64_t>(headSizeK_);
        numBlocks_ = static_cast<uint32_t>(tokenSize_) * sizeof(T) / BLOCK_SIZE;
        copyFp16Num = TOTAL_BUFFER_SIZE / sizeof(T) / BUFFER_NUM;    // partA和partC搬运的数据量
 
        InitScalarTBuf<int32_t>(scalarBuf_, numHeads_ * numBatchs_);
        scalarLocal_ = scalarBuf_.Get<int32_t>();
 
        winsLocal_ = scalarLocal_[scalarBufStart];
        scalarBufStart += RoundUp(numHeads_ * numBatchs_ * sizeof(int32_t), BLOCK_SIZE) / sizeof(int32_t);
 
        seqLenLocal_ = scalarLocal_[scalarBufStart];
        scalarBufStart += RoundUp(numBatchs_ * sizeof(int32_t), BLOCK_SIZE) / sizeof(int32_t);
 
        csumSeqLenLocal_ = scalarLocal_[scalarBufStart];
        scalarBufStart += RoundUp(numBatchs_ * sizeof(int32_t), BLOCK_SIZE) / sizeof(int32_t);
        
        offsetLocal_ = scalarLocal_[scalarBufStart];
    }

    template <typename T>
    __aicore__ inline void RopeInitBuf(
        AscendC::LocalTensor<T>& copyLocalPing_,
        AscendC::LocalTensor<T>& copyLocalPong_,
        AscendC::LocalTensor<T>& tokenLocal_,
        AscendC::LocalTensor<float>& tokenCompressLocal_,
        AscendC::LocalTensor<float>& tokenSumLocal_,
        uint16_t& repeatDivisor, uint16_t& addNumPerLoop)
    {
        uint32_t computeBufStart = 0;
        uint32_t copyBufStart = 0;

        // 掩码模式一次计算256B，需要考虑对齐
        repeatDivisor = (tokenSize_ <= MAX_MASK_NUM && tokenSize_ % MAX_MASK_NUM == 0) ? MAX_MASK_NUM : MIN_MASK_NUM;
        addNumPerLoop = PER_FLOAT_NUM / tokenSize_;

        InitTBuf<T>(totalBuf_, TOTAL_BUFFER_SIZE / sizeof(T));
        tokenLocal_ = totalBuf_.GetWithOffset<T>(static_cast<uint32_t>(MAX_FLOAT_NUM), computeBufStart);
        
        computeBufStart += MAX_FLOAT_NUM * sizeof(T);
        tokenCompressLocal_ = totalBuf_.GetWithOffset<float>(static_cast<uint32_t>(MAX_FLOAT_NUM), computeBufStart);

        computeBufStart += MAX_FLOAT_NUM * sizeof(float);
        tokenSumLocal_ = totalBuf_.GetWithOffset<float>(static_cast<uint32_t>(MAX_FLOAT_NUM), computeBufStart);

        copyLocalPing_ = totalBuf_.GetWithOffset<T>(static_cast<uint32_t>(copyFp16Num), copyBufStart);
        copyBufStart += TOTAL_BUFFER_SIZE / BUFFER_NUM;
        copyLocalPong_ = totalBuf_.GetWithOffset<T>(static_cast<uint32_t>(copyFp16Num), copyBufStart);
        }

    template <typename T>
    __aicore__ inline void OmniInitBuf(
        AscendC::LocalTensor<T>& copyLocalPing_,
        AscendC::LocalTensor<T>& copyLocalPong_)
    {
        uint32_t copyBufStart = 0;
        InitTBuf<T>(totalBuf_, TOTAL_BUFFER_SIZE / sizeof(T));

        copyLocalPing_ = totalBuf_.GetWithOffset<T>(static_cast<uint32_t>(copyFp16Num), copyBufStart);
        copyBufStart += TOTAL_BUFFER_SIZE / BUFFER_NUM;
        copyLocalPong_ = totalBuf_.GetWithOffset<T>(static_cast<uint32_t>(copyFp16Num), copyBufStart);
        }

    __aicore__ inline void PreProcess(
        AscendC::GlobalTensor<int32_t>& winsInputGt_,
        AscendC::GlobalTensor<int32_t>& seqLenInputGt_,
        AscendC::GlobalTensor<int32_t>& offsetInputGt_)
    {
        DataCopy(winsLocal_, winsInputGt_, RoundUp(numHeads_ * numBatchs_, BLOCK_SIZE / sizeof(int32_t)));
        AscendC::PipeBarrier<PIPE_MTE2>();
        DataCopy(seqLenLocal_, seqLenInputGt_, RoundUp(numBatchs_, BLOCK_SIZE / sizeof(int32_t)));
        AscendC::PipeBarrier<PIPE_MTE2>();
        DataCopy(offsetLocal_, offsetInputGt_, RoundUp(numHeads_ * numBatchs_, BLOCK_SIZE / sizeof(int32_t)));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);
        csumSeqLenLocal_.SetValue(0, 0);
        for (uint32_t i = 1; i < numBatchs_; i++) { // 获取累加的seqlen
            csumSeqLenLocal_.SetValue(i, csumSeqLenLocal_.GetValue(i-1) + seqLenLocal_.GetValue(i-1));
        }
    }

    __aicore__ inline void ParamSet(
        AscendC::GlobalTensor<int32_t>& slotInputGt_,
        uint32_t startTaskOffset)
    {
        slotValue = slotInputGt_.GetValue(startTaskOffset);
        batchId = (startTaskOffset) / numHeads_;
        headId = (startTaskOffset) % numHeads_;
        headWin = static_cast<uint32_t>(winsLocal_.GetValue(startTaskOffset));
        seqLen = static_cast<uint32_t>(seqLenLocal_.GetValue(batchId));
        consumSeqLen = static_cast<uint32_t>(csumSeqLenLocal_.GetValue(batchId));
        offsetPerLine = (numHeads_ - 1) * numBlocks_;  // 每一个token间隔(num_heads-1)*tokenSize 此处单位为32B
        offsetIdx = offsetLocal_.GetValue(startTaskOffset);
        offsetIdx = (offsetIdx == -1 || headWin == 0) ? seqLen : offsetIdx; // 当等于-1时, 跳过当前头压缩
    }

protected:

    uint32_t numTokens_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t headSizeK_ = 0;
    uint32_t headSizeV_ = 0;
    uint32_t numBatchs_ = 0;

    uint32_t coreNums_ = 0;
    uint32_t perCoreTaskNum_ = 0;
    uint32_t tailTaskNum_ = 0;
    uint32_t blockId_ = 0;
    uint32_t startTaskId_ = 0;
    uint32_t offsetK_ = 0;
    uint32_t offsetV_ = 0;
    uint32_t *stride_{nullptr};

    uint64_t tokenSize_ = 0;
    uint32_t numBlocks_ = 0;
    uint32_t copyFp16Num = 0;
    AscendC::LocalTensor<int32_t> scalarLocal_; // 存放下面四个标量的总LocalTensor
    AscendC::LocalTensor<int32_t> winsLocal_; // 临时存放 wins
    AscendC::LocalTensor<int32_t> seqLenLocal_; // 临时存放 seqLen
    AscendC::LocalTensor<int32_t> csumSeqLenLocal_; // 临时存放 comsumSeqLen
    AscendC::LocalTensor<int32_t> offsetLocal_; // 临时存放 offset
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf_, totalBuf_;

    uint32_t slotValue = 0;
    uint32_t batchId = 0;
    uint32_t headId = 0;
    uint32_t headWin = 0;
    uint32_t seqLen = 0;
    uint32_t consumSeqLen = 0;
    uint32_t offsetPerLine = 0;
    int32_t offsetIdx = 0;

    AscendC::TPipe* pipe_;
};

template <typename T = ReshapeAndCacheTilingData, bool NCT = false>
inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, T *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220) || (__CCE_AICORE__ == 300)
    tilingdata->numTokens = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->numHeads = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->headSizeK = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
    tilingdata->headSizeV = (*(const __gm__ uint32_t *)(p_tilingdata + 12));
    tilingdata->blockSize = (*(const __gm__ uint32_t *)(p_tilingdata + 16));
    tilingdata->typeByte = (*(const __gm__ uint32_t *)(p_tilingdata + 20));
    tilingdata->numBatchs = (*(const __gm__ uint32_t *)(p_tilingdata + 24));
    if constexpr (NCT) {
        tilingdata->offsetK = (*(const __gm__ uint32_t *)(p_tilingdata + 28));
        tilingdata->offsetV = (*(const __gm__ uint32_t *)(p_tilingdata + 32));
        for (uint32_t i = 0; i < SHAPE_DIMS; i++) {
        tilingdata->stride[i] = (*(const __gm__ uint32_t *)(p_tilingdata + 36 + i * sizeof(uint32_t)));
        }
    }
#endif
}

#define GET_TILING_DATA(tilingData, tilingArg)    \
    ReshapeAndCacheTilingData tilingData;               \
    InitTilingData(tilingArg, &(tilingData))
#endif