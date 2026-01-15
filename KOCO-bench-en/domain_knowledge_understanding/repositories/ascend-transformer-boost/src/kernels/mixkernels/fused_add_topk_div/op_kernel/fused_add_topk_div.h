/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef FUSED_ADD_TOPK_DIV_H_
#define FUSED_ADD_TOPK_DIV_H_

#include "fused_add_topk_div_assist.h"
#include "mixkernels/fused_add_topk_div/tiling/tiling_data.h"
#include "kernels/utils/kernel/common.h"
#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t BASE_COUNT = 256;
constexpr uint32_t REPEAT_BYTES = 256;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t SORT_UNIT = 32;
constexpr uint32_t ADD_COUNT_THIRTY_TWO = 32;
constexpr uint32_t ADD_COUNT_SIXTY_FOUR = 64;
constexpr uint32_t ADD_COUNT_ONE_TWENTY_EIGHT = 128;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t BUFFER_NUM_ONE = 1;
constexpr float FLOAT32_NEG_INF = -3.4e38;
constexpr half FLOAT16_NEG_INF = -65504;
constexpr uint32_t BROADCAST_DIM = 2;
constexpr uint32_t BROADCAST_AXIS = 1;
constexpr uint32_t SORTED_COEF = 2;
constexpr int32_t FLOAT_BYTES = 4;
constexpr uint8_t REPEAT_STRIDE_EIGHT = 8;

template <typename inputT, typename calT, uint32_t enableExpertMapping>
class FusedAddTopkDiv {
public:
    __aicore__ inline FusedAddTopkDiv(){};
    __aicore__ inline void InitTilingData(AtbOps::FusedAddTopkDivTilingData &__restrict tilingData, GM_ADDR x,
        GM_ADDR addNum, GM_ADDR mappingNum, GM_ADDR mappingTable, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace);
    __aicore__ inline void InitBuffer(TPipe *inputPipe);
    __aicore__ inline void Process();
    __aicore__ inline void CopyInAddNum();
    __aicore__ inline void CopyInX(const int32_t loop);
    __aicore__ inline void ActivateAndAdd();
    __aicore__ inline void GroupTopkImpl();
    __aicore__ inline void GroupReduceSumInternelImpl();
    __aicore__ inline void GatherSigmoidImpl();
    __aicore__ inline void NormImpl();
    __aicore__ inline void CopyFromWorkspace();
    __aicore__ inline void CopyToWorkspace();
    __aicore__ inline void CopyOut(const int32_t loop);
    __aicore__ inline void CopyInMappingNum();
 
    __aicore__ inline void ProcessSortAlign();
 
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b)
    {
        return (a + b - 1) / b;
    };
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilAlign(T1 a, T2 b)
    {
        return (a + b - 1) / b * b;
    };

private:
    TPipe *pipe_;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQueue_;
    TBuf<TPosition::VECCALC> addNumInQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yOutQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> indicesOutQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM_ONE> assistQueue_;
    TBuf<TPosition::VECCALC> sigmoidBuf_;
    TQue<QuePosition::VECIN, BUFFER_NUM_ONE> sigmoidAddQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM_ONE> sortedQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM_ONE> topkValueQueue_;
    TBuf<TPosition::VECCALC> tempBuf_;

    TQue<QuePosition::VECIN, BUFFER_NUM_ONE> mappingNumQueue_;

    uint32_t secondDimSize_ = 0;
    uint32_t groupNum_ = 0;
    uint32_t groupTopk_ = 1;
    uint32_t n_ = 1;
    uint32_t k_ = 0;
    uint32_t activateType_ = 0;
    uint32_t isNorm_ = 1;
    float scale_ = 1.0;
    uint32_t groupEles_ = 0;
    uint32_t expertNum_ = 0;
    uint32_t tableDim_ = 0;
    int64_t outBatchStride_ = 0;
    int64_t batchOffset_ = 0;
    uint32_t loopBatch_ = 0;

    uint32_t groupElesAlignBlockCountFp32_ = 0;
    uint32_t groupElesAlignSortCount_ = 0;
    uint32_t secondAlignBlockCountFp32_ = 0;
    int64_t wsOffset_ = 0;
    uint32_t sortRepeatTimes_ = 1;
    uint32_t wholeSortNum_ = 1;

    GlobalTensor<inputT> mGmX_;
    GlobalTensor<inputT> mGmAddNum_;
    GlobalTensor<float> mGmY_;
    GlobalTensor<int32_t> mGmIndices_;
    GlobalTensor<float> mGmWorkspace_;
    GlobalTensor<uint32_t> mGmAssist_;

    GlobalTensor<int32_t> mappingNumGm_;
    GlobalTensor<int32_t> mappingTableGm_;
};

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void FusedAddTopkDiv<inputT, calT, enableExpertMapping>::InitTilingData(
    AtbOps::FusedAddTopkDivTilingData &__restrict tilingData, GM_ADDR x, GM_ADDR addNum,
    GM_ADDR mappingNum, GM_ADDR mappingTable, GM_ADDR y, GM_ADDR indices,
    GM_ADDR workspace)
{
    secondDimSize_ = tilingData.secondDimSize;
    groupNum_ = tilingData.groupNum;
    groupTopk_ = tilingData.groupTopk;
    n_ = tilingData.n;
    k_ = tilingData.k;
    activateType_ = tilingData.activateType;
    isNorm_ = tilingData.isNorm;
    scale_ = tilingData.scale;
    groupEles_ = tilingData.groupEles;
    expertNum_ = tilingData.expertNum;
    tableDim_ = tilingData.tableDim;

    uint32_t batchPerCore = tilingData.batchPerCore;
    uint32_t tailBatch = tilingData.tailBatch;
    uint32_t blockIdx = GetBlockIdx();
    uint64_t workspacePerCore = tilingData.workspacePerCore / sizeof(float);
    uint32_t perBlockCountFp32 = BLOCK_BYTES / sizeof(float);
    if (blockIdx < tailBatch)
    {
        loopBatch_ = batchPerCore + 1;
        batchOffset_ = blockIdx * loopBatch_;
    }
    else
    {
        loopBatch_ = batchPerCore;
        batchOffset_ = blockIdx * batchPerCore + tailBatch;
    }
    outBatchStride_ = k_ * batchOffset_;
    mGmX_.SetGlobalBuffer((__gm__ inputT *)x);
    mGmAddNum_.SetGlobalBuffer((__gm__ inputT *)addNum);
    mGmY_.SetGlobalBuffer((__gm__ float *)y);
    mGmIndices_.SetGlobalBuffer((__gm__ int32_t *)indices);
    mGmWorkspace_.SetGlobalBuffer((__gm__ float *)workspace);
    mGmAssist_.SetGlobalBuffer((__gm__ uint32_t *)assistGm);
    mappingNumGm_.SetGlobalBuffer((__gm__ int32_t *)mappingNum);
    mappingTableGm_.SetGlobalBuffer((__gm__ int32_t *)mappingTable);

    groupElesAlignBlockCountFp32_ = CeilAlign(groupEles_, perBlockCountFp32);
    groupElesAlignSortCount_ = CeilAlign(groupEles_, SORT_UNIT);
    secondAlignBlockCountFp32_ = CeilAlign(secondDimSize_, perBlockCountFp32);
    wsOffset_ = blockIdx * workspacePerCore;
    sortRepeatTimes_ = CeilDiv(secondDimSize_, SORT_UNIT);
    wholeSortNum_ = sortRepeatTimes_ * SORT_UNIT;
}

// init used buffer
template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void FusedAddTopkDiv<inputT, calT, enableExpertMapping>::InitBuffer(TPipe *inputPipe)
{
    pipe_ = inputPipe;
    uint32_t baseCountGroupNum = BASE_COUNT * groupNum_;
    pipe_->InitBuffer(xInQueue_, BUFFER_NUM, sizeof(float) * baseCountGroupNum);
    pipe_->InitBuffer(addNumInQueue_, sizeof(float) * BASE_COUNT);
    pipe_->InitBuffer(yOutQueue_, BUFFER_NUM, sizeof(float) * BASE_COUNT);
    pipe_->InitBuffer(indicesOutQueue_, BUFFER_NUM, sizeof(int32_t) * BASE_COUNT);
    pipe_->InitBuffer(sigmoidBuf_, sizeof(float) * BASE_COUNT);
    pipe_->InitBuffer(sigmoidAddQueue_, BUFFER_NUM_ONE, sizeof(float) * BASE_COUNT);
    pipe_->InitBuffer(tempBuf_, sizeof(float) * baseCountGroupNum);
    pipe_->InitBuffer(sortedQueue_, BUFFER_NUM_ONE, sizeof(int64_t) * baseCountGroupNum);
    pipe_->InitBuffer(topkValueQueue_, BUFFER_NUM_ONE, sizeof(float) * baseCountGroupNum);
    pipe_->InitBuffer(assistQueue_, BUFFER_NUM_ONE, sizeof(uint32_t) * BASE_COUNT);
    // new mapping
    pipe_->InitBuffer(mappingNumQueue_, BUFFER_NUM_ONE, sizeof(int32_t) * expertNum_);
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void FusedAddTopkDiv<inputT, calT, enableExpertMapping>::CopyInAddNum()
{
    LocalTensor<float> addNumLocal = addNumInQueue_.Get<float>();
    uint32_t secondDimSizeInputBytes = secondDimSize_ * sizeof(inputT);
    if constexpr (IsSameType<inputT, float>::value)
    {
        DataCopyPad(addNumLocal, mGmAddNum_, {1, secondDimSizeInputBytes, 0, 0, 0}, {false, 0, 0, 0});
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    }
    else
    {
        DataCopyPad(addNumLocal[secondAlignBlockCountFp32_].template ReinterpretCast<inputT>(), mGmAddNum_,
                    {1, secondDimSizeInputBytes, 0, 0, 0}, {false, 0, 0, 0});
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast(addNumLocal, addNumLocal[secondAlignBlockCountFp32_].template ReinterpretCast<inputT>(),
             RoundMode::CAST_NONE, secondDimSize_);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void FusedAddTopkDiv<inputT, calT, enableExpertMapping>::CopyInX(const int32_t loop)
{
    LocalTensor<float> xLocal = xInQueue_.AllocTensor<float>();
    int64_t xOffset = loop * secondDimSize_ + batchOffset_ * static_cast<int64_t>(secondDimSize_);
    uint32_t secondDimSizeInputBytes = secondDimSize_ * sizeof(inputT);
    if constexpr (IsSameType<inputT, float>::value)
    {
        DataCopyPad(xLocal, mGmX_[xOffset], {1, secondDimSizeInputBytes, 0, 0, 0}, {false, 0, 0, 0});
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    }
    else
    {
        DataCopyPad(xLocal[secondAlignBlockCountFp32_].template ReinterpretCast<inputT>(), mGmX_[xOffset],
                    {1, secondDimSizeInputBytes, 0, 0, 0}, {false, 0, 0, 0});
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Cast(xLocal, xLocal[secondAlignBlockCountFp32_].template ReinterpretCast<inputT>(),
             RoundMode::CAST_NONE, secondDimSize_);
        AscendC::PipeBarrier<PIPE_V>();
    }
    xInQueue_.EnQue(xLocal);
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void FusedAddTopkDiv<inputT, calT, enableExpertMapping>::ActivateAndAdd()
{
    LocalTensor<float> xLocal = xInQueue_.DeQue<float>();
    LocalTensor<float> addNumLocal = addNumInQueue_.Get<float>();
    LocalTensor<uint8_t> sharedTmpBuffer = tempBuf_.Get<uint8_t>();
    LocalTensor<float> sigmoidTensor = sigmoidBuf_.Get<float>();
    LocalTensor<float> sigmoidAddTensor = sigmoidAddQueue_.AllocTensor<float>();

    Sigmoid(sigmoidTensor, xLocal, sharedTmpBuffer, secondDimSize_);
    AscendC::PipeBarrier<PIPE_V>();
    Add(sigmoidAddTensor, sigmoidTensor, addNumLocal, secondDimSize_);
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    xInQueue_.EnQue(xLocal);
    sigmoidAddQueue_.EnQue(sigmoidAddTensor);
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void FusedAddTopkDiv<inputT, calT, enableExpertMapping>::CopyToWorkspace()
{
    LocalTensor<float> sigmoidAddTensor = sigmoidAddQueue_.DeQue<float>();
    DataCopyPad(mGmWorkspace_[wsOffset_], sigmoidAddTensor, {1, (uint32_t)(secondDimSize_ * sizeof(float)), 0, 0, 0});
    sigmoidAddQueue_.EnQue(sigmoidAddTensor);
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void FusedAddTopkDiv<inputT, calT, enableExpertMapping>::CopyFromWorkspace()
{
    LocalTensor<float> xLocal = xInQueue_.DeQue<float>();
    LocalTensor<uint32_t> assistLocal = assistQueue_.AllocTensor<uint32_t>();
    DataCopyExtParams xWorkspaceGroupCopyParams{(uint16_t)1, (uint32_t)(groupEles_ * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> xWorkspaceGroupPadParams{true, 0, (uint8_t)(groupElesAlignBlockCountFp32_ - groupEles_),
                                                          (float)FLOAT32_NEG_INF};
    ArithProgression(assistLocal.template ReinterpretCast<int32_t>(), 0, 1, ADD_COUNT_THIRTY_TWO);
    AscendC::PipeBarrier<PIPE_V>();
    event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
    Duplicate<float>(xLocal, FLOAT32_NEG_INF, groupElesAlignSortCount_ * groupNum_);
    Adds(assistLocal[ADD_COUNT_THIRTY_TWO].template ReinterpretCast<int32_t>(),
         assistLocal.template ReinterpretCast<int32_t>(), int32_t(0), ADD_COUNT_THIRTY_TWO);
    event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    for (size_t i = 0; i < groupNum_; i++)
    {
        DataCopyPad(xLocal[groupElesAlignSortCount_ * i], mGmWorkspace_[wsOffset_ + groupEles_ * i],
                    xWorkspaceGroupCopyParams, xWorkspaceGroupPadParams);
    }
    AscendC::PipeBarrier<PIPE_V>();
    Adds(assistLocal[ADD_COUNT_SIXTY_FOUR].template ReinterpretCast<int32_t>(),
         assistLocal.template ReinterpretCast<int32_t>(), int32_t(0), ADD_COUNT_SIXTY_FOUR);
    AscendC::PipeBarrier<PIPE_V>();
    Adds(assistLocal[ADD_COUNT_ONE_TWENTY_EIGHT].template ReinterpretCast<int32_t>(),
         assistLocal.template ReinterpretCast<int32_t>(), int32_t(0), ADD_COUNT_ONE_TWENTY_EIGHT);
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);

    xInQueue_.EnQue(xLocal);
    assistQueue_.EnQue(assistLocal);
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void FusedAddTopkDiv<inputT, calT, enableExpertMapping>::GroupReduceSumInternelImpl()
{
    LocalTensor<float> xLocal = xInQueue_.DeQue<float>();
    LocalTensor<uint32_t> assistLocal = assistQueue_.DeQue<uint32_t>();
    LocalTensor<float> sortedTensor = sortedQueue_.AllocTensor<float>();
    LocalTensor<float> topkGroupValue = topkValueQueue_.AllocTensor<float>();
    LocalTensor<float> tempTensor = tempBuf_.Get<float>();

    Duplicate(topkGroupValue, FLOAT32_NEG_INF, wholeSortNum_);
    AscendC::PipeBarrier<PIPE_V>();
    Sort32<calT>(sortedTensor, xLocal, assistLocal, groupNum_);
    AscendC::PipeBarrier<PIPE_V>();

    uint64_t rsvdCnt = 0;
    GatherMaskParams gatherMaskParams{1, uint8_t(groupNum_), REPEAT_STRIDE_EIGHT, REPEAT_STRIDE_EIGHT};
    GatherMask(tempTensor, sortedTensor, 1, false, 0, gatherMaskParams, rsvdCnt);
    AscendC::PipeBarrier<PIPE_V>();

    for (size_t i = 0; i < groupNum_; i++)
    {
        ReduceSum<calT>(xLocal[i * SORT_UNIT], tempTensor[i * SORT_UNIT], sortedTensor, n_);
    }
    AscendC::PipeBarrier<PIPE_V>();

    auto gatherOffset = sortedTensor.template ReinterpretCast<int32_t>();
    Muls(gatherOffset, assistLocal.template ReinterpretCast<int32_t>(), int32_t(sizeof(float) * SORT_UNIT), groupNum_);
    AscendC::PipeBarrier<PIPE_V>();
    Gather(topkGroupValue, xLocal, gatherOffset.template ReinterpretCast<uint32_t>(), 0, groupNum_);

    xInQueue_.FreeTensor(xLocal);
    assistQueue_.EnQue(assistLocal);
    sortedQueue_.FreeTensor(sortedTensor);
    topkValueQueue_.EnQue(topkGroupValue);
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void FusedAddTopkDiv<inputT, calT, enableExpertMapping>::GroupTopkImpl()
{
    LocalTensor<float> xLocal = xInQueue_.AllocTensor<float>();
    LocalTensor<uint32_t> assistLocal = assistQueue_.DeQue<uint32_t>();
    LocalTensor<float> sortedTensor = sortedQueue_.AllocTensor<float>();
    LocalTensor<float> topkGroupValue = topkValueQueue_.DeQue<float>();
    LocalTensor<float> sigmoidAddTensor = sigmoidAddQueue_.DeQue<float>();

    Sort32<calT>(sortedTensor, topkGroupValue, assistLocal, 1);
    AscendC::PipeBarrier<PIPE_V>();

    auto dstOffset = sortedTensor.template ReinterpretCast<uint32_t>();
    Duplicate(topkGroupValue, float(0), groupNum_);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    for (size_t i = 0; i < groupTopk_; i++)
    {
        int32_t selectedGroupIndex = dstOffset.GetValue(SORTED_COEF * i + 1);
        topkGroupValue.SetValue(selectedGroupIndex, float(1));
    }
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    uint32_t dstShape[BROADCAST_DIM] = {(uint32_t)groupNum_, (uint32_t)groupEles_};
    uint32_t srcShape[BROADCAST_DIM] = {(uint32_t)groupNum_, 1};
    LocalTensor<uint8_t> sharedTmpBuffer = tempBuf_.Get<uint8_t>();
    AscendC::PipeBarrier<PIPE_V>();
    BroadCast<float, BROADCAST_DIM, BROADCAST_AXIS>(sortedTensor, topkGroupValue, dstShape, srcShape, sharedTmpBuffer);
    pipe_barrier(PIPE_V);
    Mul(topkGroupValue, sigmoidAddTensor, sortedTensor, secondDimSize_);

    xInQueue_.FreeTensor(xLocal);
    assistQueue_.EnQue(assistLocal);
    sortedQueue_.FreeTensor(sortedTensor);
    topkValueQueue_.EnQue(topkGroupValue);
    sigmoidAddQueue_.FreeTensor(sigmoidAddTensor);
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void FusedAddTopkDiv<inputT, calT, enableExpertMapping>::GatherSigmoidImpl()
{
    LocalTensor<int32_t> indicesLocal = indicesOutQueue_.AllocTensor<int32_t>();
    LocalTensor<float> sigmoidTensor = sigmoidBuf_.Get<float>();
    LocalTensor<uint32_t> assistLocal = assistQueue_.DeQue<uint32_t>();
    LocalTensor<float> sortedTensor = sortedQueue_.AllocTensor<float>();
    LocalTensor<float> groupTopkValue = topkValueQueue_.DeQue<float>();
    LocalTensor<float> yLocal = yOutQueue_.AllocTensor<float>();
    LocalTensor<float> tempTensor = tempBuf_.Get<float>();

    ArithProgression(assistLocal.template ReinterpretCast<int32_t>(), 0, 1, wholeSortNum_);
    AscendC::PipeBarrier<PIPE_V>();

    Sort<float, true>(sortedTensor, groupTopkValue, assistLocal, tempTensor, sortRepeatTimes_);
    AscendC::PipeBarrier<PIPE_V>();

    Extract(tempTensor, indicesLocal.template ReinterpretCast<uint32_t>(), sortedTensor, sortRepeatTimes_);
    AscendC::PipeBarrier<PIPE_V>();

    Muls(assistLocal.template ReinterpretCast<int32_t>(), indicesLocal, FLOAT_BYTES, secondDimSize_);
    AscendC::PipeBarrier<PIPE_V>();
    Gather(yLocal, sigmoidTensor, assistLocal, 0, secondDimSize_);
    AscendC::PipeBarrier<PIPE_V>();

    indicesOutQueue_.EnQue(indicesLocal);
    assistQueue_.FreeTensor(assistLocal);
    sortedQueue_.FreeTensor(sortedTensor);
    topkValueQueue_.FreeTensor(groupTopkValue);
    yOutQueue_.EnQue(yLocal);
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void FusedAddTopkDiv<inputT, calT, enableExpertMapping>::NormImpl()
{
    LocalTensor<float> yLocal = yOutQueue_.DeQue<float>();
    LocalTensor<float> tempTensor = tempBuf_.Get<float>();
    LocalTensor<float> reduceSumValueTensor = sigmoidBuf_.Get<float>();

    ReduceSum<calT>(tempTensor, yLocal, tempTensor, k_);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    float reduceSumValue = tempTensor.GetValue(0);
    AscendC::PipeBarrier<PIPE_V>();
    Duplicate<float>(reduceSumValueTensor, reduceSumValue, k_);
    AscendC::PipeBarrier<PIPE_V>();
    Div(yLocal, yLocal, reduceSumValueTensor, k_);
    AscendC::PipeBarrier<PIPE_V>();
    Muls(yLocal, yLocal, scale_, k_);

    yOutQueue_.EnQue(yLocal);
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void FusedAddTopkDiv<inputT, calT, enableExpertMapping>::CopyInMappingNum()
{
    LocalTensor<int32_t> mappingNumLocal = mappingNumQueue_.AllocTensor<int32_t>();
    DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(expertNum_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(mappingNumLocal, mappingNumGm_, copyParams, padParams);
    mappingNumQueue_.EnQue<int32_t>(mappingNumLocal);
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void FusedAddTopkDiv<inputT, calT, enableExpertMapping>::CopyOut(const int32_t loop)
{
    int64_t offset = outBatchStride_ + loop * k_;
    LocalTensor<float> yLocal = yOutQueue_.DeQue<float>();
    LocalTensor<int32_t> indicesLocal = indicesOutQueue_.DeQue<int32_t>();
    
    if constexpr (enableExpertMapping) {
        // CopyIn mappingNum
        CopyInMappingNum();
        
        // Compute
        LocalTensor<int32_t> mappingNumLocal = mappingNumQueue_.DeQue<int32_t>();

        // Update Indices
        event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

        for (size_t kI = 0; kI < k_; kI++) {
            uint32_t expertId = indicesLocal.GetValue(kI);
            uint32_t expertMappingNum = mappingNumLocal.GetValue(expertId);
            uint32_t redundantOffset = expertMappingNum == 0 ? 0 : (batchOffset_ + loop) % expertMappingNum;
            uint32_t tableOffset = expertId * tableDim_ + redundantOffset;
            indicesLocal.SetValue(kI, mappingTableGm_[tableOffset].GetValue(0));
        }
        mappingNumQueue_.FreeTensor<int32_t>(mappingNumLocal);
        event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    }

    DataCopyExtParams copyParams{1, (uint32_t)(k_ * sizeof(float)), 0, 0, 0};
    DataCopyPad(mGmY_[offset], yLocal, copyParams);
    DataCopyPad(mGmIndices_[offset], indicesLocal, copyParams);
    yOutQueue_.FreeTensor(yLocal);
    indicesOutQueue_.FreeTensor(indicesLocal);
}

template <typename inputT, typename calT, uint32_t enableExpertMapping>
__aicore__ inline void FusedAddTopkDiv<inputT, calT, enableExpertMapping>::Process()
{
    CopyInAddNum();
    for (size_t loop = 0; loop < loopBatch_; loop++)
    {
        CopyInX(loop);
        ActivateAndAdd();
        CopyToWorkspace();
        CopyFromWorkspace();
        GroupReduceSumInternelImpl();
        AscendC::PipeBarrier<PIPE_V>();
        GroupTopkImpl();
        GatherSigmoidImpl();
 
        if (isNorm_ == 1)
        {
            NormImpl();
        }
        CopyOut(loop);
    }
}
#endif // FUSED_ADD_TOPK_DIV_H_
