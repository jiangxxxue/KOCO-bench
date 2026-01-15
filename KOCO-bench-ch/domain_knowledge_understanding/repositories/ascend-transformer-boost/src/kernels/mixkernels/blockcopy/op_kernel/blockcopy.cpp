/*
* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "kernel_operator.h"
#include "mixkernels/utils/common/kernel/kernel_utils.h"
#include "mixkernels/blockcopy/tiling/tiling_data.h"

using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_LENGTH = 128;
constexpr uint32_t OUT_UB_SIZE = 1024 * 45;
constexpr int32_t BITS_PER_BYTE = 8;
constexpr int32_t TRANS_RATIO = 32 / sizeof(int32_t);
constexpr uint32_t INT32_SIZE = sizeof(int32_t);

template <typename T>
class BlockCopy {
public:
    __aicore__ inline BlockCopy() {}

    __aicore__ inline void Init(GM_ADDR kCache, GM_ADDR vCache,
                                GM_ADDR srcBlockIndices, GM_ADDR dstBlockIndices,
                                GM_ADDR cumSum, AtbOps::BlockCopyTilingData *tiling_data, TPipe* pipeIn)
    {
        InitParams(tiling_data);
        kCacheGm.SetGlobalBuffer((__gm__ T *)kCache, blockCount_ * blockSizeinElement_);
        vCacheGm.SetGlobalBuffer((__gm__ T *)vCache, blockCount_ * blockSizeinElement_);
        srcBlockIndicesGm.SetGlobalBuffer((__gm__ int32_t *)srcBlockIndices, sourceCount_);
        dstBlockIndicesGm.SetGlobalBuffer((__gm__ int32_t *)dstBlockIndices, destinationCount_);
        cumSumGm.SetGlobalBuffer((__gm__ int32_t *)cumSum, sourceCount_);

        pipe = pipeIn;
        pipe->InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * INT32_SIZE);
        pipe->InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * INT32_SIZE);
        pipe->InitBuffer(inQueueA, BUFFER_NUM, TILE_LENGTH * INT32_SIZE);
        pipe->InitBuffer(onesBuf, TILE_LENGTH * INT32_SIZE);
        pipe->InitBuffer(floatBuf, TILE_LENGTH * INT32_SIZE);
        pipe->InitBuffer(positionBuf, TILE_LENGTH * INT32_SIZE);
        pipe->InitBuffer(maskBuf, TILE_LENGTH / BITS_PER_BYTE);
        pipe->InitBuffer(src2dstQueueK, BUFFER_NUM, OUT_UB_SIZE);
        pipe->InitBuffer(src2dstQueueV, BUFFER_NUM, OUT_UB_SIZE);
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = (sourceCount_ + TILE_LENGTH - 1) / TILE_LENGTH;
        int32_t tailLength = sourceCount_ % TILE_LENGTH;
        if (tailLength == 0) {
            tailLength = TILE_LENGTH;
        }
        for (int32_t i = 0; i < loopCount - 1; i++) {
            if (cumSumOffset_ >= 0) {
                break;
            }
            SearchCopyIn(i, TILE_LENGTH);
            SearchCompute(i, TILE_LENGTH);
        }
        if (cumSumOffset_ < 0) {
            SearchCopyIn(loopCount - 1, tailLength);
            SearchCompute(loopCount - 1, tailLength);
        }

        loopCount = (perCoreCopyCount_ + TILE_LENGTH - 1) / TILE_LENGTH;
        tailLength = perCoreCopyCount_ % TILE_LENGTH;
        if (tailLength == 0) {
            tailLength = TILE_LENGTH;
        }
        for (int32_t i = 0; i < loopCount - 1; i++) {
            CopyIndicesIn(i, TILE_LENGTH);
            CopyBlocks(i, TILE_LENGTH);
        }
        CopyIndicesIn(loopCount - 1, tailLength);
        CopyBlocks(loopCount - 1, tailLength);
    }

private:
    __aicore__ inline void InitParams(AtbOps::BlockCopyTilingData *tiling_data)
    {
        blockIdx_ = GetBlockIdx();
        if (blockIdx_ == 0) {
            cumSumOffset_ = 0;
        }
        perCoreCopyCount_ = tiling_data->perCoreCopyCount;
        tailCoreCopyCount_ = tiling_data->tailCoreCopyCount;
        if (blockIdx_ < tailCoreCopyCount_) {
            perCoreCopyCount_++;
            gmOffset_ = perCoreCopyCount_ * blockIdx_;
        } else {
            gmOffset_ = perCoreCopyCount_ * blockIdx_ + tailCoreCopyCount_;
        }
        blockCount_ = tiling_data->blockCount;
        blockSizeinElement_ = tiling_data->blockSize * tiling_data->numHead * tiling_data->headSizeK;
        sourceCount_ = tiling_data->sourceCount;
        destinationCount_ = tiling_data->destinationCount;
        typeByte_ = tiling_data->typeByte;
        blockDim_ = tiling_data->blockDim;
        outUbLength_ = OUT_UB_SIZE / typeByte_;
        cacheCopyLoopCount_ = (blockSizeinElement_ + outUbLength_ - 1) / outUbLength_;
        cacheCopyTailLength_ = blockSizeinElement_ % outUbLength_;
        if (cacheCopyTailLength_ == 0) {
            cacheCopyTailLength_ = outUbLength_;
        }
    }

    __aicore__ inline void SearchCopyIn(int32_t progress, int32_t processLength)
    {
        LocalTensor<int32_t> cumSumLocal = inQueueX.AllocTensor<int32_t>();
        int32_t processDataSizeResidual = processLength % TRANS_RATIO;
        if (processDataSizeResidual == 0) {
            DataCopy(cumSumLocal, cumSumGm[progress * TILE_LENGTH], processLength);
        } else {
            int32_t paddingLength = TRANS_RATIO - processDataSizeResidual;
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(processLength * INT32_SIZE), 0, 0, 0};
            DataCopyPadExtParams<int32_t> padParams{true, 0, static_cast<uint8_t>(paddingLength), INT32_MAX};
            DataCopyPad(cumSumLocal, cumSumGm[progress * TILE_LENGTH], copyParams, padParams);
        }
        inQueueX.EnQue(cumSumLocal);
    }

    __aicore__ inline void SearchCompute(int32_t progress, int32_t processLength)
    {
        LocalTensor<int32_t> cumSumLocal = inQueueX.DeQue<int32_t>();
        LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
        LocalTensor<float> onesLocal = onesBuf.Get<float>();
        LocalTensor<float> positionLocal = positionBuf.Get<float>();
        int32_t processLengthPad = (processLength + TRANS_RATIO - 1) / TRANS_RATIO * TRANS_RATIO;
        event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
        Duplicate(cumSumLocal[processLengthPad], INT32_MAX, TILE_LENGTH - processLengthPad);
        Duplicate(onesLocal, static_cast<float>(1), TILE_LENGTH);
        int32_t endCum = cumSumLocal.GetValue(processLength - 1);
        if (gmOffset_ < endCum) {
            LocalTensor<float> cumSumLocalFloat = floatBuf.Get<float>();
            Cast(cumSumLocalFloat, cumSumLocal, RoundMode::CAST_FLOOR, TILE_LENGTH);
            PipeBarrier<PIPE_V>();
            CompareScalar(maskLocal, cumSumLocalFloat, static_cast<float>(gmOffset_), CMPMODE::LE, TILE_LENGTH);
            PipeBarrier<PIPE_V>();
            Select(positionLocal, maskLocal, onesLocal, static_cast<float>(0),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, TILE_LENGTH);
            PipeBarrier<PIPE_V>();
            ReduceSum(positionLocal, positionLocal, onesLocal, TILE_LENGTH);
            event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIDVToS);
            WaitFlag<HardEvent::V_S>(eventIDVToS);
            cumSumOffset_ = progress * TILE_LENGTH + positionLocal.GetValue(0);
        }
        inQueueX.FreeTensor(cumSumLocal);
    }

    __aicore__ inline void CopyIndicesIn(int32_t progress, int32_t processLength)
    {
        LocalTensor<int32_t> dstBlockIndicesLocal = inQueueY.AllocTensor<int32_t>();
        LocalTensor<int32_t> srcBlockIndicesLocal = inQueueX.AllocTensor<int32_t>();
        LocalTensor<int32_t> cumSumLocal = inQueueA.AllocTensor<int32_t>();
        int32_t cumSumProcessLength = ((sourceCount_ - cumSumOffset_) < processLength ?
                                       (sourceCount_ - cumSumOffset_) : processLength);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(processLength * INT32_SIZE), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{false, 0, static_cast<uint8_t>(0), 0};
        DataCopyExtParams copyParams1{1, static_cast<uint32_t>(cumSumProcessLength * INT32_SIZE), 0, 0, 0};
        DataCopyPad(cumSumLocal, cumSumGm[cumSumOffset_], copyParams1, padParams);
        DataCopyPad(srcBlockIndicesLocal, srcBlockIndicesGm[cumSumOffset_], copyParams1, padParams);
        DataCopyPad(dstBlockIndicesLocal, dstBlockIndicesGm[gmOffset_ + progress * TILE_LENGTH], copyParams, padParams);
        inQueueY.EnQue(dstBlockIndicesLocal);
        inQueueX.EnQue(srcBlockIndicesLocal);
        inQueueA.EnQue(cumSumLocal);
    }

    __aicore__ inline void CopyBlocks(int32_t progress, int32_t processLength)
    {
        LocalTensor<int32_t> dstBlockIndicesLocal = inQueueY.DeQue<int32_t>();
        LocalTensor<int32_t> srcBlockIndicesLocal = inQueueX.DeQue<int32_t>();
        LocalTensor<int32_t> cumSumLocal = inQueueA.DeQue<int32_t>();
        event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
        int32_t processIndex = 0;
        int32_t completedCount = 0;
        int64_t previousCum = gmOffset_ + progress * TILE_LENGTH;
        WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
        while (completedCount < processLength) {
            int32_t srcBlockIndex = srcBlockIndicesLocal.GetValue(processIndex);
            int64_t currentCum = static_cast<int64_t>(cumSumLocal.GetValue(processIndex));
            // 在当前核处理范围内，源block地址srcBlockIndex对应的目标block地址数量
            int32_t paddingCount = (currentCum - previousCum) > (processLength - completedCount) ?
                                   (processLength - completedCount) : (currentCum - previousCum);
            if (currentCum - previousCum <= processLength - completedCount) {
                // 在当前核处理范围内，源block地址srcBlockIndex对应的目标block地址已处理完，考虑下一个源block地址
                processIndex++;
            }
            previousCum = currentCum;
            CopyOneSrc2MultiDst(srcBlockIndex, completedCount, paddingCount, dstBlockIndicesLocal);
            completedCount += paddingCount;
        }
        cumSumOffset_ += processIndex;
        inQueueA.FreeTensor(cumSumLocal);
        inQueueX.FreeTensor(srcBlockIndicesLocal);
        inQueueY.FreeTensor(dstBlockIndicesLocal);
    }

    __aicore__ inline void CopyOneSrc2MultiDst(int32_t srcBlockIndex, int32_t completedCount, int32_t paddingCount,
                                               LocalTensor<int32_t>& dstIndices)
    {
        for (int i = 0; i < paddingCount; i++) {
            int32_t dstBlockIndex = dstIndices.GetValue(completedCount + i);
            event_t eventIDSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
            SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
            WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
            for (int32_t j = 0; j < cacheCopyLoopCount_ - 1; j++) {
                CopyOneBlockIn(srcBlockIndex, j, outUbLength_);
                CopyOneBlockOut(dstBlockIndex, j, outUbLength_);
            }
            CopyOneBlockIn(srcBlockIndex, cacheCopyLoopCount_ - 1, cacheCopyTailLength_);
            CopyOneBlockOut(dstBlockIndex, cacheCopyLoopCount_ - 1, cacheCopyTailLength_);
        }
    }

    __aicore__ inline void CopyOneBlockIn(int32_t srcBlockIndex, int32_t progress, int32_t processLength)
    {
        LocalTensor<T> src2dstLocalK = src2dstQueueK.AllocTensor<T>();
        LocalTensor<T> src2dstLocalV = src2dstQueueV.AllocTensor<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(processLength * typeByte_), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, static_cast<uint8_t>(0), 0};
        DataCopyPad(src2dstLocalK, kCacheGm[srcBlockIndex * blockSizeinElement_ + progress * outUbLength_],
                    copyParams, padParams);
        DataCopyPad(src2dstLocalV, vCacheGm[srcBlockIndex * blockSizeinElement_ + progress * outUbLength_],
                    copyParams, padParams);
        src2dstQueueK.EnQue(src2dstLocalK);
        src2dstQueueV.EnQue(src2dstLocalV);
    }

    __aicore__ inline void CopyOneBlockOut(int32_t dstBlockIndex, int32_t progress, int32_t processLength)
    {
        LocalTensor<T> src2dstLocalK = src2dstQueueK.DeQue<T>();
        LocalTensor<T> src2dstLocalV = src2dstQueueV.DeQue<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(processLength * typeByte_), 0, 0, 0};
        DataCopyPad(kCacheGm[dstBlockIndex * blockSizeinElement_ + progress * outUbLength_],
                    src2dstLocalK, copyParams);
        DataCopyPad(vCacheGm[dstBlockIndex * blockSizeinElement_ + progress * outUbLength_],
                    src2dstLocalV, copyParams);
        src2dstQueueK.FreeTensor(src2dstLocalK);
        src2dstQueueV.FreeTensor(src2dstLocalV);
    }

private:
    TPipe* pipe;

    GlobalTensor<T> kCacheGm;
    GlobalTensor<T> vCacheGm;
    GlobalTensor<int32_t> srcBlockIndicesGm;
    GlobalTensor<int32_t> dstBlockIndicesGm;
    GlobalTensor<int32_t> cumSumGm;
    GlobalTensor<int32_t> srcWorkSpaceGm;
    GlobalTensor<int32_t> syncGm;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueA;
    TBuf<TPosition::VECCALC> maskBuf;
    TBuf<TPosition::VECCALC> onesBuf;
    TBuf<TPosition::VECCALC> positionBuf;
    TBuf<TPosition::VECCALC> floatBuf;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> src2dstQueueK;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> src2dstQueueV;

    uint32_t perCoreCopyCount_ = 0;
    uint32_t tailCoreCopyCount_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t blockCount_;
    uint32_t sourceCount_;
    uint32_t destinationCount_;
    uint32_t typeByte_;
    uint32_t blockDim_;
    uint32_t outUbLength_ = 0;
    uint32_t cacheCopyLoopCount_ = 0;
    uint32_t cacheCopyTailLength_ = 0;
    uint64_t blockSizeinElement_;
    int64_t gmOffset_ = 0;
    int64_t cumSumOffset_ = -1;
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AtbOps::BlockCopyTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->blockCount = (*(const __gm__ uint32_t *)(p_tilingdata + 0));
    tilingdata->blockSize = (*(const __gm__ uint32_t *)(p_tilingdata + 4));
    tilingdata->numHead = (*(const __gm__ uint32_t *)(p_tilingdata + 8));
    tilingdata->headSizeK = (*(const __gm__ uint32_t *)(p_tilingdata + 12));
    tilingdata->headSizeV = (*(const __gm__ uint32_t *)(p_tilingdata + 16));
    tilingdata->sourceCount = (*(const __gm__ uint32_t *)(p_tilingdata + 20));
    tilingdata->destinationCount = (*(const __gm__ uint32_t *)(p_tilingdata + 24));
    tilingdata->typeByte = (*(const __gm__ uint32_t *)(p_tilingdata + 28));
    tilingdata->blockDim = (*(const __gm__ uint32_t *)(p_tilingdata + 32));
    tilingdata->perCoreCopyCount = (*(const __gm__ uint32_t *)(p_tilingdata + 36));
    tilingdata->tailCoreCopyCount = (*(const __gm__ uint32_t *)(p_tilingdata + 40));
#endif
}

extern "C" __global__ __aicore__ void blockcopy(GM_ADDR kCache, GM_ADDR vCache,
                                             GM_ADDR srcBlockIndices, GM_ADDR dstBlockIndices, GM_ADDR cumSum,
                                             GM_ADDR kCacheOut, GM_ADDR vCacheOut, GM_ADDR tiling)
{
    AtbOps::BlockCopyTilingData tilingData;
    InitTilingData(tiling, &(tilingData));
    TPipe pipe;
    if (TILING_KEY_IS(100000000)) {
        BlockCopy<int8_t> op;
        op.Init(kCache, vCache, srcBlockIndices, dstBlockIndices, cumSum, &tilingData, &pipe);
        op.Process();
    }
    if (TILING_KEY_IS(200000000)) {
        BlockCopy<half> op;
        op.Init(kCache, vCache, srcBlockIndices, dstBlockIndices, cumSum, &tilingData, &pipe);
        op.Process();
    }
}
