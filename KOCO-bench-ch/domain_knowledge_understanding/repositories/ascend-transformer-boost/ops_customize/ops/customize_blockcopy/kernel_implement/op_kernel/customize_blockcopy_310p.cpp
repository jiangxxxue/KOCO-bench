/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "kernel_utils.h"
#include "../tiling/tiling_data.h"

using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_LENGTH = 128;
constexpr uint32_t OUT_UB_SIZE = 1024 * 45;
constexpr int32_t BITS_PER_BYTE = 8;
constexpr int32_t TRANS_RATIO = 32 / sizeof(int32_t);
constexpr uint32_t INT32_SIZE = sizeof(int32_t);

template <typename T> class CustomizeBlockCopy310P {
public:
    __aicore__ inline CustomizeBlockCopy310P() {}

    __aicore__ inline void Init(GM_ADDR kCache, GM_ADDR vCache, GM_ADDR srcBlockIndices, GM_ADDR dstBlockIndices,
                                GM_ADDR cumSum, AtbOps::CustomizeBlockCopyTilingData *tiling_data, TPipe *pipeIn)
    {
        InitParams(tiling_data);
        kCacheGm.SetGlobalBuffer((__gm__ T *)kCache, blockCount_ * blockSizeofElement_);
        vCacheGm.SetGlobalBuffer((__gm__ T *)vCache, blockCount_ * blockSizeofElement_);
        srcBlockIndicesGm.SetGlobalBuffer((__gm__ int32_t *)srcBlockIndices, sourceCount_);
        dstBlockIndicesGm.SetGlobalBuffer((__gm__ int32_t *)dstBlockIndices, destinationCount_);
        cumSumGm.SetGlobalBuffer((__gm__ int32_t *)cumSum, sourceCount_);

        pipe = pipeIn;
        pipe->InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * INT32_SIZE);
        pipe->InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * INT32_SIZE);
        pipe->InitBuffer(inQueueA, BUFFER_NUM, TILE_LENGTH * INT32_SIZE);
        pipe->InitBuffer(maskBuf, TILE_LENGTH / BITS_PER_BYTE);
        pipe->InitBuffer(src2dstQueueK, BUFFER_NUM, OUT_UB_SIZE);
        pipe->InitBuffer(src2dstQueueV, BUFFER_NUM, OUT_UB_SIZE);
        pipe->InitBuffer(onesBuf, TILE_LENGTH * INT32_SIZE);
        pipe->InitBuffer(floatBuf, TILE_LENGTH * INT32_SIZE);
        pipe->InitBuffer(positionBuf, TILE_LENGTH * INT32_SIZE);
    }

    __aicore__ inline void Process()
    {
        int32_t totalLoopCount = (sourceCount_ + TILE_LENGTH - 1) / TILE_LENGTH;
        int32_t tailLength = sourceCount_ % TILE_LENGTH;
        if (tailLength == 0) {
            tailLength = TILE_LENGTH;
        }
        for (int32_t i = 0; i < totalLoopCount - 1; i++) {
            if (cumSumOffset_ >= 0) {
                break;
            }
            SearchCopyIn(i, TILE_LENGTH);
            SearchCompute(i, TILE_LENGTH);
        }
        if (cumSumOffset_ < 0) {
            SearchCopyIn(totalLoopCount - 1, tailLength);
            SearchCompute(totalLoopCount - 1, tailLength);
        }

        totalLoopCount = (perCoreCopyCount_ + TILE_LENGTH - 1) / TILE_LENGTH;
        tailLength = perCoreCopyCount_ % TILE_LENGTH;
        if (tailLength == 0) {
            tailLength = TILE_LENGTH;
        }
        for (int32_t i = 0; i < totalLoopCount - 1; i++) {
            // 单次处理128个block
            CopyIndicesIn(i, TILE_LENGTH);
            CopyBlocks(i, TILE_LENGTH);
        }
        CopyIndicesIn(totalLoopCount - 1, tailLength);
        CopyBlocks(totalLoopCount - 1, tailLength);
    }

private:
    __aicore__ inline void InitParams(AtbOps::CustomizeBlockCopyTilingData *tiling_data)
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
        blockSizeofElement_ = tiling_data->blockSize * tiling_data->numHead * tiling_data->headSizeK;
        sourceCount_ = tiling_data->sourceCount;
        destinationCount_ = tiling_data->destinationCount;
        typeByte_ = tiling_data->typeByte;
        blockDim_ = tiling_data->blockDim;
        outUbLength_ = OUT_UB_SIZE / typeByte_;
        cacheCopyLoopCount_ = (blockSizeofElement_ + outUbLength_ - 1) / outUbLength_;
        cacheCopyTailLength_ = blockSizeofElement_ % outUbLength_;
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
            // 向上取整拷贝
            auto processBlocks = (processLength + TRANS_RATIO - 1) / TRANS_RATIO * TRANS_RATIO;
            DataCopy(cumSumLocal, cumSumGm[progress * TILE_LENGTH], processBlocks);
        }
        inQueueX.EnQue(cumSumLocal);
    }

    __aicore__ inline void SearchCompute(int32_t progress, int32_t processLength)
    {
        LocalTensor<int32_t> cumSumLocalTensor = inQueueX.DeQue<int32_t>();
        LocalTensor<uint8_t> maskLocalTensor = maskBuf.Get<uint8_t>();
        LocalTensor<float> onesLocalTensor = onesBuf.Get<float>();
        LocalTensor<float> positionLocalTensor = positionBuf.Get<float>();
        LocalTensor<float> cumSumLocalFloat = floatBuf.Get<float>();
        int32_t processLengthPad = (processLength + TRANS_RATIO - 1) / TRANS_RATIO * TRANS_RATIO;
        event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);

        Duplicate(cumSumLocalTensor[processLengthPad], INT32_MAX, TILE_LENGTH - processLengthPad);
        Duplicate(onesLocalTensor, static_cast<float>(1), TILE_LENGTH);
        Duplicate(cumSumLocalTensor[processLength], INT32_MAX, processLengthPad - processLength);
        int32_t endCum = cumSumLocalTensor.GetValue(processLength - 1);
        if (gmOffset_ < endCum) {
            LocalTensor<float> cumSumLocalFloat = floatBuf.Get<float>();
            Cast(cumSumLocalFloat, cumSumLocalTensor, RoundMode::CAST_NONE, TILE_LENGTH);
            PipeBarrier<PIPE_V>();
            CompareScalar(maskLocalTensor, cumSumLocalFloat, static_cast<float>(gmOffset_), CMPMODE::LE, TILE_LENGTH);
            PipeBarrier<PIPE_V>();
            Select(positionLocalTensor, maskLocalTensor, onesLocalTensor, static_cast<float>(0),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, TILE_LENGTH);
            PipeBarrier<PIPE_V>();
            ReduceSum(positionLocalTensor, positionLocalTensor, onesLocalTensor, TILE_LENGTH);
            event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIDVToS);
            WaitFlag<HardEvent::V_S>(eventIDVToS);
            cumSumOffset_ = progress * TILE_LENGTH + positionLocalTensor.GetValue(0);
        }
        inQueueX.FreeTensor(cumSumLocalTensor);
    }

    __aicore__ inline void CopyIndicesIn(int32_t progress, int32_t processLength)
    {
        LocalTensor<int32_t> dstBlockIdxLocal = inQueueY.AllocTensor<int32_t>();
        LocalTensor<int32_t> srcBlockIdxLocal = inQueueX.AllocTensor<int32_t>();
        LocalTensor<int32_t> cumSumLocal = inQueueA.AllocTensor<int32_t>();
        int32_t cumSumProcessLength =
            ((sourceCount_ - cumSumOffset_) < processLength ? (sourceCount_ - cumSumOffset_) : processLength);

        auto cumSumCopyLength = (cumSumProcessLength + TRANS_RATIO - 1) / TRANS_RATIO * TRANS_RATIO;
        DataCopy(cumSumLocal, cumSumGm[cumSumOffset_], cumSumCopyLength);
        DataCopy(srcBlockIdxLocal, srcBlockIndicesGm[cumSumOffset_], cumSumCopyLength);
        auto dstProcessBlocks = (processLength + TRANS_RATIO - 1) / TRANS_RATIO * TRANS_RATIO;
        DataCopy(dstBlockIdxLocal, dstBlockIndicesGm[gmOffset_ + progress * TILE_LENGTH], dstProcessBlocks);

        inQueueY.EnQue(dstBlockIdxLocal);
        inQueueX.EnQue(srcBlockIdxLocal);
        inQueueA.EnQue(cumSumLocal);
    }

    __aicore__ inline void CopyBlocks(int32_t progress, int32_t processLength)
    {
        LocalTensor<int32_t> dstBlockIdxLocal = inQueueY.DeQue<int32_t>();
        LocalTensor<int32_t> srcBlockIdxLocal = inQueueX.DeQue<int32_t>();
        LocalTensor<int32_t> cumSumLocal = inQueueA.DeQue<int32_t>();
        event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
        int32_t processIndex = 0;
        int32_t completedCount = 0;
        int64_t previousCumSum = gmOffset_ + progress * TILE_LENGTH;
        WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
        while (completedCount < processLength) {
            int32_t srcBlockIndex = srcBlockIdxLocal.GetValue(processIndex);
            int64_t currentCum = static_cast<int64_t>(cumSumLocal.GetValue(processIndex));
            // 在当前核处理范围内，源block地址srcBlockIndex对应的目标block地址数量
            int32_t paddingCount = (currentCum - previousCumSum) > (processLength - completedCount) ?
                                       (processLength - completedCount) :
                                       (currentCum - previousCumSum);
            if (currentCum - previousCumSum <= processLength - completedCount) {
                // 在当前核处理范围内，源block地址srcBlockIndex对应的目标block地址已处理完，考虑下一个源block地址
                processIndex++;
            }
            previousCumSum = currentCum;
            CopyOneSrc2MultiDst(srcBlockIndex, completedCount, paddingCount, dstBlockIdxLocal);
            completedCount += paddingCount;
        }
        cumSumOffset_ += processIndex;
        inQueueA.FreeTensor(cumSumLocal);
        inQueueX.FreeTensor(srcBlockIdxLocal);
        inQueueY.FreeTensor(dstBlockIdxLocal);
    }

    __aicore__ inline void CopyOneSrc2MultiDst(int32_t srcBlockIdx, int32_t completedCount, int32_t paddingCount,
                                               LocalTensor<int32_t> &dstIndices)
    {
        for (int i = 0; i < paddingCount; i++) {
            int32_t dstBlockIdx = dstIndices.GetValue(completedCount + i);
            event_t eventIDSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
            SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
            WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
            for (int32_t j = 0; j < cacheCopyLoopCount_ - 1; j++) {
                CopyOneBlockIn(srcBlockIdx, j, outUbLength_);
                CopyOneBlockOut(dstBlockIdx, j, outUbLength_);
            }
            CopyOneBlockIn(srcBlockIdx, cacheCopyLoopCount_ - 1, cacheCopyTailLength_);
            CopyOneBlockOut(dstBlockIdx, cacheCopyLoopCount_ - 1, cacheCopyTailLength_);
        }
    }

    __aicore__ inline void CopyOneBlockIn(int32_t srcBlockIndex, int32_t progress, int32_t processLength)
    {
        LocalTensor<T> src2dstLocalK = src2dstQueueK.AllocTensor<T>();
        LocalTensor<T> src2dstLocalV = src2dstQueueV.AllocTensor<T>();
        int32_t byteBlock = 32 / typeByte_;
        auto blockInLen = (processLength + byteBlock - 1) / byteBlock * byteBlock;
        DataCopy(src2dstLocalK, kCacheGm[srcBlockIndex * blockSizeofElement_ + progress * outUbLength_], blockInLen);
        DataCopy(src2dstLocalV, vCacheGm[srcBlockIndex * blockSizeofElement_ + progress * outUbLength_], blockInLen);
        src2dstQueueK.EnQue(src2dstLocalK);
        src2dstQueueV.EnQue(src2dstLocalV);
    }

    __aicore__ inline void CopyOneBlockOut(int32_t dstBlockIndex, int32_t progress, int32_t processLength)
    {
        LocalTensor<T> src2dstLocalK = src2dstQueueK.DeQue<T>();
        LocalTensor<T> src2dstLocalV = src2dstQueueV.DeQue<T>();
        int32_t byteBlock = 32 / typeByte_;
        auto outBlocks = (processLength + byteBlock - 1) / byteBlock * byteBlock;
        DataCopy(kCacheGm[dstBlockIndex * blockSizeofElement_ + progress * outUbLength_], src2dstLocalK, outBlocks);
        DataCopy(vCacheGm[dstBlockIndex * blockSizeofElement_ + progress * outUbLength_], src2dstLocalV, outBlocks);
        src2dstQueueK.FreeTensor(src2dstLocalK);
        src2dstQueueV.FreeTensor(src2dstLocalV);
    }

private:
    TPipe *pipe;
    uint32_t perCoreCopyCount_ = 0;
    uint32_t tailCoreCopyCount_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t blockCount_;
    uint32_t sourceCount_;
    uint32_t destinationCount_;
    uint64_t blockSizeofElement_;
    uint32_t typeByte_;
    uint32_t blockDim_;
    uint32_t outUbLength_ = 0;
    uint32_t cacheCopyLoopCount_ = 0;
    uint32_t cacheCopyTailLength_ = 0;
    int64_t gmOffset_ = 0;
    int64_t cumSumOffset_ = -1;

    GlobalTensor<int32_t> srcWorkSpaceGm;
    GlobalTensor<int32_t> syncGm;
    GlobalTensor<T> kCacheGm;
    GlobalTensor<T> vCacheGm;
    GlobalTensor<int32_t> srcBlockIndicesGm;
    GlobalTensor<int32_t> dstBlockIndicesGm;
    GlobalTensor<int32_t> cumSumGm;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueA;
    TBuf<TPosition::VECCALC> maskBuf;
    TBuf<TPosition::VECCALC> onesBuf;
    TBuf<TPosition::VECCALC> positionBuf;
    TBuf<TPosition::VECCALC> floatBuf;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> src2dstQueueK;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> src2dstQueueV;
};

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata,
                                      AtbOps::CustomizeBlockCopyTilingData *tilingdata)
{
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
}

extern "C" __global__ __aicore__ void customize_blockcopy(GM_ADDR kCache, GM_ADDR vCache, GM_ADDR srcBlockIndices,
                                                          GM_ADDR dstBlockIndices, GM_ADDR cumSum, GM_ADDR kCacheOut,
                                                          GM_ADDR vCacheOut, GM_ADDR tiling)
{
    AtbOps::CustomizeBlockCopyTilingData tilingData;
    InitTilingData(tiling, &(tilingData));
    TPipe pipe;
    if (TILING_KEY_IS(100000000)) {
        CustomizeBlockCopy310P<int8_t> op;
        op.Init(kCache, vCache, srcBlockIndices, dstBlockIndices, cumSum, &tilingData, &pipe);
        op.Process();
    }
    if (TILING_KEY_IS(200000000)) {
        CustomizeBlockCopy310P<half> op;
        op.Init(kCache, vCache, srcBlockIndices, dstBlockIndices, cumSum, &tilingData, &pipe);
        op.Process();
    }
}