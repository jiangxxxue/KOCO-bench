/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "kernels/utils/kernel/common.h"
#include "kernel_operator.h"
namespace PagedCacheLoad {

constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t TYPEBYTE_ID = 5;
constexpr int32_t UB_BUF_SIZE = 192 * 1024;
constexpr int32_t VEC_SUBDIM = 2;

template <typename T>
class PagedCacheLoadNd {
public:
    __aicore__ inline PagedCacheLoadNd(){};
    __aicore__ inline void Init(__gm__ uint8_t * __restrict__ keyCacheInGm,
                                __gm__ uint8_t * __restrict__ valueCacheInGm,
                                __gm__ uint8_t * __restrict__ blockTablesInGm,
                                __gm__ uint8_t * __restrict__ contextLensInGm,
                                __gm__ uint8_t * __restrict__ seqStartsInGm,
                                __gm__ uint8_t * __restrict__ keyOutGm,
                                __gm__ uint8_t * __restrict__ valueOutGm,
                                __gm__ uint8_t * __restrict__ tilingParaGm);
    __aicore__ inline void Process();
    __aicore__ inline void ParseTilingData(__gm__ uint8_t * __restrict__ tilingGm);

private:
    AscendC::TPipe pipe;

    // input

    AscendC::GlobalTensor<T> inKeyCacheGM;
    AscendC::GlobalTensor<T> inValueCacheGM;
    AscendC::GlobalTensor<int32_t> inBlockTableGM;
    AscendC::GlobalTensor<int32_t> inContextLenGM;
    AscendC::GlobalTensor<int32_t> inSeqStartsGM;

    // output
    AscendC::GlobalTensor<T> outKeyGM;
    AscendC::GlobalTensor<T> outValueGM;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf;
    AscendC::LocalTensor<T> tmpTensor;

    // tiling param
    int32_t blockSize = 0;
    int32_t numBatchs = 0;
    int32_t numblkTabCol = 0;
    int32_t tokenSizeK = 0;
    int32_t tokenSizeV = 0;
    int32_t typeByte = 0;
    int32_t hasSeqStarts = 0;
    int32_t cuSeqLens = 0;

    int32_t ubLoopPerBlkK = 0;
    int32_t ubLoopPerBlkV = 0;
    int32_t ubBaseK = 0;
    int32_t ubBaseV = 0;
    int32_t ubBaseKTail = 0;
    int32_t ubBaseVTail = 0;
    int32_t cachelineBlkKTail = 0;
    int32_t cachelineBlkVTail = 0;
};

template <typename T>
__aicore__ inline void PagedCacheLoadNd<T>::Init(
    __gm__ uint8_t * __restrict__ keyCacheInGm,
    __gm__ uint8_t * __restrict__ valueCacheInGm,
    __gm__ uint8_t * __restrict__ blockTablesInGm,
    __gm__ uint8_t * __restrict__ contextLensInGm,
    __gm__ uint8_t * __restrict__ seqStartsInGm,
    __gm__ uint8_t * __restrict__ keyOutGm,
    __gm__ uint8_t * __restrict__ valueOutGm,
    __gm__ uint8_t * __restrict__ tilingParaGm)
{
    ParseTilingData(tilingParaGm);

    // input
    inKeyCacheGM.SetGlobalBuffer((__gm__ T *)keyCacheInGm);
    inValueCacheGM.SetGlobalBuffer((__gm__ T *)valueCacheInGm);
    inBlockTableGM.SetGlobalBuffer((__gm__ int32_t *)blockTablesInGm);
    inContextLenGM.SetGlobalBuffer((__gm__ int32_t *)contextLensInGm);
    inSeqStartsGM.SetGlobalBuffer((__gm__ int32_t *)seqStartsInGm);
    // output
    outKeyGM.SetGlobalBuffer((__gm__ T *)keyOutGm);
    outValueGM.SetGlobalBuffer((__gm__ T *)valueOutGm);

    pipe.InitBuffer(tmpBuf, UB_BUF_SIZE); // ensure the size of one token less than 148kb
    tmpTensor = tmpBuf.Get<T>();
 
    ubLoopPerBlkK = (blockSize * tokenSizeK * sizeof(T) + UB_BUF_SIZE - 1) / UB_BUF_SIZE;
    ubLoopPerBlkV = (blockSize * tokenSizeV * sizeof(T) + UB_BUF_SIZE - 1) / UB_BUF_SIZE;
    int32_t cachelineBlkK = (blockSize * tokenSizeK * sizeof(T) + 512 - 1) / 512;
    cachelineBlkKTail = blockSize * tokenSizeK * sizeof(T) % 512;
    cachelineBlkKTail = cachelineBlkKTail == 0 ? 512 : cachelineBlkKTail;
    ubBaseK = cachelineBlkK / ubLoopPerBlkK;
    ubBaseKTail = cachelineBlkK % ubLoopPerBlkK;
    int32_t cachelineBlkV = (blockSize * tokenSizeV * sizeof(T) + 512 - 1) / 512;
    cachelineBlkVTail = blockSize * tokenSizeV * sizeof(T) % 512;
    cachelineBlkVTail = cachelineBlkVTail == 0 ? 512 : cachelineBlkVTail;
    ubBaseV = cachelineBlkV / ubLoopPerBlkV;
    ubBaseVTail = cachelineBlkV % ubLoopPerBlkV;
}


template <typename T>
__aicore__ inline void PagedCacheLoadNd<T>::ParseTilingData(__gm__ uint8_t * __restrict__ tilingGm)
{
    auto tilingBuf = reinterpret_cast<__gm__ uint8_t *>(tilingGm);

    int64_t locId = 0;
    blockSize = (*(__gm__ int32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(int32_t)));
    locId++;
    numBatchs = (*(__gm__ int32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(int32_t)));
    locId++;
    numblkTabCol = (*(__gm__ int32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(int32_t)));
    locId++;
    tokenSizeK = (*(__gm__ int32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(int32_t)));
    locId++;
    tokenSizeV = (*(__gm__ int32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(int32_t)));
    locId++;
    typeByte = (*(__gm__ int32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(int32_t)));
    locId++;
    hasSeqStarts = (*(__gm__ int32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(int32_t)));
    locId++;
    cuSeqLens = (*(__gm__ int32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(int32_t)));
}

template <typename T>
__aicore__ inline void PagedCacheLoadNd<T>::Process()
{
    int32_t vecIdx = AscendC::GetBlockIdx(); // current vector core id
    int32_t coreNum = AscendC::GetBlockNum(); // total vector core number
    vecIdx = (vecIdx % VEC_SUBDIM) * (coreNum / VEC_SUBDIM) + (vecIdx / VEC_SUBDIM);
    int32_t taskID = 0;
    int32_t tokenStart = 0;
    if (cuSeqLens) tokenStart = inContextLenGM.GetValue(0);
    for (int32_t batchId = 0; batchId < numBatchs; batchId++) {
        int32_t tokenEnd, numToken;
        if (cuSeqLens) {
            tokenEnd = inContextLenGM.GetValue(batchId + 1);
            numToken = tokenEnd - tokenStart;
        } else {
            numToken = inContextLenGM.GetValue(batchId);
            tokenEnd = tokenStart + numToken;
        }
        int32_t numBlk = (numToken + blockSize - 1) / blockSize;
        int32_t lastBlkToken = numToken % blockSize;
        lastBlkToken = lastBlkToken == 0 ? blockSize : lastBlkToken;
        int32_t tableOffset = 0;
        if (hasSeqStarts) {
            tableOffset = inSeqStartsGM.GetValue(batchId) / blockSize;
        }
        for (int tableID = 0; tableID < numBlk; tableID++) {
            int32_t blockID = inBlockTableGM.GetValue(batchId * numblkTabCol + tableOffset + tableID);
            int32_t ubLoopPerBlkK_ = ubLoopPerBlkK;
            int32_t ubLoopPerBlkV_ = ubLoopPerBlkV;
            int32_t cachelineBlkKTail_ = cachelineBlkKTail;
            int32_t ubBaseK_ = ubBaseK;
            int32_t ubBaseKTail_ = ubBaseKTail;
            int32_t cachelineBlkVTail_ = cachelineBlkVTail;
            int32_t ubBaseV_ = ubBaseV;
            int32_t ubBaseVTail_ = ubBaseVTail;
            if (tableID == (numBlk - 1)) {
                ubLoopPerBlkK_ = (lastBlkToken * tokenSizeK * sizeof(T) + UB_BUF_SIZE - 1) / UB_BUF_SIZE;
                ubLoopPerBlkV_ = (lastBlkToken * tokenSizeV * sizeof(T) + UB_BUF_SIZE - 1) / UB_BUF_SIZE;
                int32_t cachelineBlkK = (lastBlkToken * tokenSizeK * sizeof(T) + 512 - 1) / 512;
                cachelineBlkKTail_ = lastBlkToken * tokenSizeK * sizeof(T) % 512;
                cachelineBlkKTail_ = cachelineBlkKTail_ == 0 ? 512 : cachelineBlkKTail_;
                ubBaseK_ = cachelineBlkK / ubLoopPerBlkK_;
                ubBaseKTail_ = cachelineBlkK % ubLoopPerBlkK_;
                int32_t cachelineBlkV = (lastBlkToken * tokenSizeV * sizeof(T) + 512 - 1) / 512;
                cachelineBlkVTail_ = lastBlkToken * tokenSizeV * sizeof(T) % 512;
                cachelineBlkVTail_ = cachelineBlkVTail_ == 0 ? 512 : cachelineBlkVTail_;
                ubBaseV_ = cachelineBlkV / ubLoopPerBlkV_;
                ubBaseVTail_ = cachelineBlkV % ubLoopPerBlkV_;
            }
            int32_t blkKOffset = 0;
            for (int loopID = 0; loopID < ubLoopPerBlkK_; loopID++) {
                int32_t realUbBaseK_ = (ubBaseK_ + (loopID < ubBaseKTail_ ? 1 : 0)) * 512 / sizeof(T);
                if (loopID == (ubLoopPerBlkK_ - 1)) {
                    realUbBaseK_ += (cachelineBlkKTail_ - 512) / sizeof(T);
                }
                if (taskID++ % coreNum == vecIdx) {
                    uint16_t copyLen = realUbBaseK_ * sizeof(T) / 32;
                    AscendC::DataCopyParams copyParams = {1, copyLen, 0, 0};
                    DataCopy(tmpTensor, inKeyCacheGM[blockID * blockSize * tokenSizeK + blkKOffset], copyParams);
                    SET_FLAG(MTE2, MTE3, EVENT_ID1);
                    WAIT_FLAG(MTE2, MTE3, EVENT_ID1);
                    DataCopy(outKeyGM[(tokenStart + tableID * blockSize) * tokenSizeK + blkKOffset], tmpTensor, copyParams);
                    SET_FLAG(MTE3, MTE2, EVENT_ID1);
                    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
                }
                blkKOffset += realUbBaseK_;
            }
            int32_t blkVOffset = 0;
            for (int loopID = 0; loopID < ubLoopPerBlkV_; loopID++) {
                int32_t realUbBaseV_ = (ubBaseV_ + (loopID < ubBaseVTail_ ? 1 : 0)) * 512 / sizeof(T);
                if (loopID == (ubLoopPerBlkV_ - 1)) {
                    realUbBaseV_ += (cachelineBlkVTail_ - 512) / sizeof(T);
                }
                if (taskID++ % coreNum == vecIdx) {
                    uint16_t copyLen = realUbBaseV_ * sizeof(T) / 32;
                    AscendC::DataCopyParams copyParams = {1, copyLen, 0, 0};
                    DataCopy(tmpTensor, inValueCacheGM[blockID * blockSize * tokenSizeV + blkVOffset], copyParams);
                    SET_FLAG(MTE2, MTE3, EVENT_ID1);
                    WAIT_FLAG(MTE2, MTE3, EVENT_ID1);
                    DataCopy(outValueGM[(tokenStart + tableID * blockSize) * tokenSizeV + blkVOffset], tmpTensor, copyParams);
                    SET_FLAG(MTE3, MTE2, EVENT_ID1);
                    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
                }
                blkVOffset += realUbBaseV_;
            }
        }
        tokenStart = tokenEnd;
    }
}
}


extern "C" __global__ __aicore__ void paged_cache_load_nd(
    __gm__ uint8_t * __restrict__ keyCacheInGm,
    __gm__ uint8_t * __restrict__ valueCacheInGm,
    __gm__ uint8_t * __restrict__ blockTablesInGm,
    __gm__ uint8_t * __restrict__ contextLensInGm,
    __gm__ uint8_t * __restrict__ keyInGm,
    __gm__ uint8_t * __restrict__ valueInGm,
    __gm__ uint8_t * __restrict__ seqStartsInGm,
    __gm__ uint8_t * __restrict__ keyOutGm,
    __gm__ uint8_t * __restrict__ valueOutGm,
    __gm__ uint8_t * __restrict__ tilingParaGm)
{
    int32_t typeByte = ((__gm__ uint32_t *)tilingParaGm)[PagedCacheLoad::TYPEBYTE_ID];
    if (typeByte == 1) {
        PagedCacheLoad::PagedCacheLoadNd<int8_t> op;
        op.Init(keyCacheInGm, valueCacheInGm, blockTablesInGm,
                contextLensInGm, seqStartsInGm, keyOutGm, valueOutGm, tilingParaGm);
        op.Process();
    } else if (typeByte == 2) {
        PagedCacheLoad::PagedCacheLoadNd<half> op;
        op.Init(keyCacheInGm, valueCacheInGm, blockTablesInGm,
                contextLensInGm, seqStartsInGm, keyOutGm, valueOutGm, tilingParaGm);
        op.Process();
    }
}