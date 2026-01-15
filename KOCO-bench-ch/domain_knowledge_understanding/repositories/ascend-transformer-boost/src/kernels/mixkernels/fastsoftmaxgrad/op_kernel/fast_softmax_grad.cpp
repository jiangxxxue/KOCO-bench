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
#include "mixkernels/fastsoftmaxgrad/tiling/tiling_data.h"

using namespace AscendC;
using namespace AtbOps;

class FastSoftmaxGrad {
public:
    __aicore__ inline FastSoftmaxGrad() {}
    __aicore__ inline void Init(GM_ADDR yInput, GM_ADDR yGrad, GM_ADDR xGrad,
        const FastSoftMaxGradSampleTilingData &tilingData)
    {
        sampleSeqLenOrigin = tilingData.sampleSeqLenOrigin;
        sampleSeqLen = tilingData.sampleSeqLen;
        tileRowNum = tilingData.tileRowNum;
        tailTileRowNum = tilingData.tailTileRowNum;
        tileLengthOrigin = tileRowNum * sampleSeqLenOrigin;
        if (GetBlockIdx() < tilingData.formerCoreNum) {
            coreTileNum = tilingData.formerCoreTileNum;
            innerDataOffset = static_cast<uint64_t>(GetBlockIdx()) * tilingData.formerCoreTileNum * tileLengthOrigin;
        } else {
            coreTileNum = tilingData.latterCoreTileNum;
            innerDataOffset = static_cast<uint64_t>((GetBlockIdx() * tilingData.latterCoreTileNum +
                tilingData.formerCoreNum) * tileLengthOrigin);
        }
        isLastCore =
            (tailTileRowNum > 0) && (GetBlockIdx() == tilingData.formerCoreNum + tilingData.latterCoreNum - 1);

        yInputGm.SetGlobalBuffer((__gm__ half*)yInput + tilingData.dataOffset, tilingData.dataLength);
        yGradGm.SetGlobalBuffer((__gm__ half*)yGrad + tilingData.dataOffset, tilingData.dataLength);
        xGradGm.SetGlobalBuffer((__gm__ half*)xGrad + tilingData.dataOffset, tilingData.dataLength);

        softMaxGradTiling = *reinterpret_cast<const SoftMaxTiling *>(tilingData.softMaxGradTilingBuffer);
        tailSoftMaxGradTiling = *reinterpret_cast<const SoftMaxTiling *>(tilingData.tailSoftMaxGradTilingBuffer);

        pipe.InitBuffer(yInputQueue, BUFFER_NUM, tileRowNum * tilingData.innerSize);
        pipe.InitBuffer(yGradQueue, BUFFER_NUM, tileRowNum * tilingData.innerSize);
        pipe.InitBuffer(xGradQueue, BUFFER_NUM, tileRowNum * tilingData.innerSize);
        pipe.InitBuffer(sharedTmpBuffer, SHARED_TMP_SIZE);
        softmaxTmpUb = sharedTmpBuffer.Get<uint8_t>();
    }

    __aicore__ inline void Process()
    {
        for (uint32_t tileIndex = 0; tileIndex < coreTileNum; ++tileIndex) {
            CopyIn(tileRowNum);
            Compute(tileRowNum, softMaxGradTiling);
            CopyOut(tileRowNum);
            innerDataOffset += tileLengthOrigin;
        }
        if (isLastCore) {
            CopyIn(tailTileRowNum);
            Compute(tailTileRowNum, tailSoftMaxGradTiling);
            CopyOut(tailTileRowNum);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t currentTileRowNum)
    {
        LocalTensor<half> yInputUb = yInputQueue.AllocTensor<half>();
        LocalTensor<half> yGradUb = yGradQueue.AllocTensor<half>();
        DataCopyPad(yInputUb, yInputGm[innerDataOffset],
            DataCopyParams(currentTileRowNum, sampleSeqLenOrigin * DATA_BYTESIZE, 0, 0), DataCopyPadParams());
        DataCopyPad(yGradUb, yGradGm[innerDataOffset],
            DataCopyParams(currentTileRowNum, sampleSeqLenOrigin * DATA_BYTESIZE, 0, 0), DataCopyPadParams());
        yInputQueue.EnQue<half>(yInputUb);
        yGradQueue.EnQue<half>(yGradUb);
    }

    __aicore__ inline void Compute(uint32_t currentTileRowNum, const SoftMaxTiling &currentSoftMaxTiling)
    {
        LocalTensor<half> yInputUb = yInputQueue.DeQue<half>();
        LocalTensor<half> yGradUb = yGradQueue.DeQue<half>();
        LocalTensor<half> xGradUb = xGradQueue.AllocTensor<half>();

        SoftMaxShapeInfo srcShape = { currentTileRowNum, sampleSeqLen, currentTileRowNum, sampleSeqLenOrigin };
        SoftmaxGrad<half, false>(xGradUb, yGradUb, yInputUb, softmaxTmpUb, currentSoftMaxTiling, false, srcShape);

        xGradQueue.EnQue<half>(xGradUb);
        yInputQueue.FreeTensor<half>(yInputUb);
        yGradQueue.FreeTensor<half>(yGradUb);
    }

    __aicore__ inline void CopyOut(uint32_t currentTileRowNum)
    {
        LocalTensor<half> xGradUb = xGradQueue.DeQue<half>();
        DataCopyPad(xGradGm[innerDataOffset], xGradUb,
            DataCopyParams(currentTileRowNum, sampleSeqLenOrigin * DATA_BYTESIZE, 0, 0));
        xGradQueue.FreeTensor(xGradUb);
    }

    int64_t headNum;
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> yInputQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> yGradQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> xGradQueue;
    TBuf<TPosition::VECCALC> sharedTmpBuffer;
    LocalTensor<uint8_t> softmaxTmpUb;
    GlobalTensor<half> yInputGm;
    GlobalTensor<half> yGradGm;
    GlobalTensor<half> xGradGm;
    uint32_t sampleSeqLenOrigin;
    uint32_t sampleSeqLen;
    uint32_t tileRowNum;
    uint32_t tailTileRowNum;
    uint32_t coreTileNum;
    uint64_t innerDataOffset;
    uint32_t tileLengthOrigin;
    bool isLastCore;
    SoftMaxTiling softMaxGradTiling;
    SoftMaxTiling tailSoftMaxGradTiling;
};

inline __aicore__ FastSoftMaxGradTilingData GetTilingData(const GM_ADDR tiling)
{
    auto tilingDataPointer = reinterpret_cast<const __gm__ FastSoftMaxGradTilingData *>(tiling);
    FastSoftMaxGradTilingData tilingData;
    tilingData.batchSize = tilingDataPointer->batchSize;
    tilingData.headNum = tilingDataPointer->headNum;
    return tilingData;
}

inline __aicore__ FastSoftMaxGradSampleTilingData GetSampleTilingData(const GM_ADDR tiling)
{
    auto sampleTilingDataPointer = reinterpret_cast<const __gm__ FastSoftMaxGradSampleTilingData *>(tiling);
    FastSoftMaxGradSampleTilingData sampleTilingData;
    sampleTilingData.sampleSeqLenOrigin = sampleTilingDataPointer->sampleSeqLenOrigin;
    sampleTilingData.sampleSeqLen = sampleTilingDataPointer->sampleSeqLen;
    sampleTilingData.dataOffset = sampleTilingDataPointer->dataOffset;
    sampleTilingData.dataLength = sampleTilingDataPointer->dataLength;
    sampleTilingData.outerSize = sampleTilingDataPointer->outerSize;
    sampleTilingData.innerSize = sampleTilingDataPointer->innerSize;
    sampleTilingData.tileRowNum = sampleTilingDataPointer->tileRowNum;
    sampleTilingData.tailTileRowNum = sampleTilingDataPointer->tailTileRowNum;
    sampleTilingData.formerCoreNum = sampleTilingDataPointer->formerCoreNum;
    sampleTilingData.latterCoreNum = sampleTilingDataPointer->latterCoreNum;
    sampleTilingData.formerCoreTileNum = sampleTilingDataPointer->formerCoreTileNum;
    sampleTilingData.latterCoreTileNum = sampleTilingDataPointer->latterCoreTileNum;
    for (uint32_t i = 0; i < SOFTMAX_TILING_SIZE; ++i) {
        sampleTilingData.softMaxGradTilingBuffer[i] = sampleTilingDataPointer->softMaxGradTilingBuffer[i];
        sampleTilingData.tailSoftMaxGradTilingBuffer[i] = sampleTilingDataPointer->tailSoftMaxGradTilingBuffer[i];
    }
    return sampleTilingData;
}

extern "C" __global__ __aicore__ void fastsoftmaxgrad(GM_ADDR yInput, GM_ADDR yGrad, GM_ADDR xGrad, GM_ADDR tiling)
{
    FastSoftMaxGradTilingData tilingData = GetTilingData(tiling);
    tiling += sizeof(FastSoftMaxGradTilingData);
    for (uint32_t sampleIndex = 0; sampleIndex < tilingData.batchSize; ++sampleIndex) {
        FastSoftMaxGradSampleTilingData sampleTilingData = GetSampleTilingData(tiling);
        FastSoftmaxGrad op;
        op.Init(yInput, yGrad, xGrad, sampleTilingData);
        op.Process();
        tiling += sizeof(FastSoftMaxGradSampleTilingData);
    }
}
