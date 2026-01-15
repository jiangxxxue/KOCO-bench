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
#include "mixkernels/fastsoftmax/tiling/tiling_data.h"

using namespace AscendC;
using namespace AtbOps;

class FastSoftmax {
public:
    __aicore__ inline FastSoftmax() {}
    __aicore__ inline void Init(GM_ADDR dataInput, GM_ADDR dataOutput, const FastSoftMaxSampleTilingData &tilingData)
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
        isLastCore = (tailTileRowNum > 0) &&
            (GetBlockIdx() == tilingData.formerCoreNum + tilingData.latterCoreNum - 1);

        dataInputGm.SetGlobalBuffer((__gm__ half*)dataInput + tilingData.dataOffset, tilingData.dataLength);
        dataOutputGm.SetGlobalBuffer((__gm__ half*)dataOutput + tilingData.dataOffset, tilingData.dataLength);

        softMaxTiling = *reinterpret_cast<const SoftMaxTiling *>(tilingData.softMaxTilingBuffer);
        tailSoftMaxTiling = *reinterpret_cast<const SoftMaxTiling *>(tilingData.tailSoftMaxTilingBuffer);

        pipe.InitBuffer(dataInputQueue, BUFFER_NUM, tileRowNum * tilingData.innerSize);
        pipe.InitBuffer(dataOutputQueue, BUFFER_NUM, tileRowNum * tilingData.innerSize);
        pipe.InitBuffer(sharedTmpBuffer, SHARED_TMP_SIZE);
        softmaxTmpUb = sharedTmpBuffer.Get<uint8_t>();
    }

    __aicore__ inline void Process()
    {
        for (uint32_t tileIndex = 0; tileIndex < coreTileNum; ++tileIndex) {
            CopyIn(tileRowNum);
            Compute(tileRowNum, softMaxTiling);
            CopyOut(tileRowNum);
            innerDataOffset += tileLengthOrigin;
        }
        if (isLastCore) {
            CopyIn(tailTileRowNum);
            Compute(tailTileRowNum, tailSoftMaxTiling);
            CopyOut(tailTileRowNum);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t currentTileRowNum)
    {
        LocalTensor<half> dataInputUb = dataInputQueue.AllocTensor<half>();
        DataCopyPad(dataInputUb, dataInputGm[innerDataOffset],
            DataCopyParams(currentTileRowNum, sampleSeqLenOrigin * DATA_BYTESIZE, 0, 0), DataCopyPadParams());
        dataInputQueue.EnQue<half>(dataInputUb);
    }

    __aicore__ inline void Compute(uint32_t currentTileRowNum, const SoftMaxTiling &currentSoftMaxTiling)
    {
        LocalTensor<half> dataInputUb = dataInputQueue.DeQue<half>();
        LocalTensor<half> dataOutputUb = dataOutputQueue.AllocTensor<half>();

        SoftMaxShapeInfo srcShape = { currentTileRowNum, sampleSeqLen, currentTileRowNum, sampleSeqLenOrigin };
        SoftMax<half, false>(dataOutputUb, dataInputUb, softmaxTmpUb, currentSoftMaxTiling, srcShape);

        dataOutputQueue.EnQue<half>(dataOutputUb);
        dataInputQueue.FreeTensor<half>(dataInputUb);
    }

    __aicore__ inline void CopyOut(uint32_t currentTileRowNum)
    {
        LocalTensor<half> dataOutputUb = dataOutputQueue.DeQue<half>();
        DataCopyPad(dataOutputGm[innerDataOffset], dataOutputUb,
            DataCopyParams(currentTileRowNum, sampleSeqLenOrigin * DATA_BYTESIZE, 0, 0));
        dataOutputQueue.FreeTensor(dataOutputUb);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> dataInputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dataOutputQueue;
    TBuf<TPosition::VECCALC> sharedTmpBuffer;
    LocalTensor<uint8_t> softmaxTmpUb;
    GlobalTensor<half> dataInputGm;
    GlobalTensor<half> dataOutputGm;
    uint32_t sampleSeqLenOrigin;
    uint32_t sampleSeqLen;
    uint32_t tileRowNum;
    uint32_t tailTileRowNum;
    uint32_t coreTileNum;
    uint64_t innerDataOffset;
    uint32_t tileLengthOrigin;
    bool isLastCore;
    SoftMaxTiling softMaxTiling;
    SoftMaxTiling tailSoftMaxTiling;
};

inline __aicore__ FastSoftMaxTilingData GetTilingData(const GM_ADDR tiling)
{
    auto tilingDataPointer = reinterpret_cast<const __gm__ FastSoftMaxTilingData *>(tiling);
    FastSoftMaxTilingData tilingData;
    tilingData.batchSize = tilingDataPointer->batchSize;
    tilingData.headNum = tilingDataPointer->headNum;
    return tilingData;
}

inline __aicore__ FastSoftMaxSampleTilingData GetSampleTilingData(const GM_ADDR tiling)
{
    auto sampleTilingDataPointer = reinterpret_cast<const __gm__ FastSoftMaxSampleTilingData *>(tiling);
    FastSoftMaxSampleTilingData sampleTilingData;
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
        sampleTilingData.softMaxTilingBuffer[i] = sampleTilingDataPointer->softMaxTilingBuffer[i];
        sampleTilingData.tailSoftMaxTilingBuffer[i] = sampleTilingDataPointer->tailSoftMaxTilingBuffer[i];
    }
    return sampleTilingData;
}

extern "C" __global__ __aicore__ void fastsoftmax(GM_ADDR dataInput, GM_ADDR dataOutput, GM_ADDR tiling)
{
    FastSoftMaxTilingData tilingData = GetTilingData(tiling);
    tiling += sizeof(FastSoftMaxTilingData);
    for (uint32_t sampleIndex = 0; sampleIndex < tilingData.batchSize; ++sampleIndex) {
        FastSoftMaxSampleTilingData sampleTilingData = GetSampleTilingData(tiling);
        FastSoftmax op;
        op.Init(dataInput, dataOutput, sampleTilingData);
        op.Process();
        tiling += sizeof(FastSoftMaxSampleTilingData);
    }
}
