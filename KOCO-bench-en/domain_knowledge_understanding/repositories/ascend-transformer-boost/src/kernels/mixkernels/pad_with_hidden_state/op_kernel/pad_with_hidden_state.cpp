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
#include "mixkernels/pad_with_hidden_state/tiling/tiling_data.h"

using namespace AscendC;
using namespace AtbOps;

inline __aicore__ PadWithHiddenStateTilingData GetTilingData(const GM_ADDR tiling);

class PadWithHiddenState {
public:
    __aicore__ inline PadWithHiddenState() {}

    __aicore__ inline void Init(GM_ADDR dataInput, GM_ADDR dataOutput, const GM_ADDR tiling)
    {
        tilingData = GetTilingData(tiling);
        unpadDataGm.SetGlobalBuffer((__gm__ half *)dataInput, tilingData.unpadDataLength);
        padDataGm.SetGlobalBuffer((__gm__ half *)dataOutput, tilingData.totalDataLength);
        pipe.InitBuffer(inputQueue, BUFFER_NUM, tilingData.bufferSize);
        pipe.InitBuffer(outputQueue, BUFFER_NUM, tilingData.bufferSize);
    }

    __aicore__ inline void Process()
    {
        bool isLastCore = (GetBlockIdx() == tilingData.coreNum - 1);
        for (uint32_t sampleIndex = 0; sampleIndex < tilingData.batchSize; ++sampleIndex) {
            uint64_t inputOffset = static_cast<uint64_t>(tilingData.inputOffset[sampleIndex]);
            uint64_t outputOffset = static_cast<uint64_t>(tilingData.outputOffset[sampleIndex]);

            TileInfo &copyTileInfo = tilingData.copyTileInfo[sampleIndex];
            uint64_t tileNum = GetTileNum(copyTileInfo);
            uint64_t innerOffset = GetInnerOffset(copyTileInfo);
            for (uint32_t tileIndex = 0; tileIndex < tileNum; ++tileIndex, innerOffset += tilingData.tileLength) {
                CopyIn(inputOffset + innerOffset, tilingData.tileLength);
                Compute();
                CopyOut(outputOffset + innerOffset, tilingData.tileLength);
            }
            if (isLastCore && (copyTileInfo.lastTileLength > 0)) {
                CopyIn(inputOffset + innerOffset, copyTileInfo.lastTileLength);
                Compute();
                CopyOut(outputOffset + innerOffset, copyTileInfo.lastTileLength);
            }
        }
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t length)
    {
        LocalTensor<half> dataInputUb = inputQueue.AllocTensor<half>();
        if (length % ELEMENT_PER_BASIC_BLOCK == 0) {
            DataCopy(dataInputUb, unpadDataGm[offset], length);
        } else {
            DataCopyPad(dataInputUb, unpadDataGm[offset], DataCopyParams(1, length * ELEMENT_SIZE, 0, 0),
                DataCopyPadParams());
        }
        inputQueue.EnQue<half>(dataInputUb);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<half> dataInputUb = inputQueue.DeQue<half>();
        LocalTensor<half> dataOutputUb = outputQueue.AllocTensor<half>();
        DataCopy(dataOutputUb, dataInputUb, tilingData.tileLength);
        inputQueue.FreeTensor(dataInputUb);
        outputQueue.EnQue<half>(dataOutputUb);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t length)
    {
        LocalTensor<half> dataOutputUb = outputQueue.DeQue<half>();
        if (length % ELEMENT_PER_BASIC_BLOCK == 0) {
            DataCopy(padDataGm[offset], dataOutputUb, length);
        } else {
            DataCopyPad(padDataGm[offset], dataOutputUb, DataCopyParams(1, length * ELEMENT_SIZE, 0, 0));
        }
        outputQueue.FreeTensor(dataOutputUb);
    }

private:

    __aicore__ inline uint64_t GetTileNum(const TileInfo &tileInfo)
    {
        if (GetBlockIdx() < tileInfo.formerCoreNum) {
            return tileInfo.formerCoreTileNum;
        } else {
            return tileInfo.formerCoreTileNum - 1;
        }
    }

    __aicore__ inline uint64_t GetInnerOffset(const TileInfo &tileInfo)
    {
        if (GetBlockIdx() < tileInfo.formerCoreNum) {
            return GetBlockIdx() * tileInfo.formerCoreTileNum * tilingData.tileLength;
        } else {
            return (tileInfo.formerCoreNum * tileInfo.formerCoreTileNum +
                (GetBlockIdx() - tileInfo.formerCoreNum) * (tileInfo.formerCoreTileNum - 1)) * tilingData.tileLength;
        }
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    GlobalTensor<half> unpadDataGm;
    GlobalTensor<half> padDataGm;
    PadWithHiddenStateTilingData tilingData;
};

inline __aicore__ PadWithHiddenStateTilingData GetTilingData(const GM_ADDR tiling)
{
    auto tilingDataPointer = reinterpret_cast<const __gm__ PadWithHiddenStateTilingData *>(tiling);
    PadWithHiddenStateTilingData tilingData;
    tilingData.coreNum = tilingDataPointer->coreNum;
    tilingData.batchSize = tilingDataPointer->batchSize;
    tilingData.maxSeqLen = tilingDataPointer->maxSeqLen;
    tilingData.bufferSize = tilingDataPointer->bufferSize;
    tilingData.tileLength = tilingDataPointer->tileLength;
    tilingData.totalDataLength = tilingDataPointer->totalDataLength;
    tilingData.unpadDataLength = tilingDataPointer->unpadDataLength;
    for (uint32_t sampleIndex = 0; sampleIndex < tilingData.batchSize; ++sampleIndex) {
        tilingData.inputOffset[sampleIndex] = tilingDataPointer->inputOffset[sampleIndex];
        tilingData.outputOffset[sampleIndex] = tilingDataPointer->outputOffset[sampleIndex];
        tilingData.paddingOffset[sampleIndex] = tilingDataPointer->paddingOffset[sampleIndex];
        tilingData.copyTileInfo[sampleIndex].formerCoreNum =
            tilingDataPointer->copyTileInfo[sampleIndex].formerCoreNum;
        tilingData.copyTileInfo[sampleIndex].formerCoreTileNum =
            tilingDataPointer->copyTileInfo[sampleIndex].formerCoreTileNum;
        tilingData.copyTileInfo[sampleIndex].lastTileLength =
            tilingDataPointer->copyTileInfo[sampleIndex].lastTileLength;
        tilingData.paddingTileInfo[sampleIndex].formerCoreNum =
            tilingDataPointer->paddingTileInfo[sampleIndex].formerCoreNum;
        tilingData.paddingTileInfo[sampleIndex].formerCoreTileNum =
            tilingDataPointer->paddingTileInfo[sampleIndex].formerCoreTileNum;
        tilingData.paddingTileInfo[sampleIndex].lastTileLength =
            tilingDataPointer->paddingTileInfo[sampleIndex].lastTileLength;
    }
    return tilingData;
}

extern "C" __global__ __aicore__ void pad_with_hidden_state(GM_ADDR dataInput, GM_ADDR dataOutput, GM_ADDR tiling)
{
    PadWithHiddenState op;
    op.Init(dataInput, dataOutput, tiling);
    op.Process();
}
