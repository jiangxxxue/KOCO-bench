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
#include "mixkernels/unpad_with_hidden_state/tiling/tiling_data.h"

using namespace AscendC;
using namespace AtbOps;

inline __aicore__ UnpadWithHiddenStateTilingData GetTilingData(const GM_ADDR tiling);

class UnpadWithHiddenState {
public:
    __aicore__ inline UnpadWithHiddenState() {}

    __aicore__ inline void Init(GM_ADDR dataInput, GM_ADDR dataOutput, const GM_ADDR tiling)
    {
        tilingData = GetTilingData(tiling);
        padDataGm.SetGlobalBuffer((__gm__ half *)dataInput, tilingData.totalDataLength);
        unpadDataGm.SetGlobalBuffer((__gm__ half *)dataOutput, tilingData.unpadDataLength);
        pipe.InitBuffer(inputQueue, BUFFER_NUM, tilingData.bufferSize);
        pipe.InitBuffer(outputQueue, BUFFER_NUM, tilingData.bufferSize);
    }

    __aicore__ inline void Process()
    {
        bool isLastCore = (GetBlockIdx() == tilingData.coreNum - 1);
        for (uint32_t sampleIndex = 0; sampleIndex < tilingData.batchSize; ++sampleIndex) {
            uint64_t inputOffset = static_cast<uint64_t>(tilingData.inputOffset[sampleIndex]);
            uint64_t outputOffset = static_cast<uint64_t>(tilingData.outputOffset[sampleIndex]);

            TileInfo &tileInfo = tilingData.tileInfo[sampleIndex];
            uint32_t tileNum;
            if (GetBlockIdx() < tileInfo.formerCoreNum) {
                tileNum = tileInfo.formerCoreTileNum;
            } else {
                tileNum = tileInfo.formerCoreTileNum - 1;
            }
            uint64_t innerOffset;
            if (GetBlockIdx() < tileInfo.formerCoreNum) {
                innerOffset = static_cast<uint64_t>(GetBlockIdx()) * tileInfo.formerCoreTileNum *
                    tilingData.tileLength;
            } else {
                innerOffset = static_cast<uint64_t>((tileInfo.formerCoreNum * tileInfo.formerCoreTileNum +
                    (GetBlockIdx() - tileInfo.formerCoreNum) * (tileInfo.formerCoreTileNum - 1)) *
                        tilingData.tileLength);
            }

            for (uint32_t tileIndex = 0; tileIndex < tileNum; ++tileIndex) {
                CopyIn(inputOffset + innerOffset, tilingData.tileLength);
                Compute();
                CopyOut(outputOffset + innerOffset, tilingData.tileLength);
                innerOffset += tilingData.tileLength;
            }
            if (isLastCore && (tileInfo.lastTileLength > 0)) {
                CopyIn(inputOffset + innerOffset, tileInfo.lastTileLength);
                Compute();
                CopyOut(outputOffset + innerOffset, tileInfo.lastTileLength);
            }
        }
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t length)
    {
        LocalTensor<half> dataInputUb = inputQueue.AllocTensor<half>();
        if (length % ELEMENT_PER_BASIC_BLOCK == 0) {
            DataCopy(dataInputUb, padDataGm[offset], length);
        } else {
            DataCopyPad(dataInputUb, padDataGm[offset], DataCopyParams(1, length * ELEMENT_SIZE, 0, 0),
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
            DataCopy(unpadDataGm[offset], dataOutputUb, length);
        } else {
            DataCopyPad(unpadDataGm[offset], dataOutputUb, DataCopyParams(1, length * ELEMENT_SIZE, 0, 0));
        }
        outputQueue.FreeTensor(dataOutputUb);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    GlobalTensor<half> padDataGm;
    GlobalTensor<half> unpadDataGm;
    UnpadWithHiddenStateTilingData tilingData;
};

inline __aicore__ UnpadWithHiddenStateTilingData GetTilingData(const GM_ADDR tiling)
{
    auto tilingDataPointer = reinterpret_cast<const __gm__ UnpadWithHiddenStateTilingData *>(tiling);
    UnpadWithHiddenStateTilingData tilingData;
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
        tilingData.tileInfo[sampleIndex].formerCoreNum = tilingDataPointer->tileInfo[sampleIndex].formerCoreNum;
        tilingData.tileInfo[sampleIndex].formerCoreTileNum = tilingDataPointer->tileInfo[sampleIndex].formerCoreTileNum;
        tilingData.tileInfo[sampleIndex].lastTileLength = tilingDataPointer->tileInfo[sampleIndex].lastTileLength;
    }
    return tilingData;
}

extern "C" __global__ __aicore__ void unpad_with_hidden_state(GM_ADDR dataInput, GM_ADDR dataOutput, GM_ADDR tiling)
{
    UnpadWithHiddenState op;
    op.Init(dataInput, dataOutput, tiling);
    op.Process();
}
