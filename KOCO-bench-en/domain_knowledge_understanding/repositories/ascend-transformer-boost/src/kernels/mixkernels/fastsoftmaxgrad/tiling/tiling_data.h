/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ASCEND_OPS_FASTSOFTMAXGRAD_TILING_DATA_H
#define ASCEND_OPS_FASTSOFTMAXGRAD_TILING_DATA_H

#include <cstdint>

namespace AtbOps {
constexpr uint32_t MAX_BATCH_SIZE = 32;
constexpr uint32_t MAX_SEQ_LEN = 4096;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t DATA_BYTESIZE = 2;
constexpr uint32_t BASICBLOCK_SIZE = 32;
constexpr uint32_t ELEMENT_PER_BASICBLOCK = BASICBLOCK_SIZE / DATA_BYTESIZE;
constexpr uint32_t ONE_BYTE_BIT_SIZE = 8;
constexpr uint32_t SOFTMAXGRAD_COMPUTE_DIM = 3;
constexpr uint32_t SOFTMAX_TILING_SIZE = 64;
constexpr uint32_t SHARED_TMP_SIZE = 49536;

struct FastSoftMaxGradTilingData {
    uint32_t batchSize;
    uint32_t headNum;
};

struct FastSoftMaxGradSampleTilingData {
    uint32_t sampleSeqLenOrigin;
    uint32_t sampleSeqLen;
    uint32_t dataOffset;
    uint32_t dataLength;
    uint32_t outerSize;
    uint32_t innerSize;
    uint32_t tileRowNum;
    uint32_t tailTileRowNum;
    uint32_t formerCoreNum;
    uint32_t latterCoreNum;
    uint32_t formerCoreTileNum;
    uint32_t latterCoreTileNum;
    uint8_t softMaxGradTilingBuffer[SOFTMAX_TILING_SIZE];
    uint8_t tailSoftMaxGradTilingBuffer[SOFTMAX_TILING_SIZE];
};

}  // namespace AtbOps

#endif