/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ASCEND_OPS_PAD_WITH_HIDDEN_STATE_TILING_DATA_H
#define ASCEND_OPS_PAD_WITH_HIDDEN_STATE_TILING_DATA_H

#include <cstdint>

namespace AtbOps {

constexpr uint32_t MAX_BATCH_SIZE = 32;
constexpr uint32_t SEQLEN_LIMIT = 4096;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t ELEMENT_SIZE = 2;
constexpr uint32_t BASIC_BLOCK_SIZE = 32;
constexpr uint32_t ELEMENT_PER_BASIC_BLOCK = BASIC_BLOCK_SIZE / ELEMENT_SIZE;
constexpr uint32_t UB_SIZE = 192 * 1024;
constexpr uint32_t PAD_WITH_HIDDEN_STATE_COMPUTE_DIM = 2;

struct TileInfo {
    uint32_t formerCoreNum;
    uint32_t formerCoreTileNum;
    uint32_t lastTileLength;
};

struct PadWithHiddenStateTilingData {
    uint32_t coreNum;
    uint32_t batchSize;
    uint32_t maxSeqLen;
    uint32_t bufferSize;
    uint32_t tileLength;
    uint32_t totalDataLength;
    uint32_t unpadDataLength;
    uint32_t inputOffset[MAX_BATCH_SIZE];
    uint32_t outputOffset[MAX_BATCH_SIZE];
    uint32_t paddingOffset[MAX_BATCH_SIZE];
    TileInfo copyTileInfo[MAX_BATCH_SIZE];
    TileInfo paddingTileInfo[MAX_BATCH_SIZE];
};

}  // namespace AtbOps

#endif  // ASCEND_OPS_PAD_WITH_HIDDEN_STATE_TILING_DATA_H
