/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ASCEND_OPS_RESHAPE_AND_CACHE_TILING_DEPENDENCY_H
#define ASCEND_OPS_RESHAPE_AND_CACHE_TILING_DEPENDENCY_H

#include <cstdint>

namespace AtbOps {
const uint32_t TASK_MULTIPLE = 2;       // wins_rope模式下KV分核，分核任务量翻倍
const uint32_t TILING_PARA_SIZE = 64;
constexpr uint32_t TILING_ID_DTYPE = 100000000;
constexpr uint32_t TILING_ID_MODE = 10000000;
constexpr uint32_t TILING_ID_MLA = 1000000;
constexpr uint32_t TILING_ID_MLA_FULL = 2000000;
constexpr uint32_t TILING_ID_NCT = 200000;
constexpr uint32_t SMALL_SHAPE = 1000;
constexpr uint32_t ALIGN = 32;
constexpr uint32_t SHAPE_DIMS = 3;
constexpr uint32_t ZERO = 0;
constexpr int32_t UB_SLOT_MAPPING_SIZE = 128 * 1024;

struct ReshapeAndCacheTilingData {
    uint32_t numTokens;
    uint32_t numHeads;
    uint32_t headSizeK;
    uint32_t headSizeV;
    uint32_t blockSize;
    uint32_t typeByte;
    uint32_t numBatchs;
};

struct ReshapeAndCacheNctTilingData {
    uint32_t numTokens;
    uint32_t numHeads;
    uint32_t headSizeK;
    uint32_t headSizeV;
    uint32_t blockSize;
    uint32_t typeByte;
    uint32_t numBatchs;
    uint32_t offsetK;
    uint32_t offsetV;
    uint32_t stride[SHAPE_DIMS];
};
}

#endif // ASCEND_OPS_RESHAPE_AND_CACHE_TILING_DEPENDENCY_H
