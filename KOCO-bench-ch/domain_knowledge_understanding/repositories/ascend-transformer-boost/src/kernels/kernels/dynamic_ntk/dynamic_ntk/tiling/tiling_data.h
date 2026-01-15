/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDOPS_DYNAMIC_NTK_TILING_DATA
#define ASDOPS_DYNAMIC_NTK_TILING_DATA

#include <cstdint>

namespace AsdOps {
struct DynamicNTKTilingData {
    uint32_t numTokens;
    uint32_t headDim;
    uint32_t batchNum;
    uint32_t freqTileLen;
    uint32_t freqTileNum;
    uint32_t freqTailLen;
    uint32_t posTileLen;
    uint32_t posLongCores;
    uint32_t posShortCores;
    uint32_t posShortLen;
    uint32_t posLongLen;
    uint32_t posTailCoreLen;
};
}
#endif