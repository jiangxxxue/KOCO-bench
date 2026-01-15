/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ASCEND_OPS_RMS_NORM_AND_ROPE_AND_RESHAPE_AND_CACHE_TILING_DATA_H
#define ASCEND_OPS_RMS_NORM_AND_ROPE_AND_RESHAPE_AND_CACHE_TILING_DATA_H
 
#include <cstdint>
 
namespace AtbOps {
struct RmsNormAndRopeAndReshapeAndCacheTilingData {
    uint32_t numRow;
    uint32_t numCol;
    uint32_t numCore;
    float avgFactor;
    uint32_t precisionMode;
    float epsilon;
    uint32_t ropeNumTokens;
    uint32_t ropeHiddenSizeK;
    uint32_t ropeHeadDimK;
    uint32_t ropeHeadNumK;
    uint32_t racNumTokens;
    uint32_t slotMappingLen;
    uint32_t racNumHeads;
    uint32_t racHeadSizeK;
    uint32_t rotaryCoeff;
    uint32_t maxRow;
};
} // namespace AtbOps
 
#endif // ASCEND_OPS_RMS_NORM_AND_ROPE_AND_RESHAPE_AND_CACHE_TILING_DATA_H