/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASDOPS_COMMON_TILING_DATA
#define ASDOPS_COMMON_TILING_DATA

#include <cstdint>

namespace {
constexpr uint32_t TENSOR_MAX_DIM_NUM = 8;
}

namespace AsdOps {
struct RmsNormCommonTilingData {
    uint32_t numCore{1};
    uint32_t numCol{1};
    uint32_t numRow{1};
    float avgFactor{1};
    float epsilon{0};
    uint32_t sliceSize{1};
    uint32_t mode{0}; // 后续准备退出
    float quantMin{-128};
    uint32_t precisionMode{0};
    uint32_t gemmaMode{0};
    uint32_t xStrides[TENSOR_MAX_DIM_NUM];
    uint32_t xOffset{0};
    uint32_t xDimNum{0};
};
struct RmsNormQuantCommonTilingData {
    uint32_t numCore{1};
    uint32_t numCol{1};
    uint32_t numRow{1};
    uint32_t avgFactor{1};
    uint32_t epsilon{0};
    float quantMin{-128};
    uint32_t sliceSize{1};
    uint32_t sliceNum{1};
    uint32_t tailSliceSize{1};
    uint32_t maxCoreNum{0};
};
} // namespace AsdOps
#endif