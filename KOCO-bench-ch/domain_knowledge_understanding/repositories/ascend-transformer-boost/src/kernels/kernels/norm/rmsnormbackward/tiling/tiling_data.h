/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_OPS_RMSNORMBACKWARD_TILING_DATA_H
#define ASCEND_OPS_RMSNORMBACKWARD_TILING_DATA_H

#include <cstdint>

namespace AsdOps {

struct RmsNormGradTilingData {
    uint32_t row;
    uint32_t col;
    float avg;
    uint32_t dataType;
    uint32_t blockFactor;
    uint32_t ubSplitDim;
    uint32_t ubFactor;
    uint32_t coreCalcNum;
    uint32_t coreCalcTail;
    uint32_t blockDim;
    uint32_t ubCalcNum;
    uint32_t ubCalcTail;
    uint32_t ubCalcLoop;
    uint32_t ubCalcTailNum;
    uint32_t ubCalcTailTail;
    uint32_t ubCalcTailLoop;
};

} // namespace AsdOps

#endif