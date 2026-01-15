/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_NORM_DYNAMIC_QUANT_TILING_DATA_H
#define ASCEND_OPS_NORM_DYNAMIC_QUANT_TILING_DATA_H

#include <cstdint>

namespace AsdOps {
struct NormDynamicQuantTilingData {
    uint32_t numCore{1};
    uint32_t numCol{1};
    uint32_t numRow{1};
    float avgFactor{1};
    float epsilon{0};
    float quantMin{-128};
    uint32_t numRowPerCore{1};
};

} // namespace AsdOps
#endif