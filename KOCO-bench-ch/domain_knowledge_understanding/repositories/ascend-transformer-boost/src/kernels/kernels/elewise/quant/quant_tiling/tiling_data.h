/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASDOPS_QUANT_TILING_DATA
#define ASDOPS_QUANT_TILING_DATA

#include <cstdint>

namespace AsdOps {
struct QuantF16TilingData {
    uint32_t numCore{0};
    uint32_t numLastDim{0};
    uint32_t numFirstDim{0};
    uint32_t nlFirstdimPerCore{0};
    uint32_t lFirstdimPerCore{0};
    uint32_t firstDimPerTimes{0};
    uint32_t inputScale{0};
    uint32_t inputOffset{0};
    float quantMin{-128};
};
} // namespace AsdOps
#endif