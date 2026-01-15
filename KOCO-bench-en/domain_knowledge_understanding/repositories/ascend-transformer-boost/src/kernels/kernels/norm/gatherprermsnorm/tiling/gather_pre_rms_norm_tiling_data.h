/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_GATHER_RMS_PRE_NORM_TILING_DATA_H
#define ASCEND_OPS_GATHER_RMS_PRE_NORM_TILING_DATA_H

#include <cstdint>

namespace AsdOps {
constexpr uint32_t BRCB_BYTE = 256;
constexpr uint32_t INT32_ALIGN_NUM = 8;

struct GatherPreRmsNormTilingData {
    uint32_t numCore{1};
    uint32_t numCol{1};
    uint32_t numRow{1};
    float avgFactor{1};
    float epsilon{0.1};
    uint32_t ubResByte{0};
    uint32_t numRowPerCore{0};
    uint32_t numRowPerCoreAlign{0};
    uint32_t numRowTailCore{0};
};
} // namespace AsdOps
#endif