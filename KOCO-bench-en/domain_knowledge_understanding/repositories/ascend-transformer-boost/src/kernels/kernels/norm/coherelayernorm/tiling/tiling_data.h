/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDOPS_COHERE_LAYER_NORM_TILING_DATA
#define ASDOPS_COHERE_LAYER_NORM_TILING_DATA

#include <cstdint>

namespace AsdOps {
struct CohereLayerNormTilingData {
    uint32_t numCore{0};
    uint32_t numCoreRows{0};
    uint32_t coreRowStrides{0};
    uint32_t coreRowRepeats{0};
    uint32_t coreRowTailStrides{0};
    uint32_t coreRowTailRepeats{0};
    uint32_t residualCoreRowStrides{0};
    uint32_t residualCoreRowRepeats{0};
    uint32_t residualCoreRowTailStrides{0};
    uint32_t residualCoreRowTailRepeats{0};
    uint32_t numColumns{0};
    uint32_t columnStrides{0};
    uint32_t columnRepeats{0};
    uint32_t residualColumnStrides{0};
    uint32_t residualColumnRepeats{0};
    uint32_t numHeads{0};
    float epsilon{0.0};
    float averageFactor{0.0};
};

struct KernelBufferInfoCohereLayerNorm {
    uint32_t fp32BufNum{0};
    uint32_t fp16BufNum{0};
    uint32_t fp16BufNumForMulRow{0};
};

} // namespace AsdOps
#endif