/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDOPS_LAYER_NORM_STRIDE_TILING_DATA
#define ASDOPS_LAYER_NORM_STRIDE_TILING_DATA

#include <cstdint>

constexpr uint32_t MAX_DIM_NUM = 8;

namespace AsdOps {
struct LayerNormStrideTilingData {
    uint32_t numCore{0};
    uint32_t numLastDim{0};
    uint32_t numFirstDim{0};
    uint32_t nlFirstdimPerCore{0};
    uint32_t lFirstdimPerCore{0};
    uint32_t firstDimPerTimes{0};
    uint32_t epsStr{0};
    float aveStr{0};

    uint32_t normMode{0};
    float zoomScale{0};
    uint32_t sliceNum{0};
    uint32_t sliceSize{0};
    uint32_t tailSliceSize{0};
    uint32_t xStrides[MAX_DIM_NUM];
    uint32_t xOffset{0};
    uint32_t xDimNum{0};
};

struct KernelBufferInfoLayerNormStride {
    uint32_t fp32BufNum{0};
    uint32_t fp16BufNum{0};
    uint32_t fp16BufNumForMulRow{0};
};

} // namespace AsdOps
#endif