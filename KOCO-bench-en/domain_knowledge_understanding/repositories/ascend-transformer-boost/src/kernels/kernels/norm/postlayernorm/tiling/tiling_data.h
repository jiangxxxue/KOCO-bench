/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDOPS_POST_LAYER_NORM_TILING_DATA
#define ASDOPS_POST_LAYER_NORM_TILING_DATA

#include <cstdint>

namespace AsdOps {
struct PostLayerNormTilingData {
    uint32_t numCore{0};
    uint32_t numLastDim{0};
    uint32_t numFirstDim{0}; // 无用待删除
    uint32_t nlFirstdimPerCore{0};
    uint32_t lFirstdimPerCore{0};
    uint32_t firstDimPerTimes{0};
    float epsStr{0};
    float aveStr{0};
    uint32_t normMode{0};
    float zoomScale{0}; // PostLayerNormQuant该字段未使用

    uint32_t sliceNum{0}; // 支持泛化新增
    uint32_t sliceSize{0}; // 支持泛化新增
    uint32_t tailSliceSize{0}; // 支持泛化新增
};

struct KernelBufferInfo {
    uint32_t fp32BufNum{0};
    uint32_t fp16BufNum{0};
    uint32_t fp16BufNumForMulRow{0};
    uint32_t i8BufNumForMulRow{0};
};
}
#endif