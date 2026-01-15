/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LCAL_LCOC_BASE_H
#define LCAL_LCOC_BASE_H

#include <cstdint>

#pragma once
namespace Lcal {
enum QuantGranularity : int {
    QUANT_GRANULARITY_UNDEFINED = -1,
    PER_TENSOR = 0,
    PER_CHANNEL = 1,
    PER_GROUP = 2,
    PER_TOKEN = 3,
    FLOAT32_SCALE_PER_CHANNEL = 4,
    QUANT_GRANULARITY_MAX = 5,
};

struct MatMulInfo {
    int64_t batchSize = 1;
    int64_t m = -1;
    int64_t k = -1;
    int64_t n = -1;
    bool transA = false;
    bool transB = false;
    bool withBias = false;
    bool isInt8 = false;
    bool weightNz = false;
};

struct TwoDimTPInfo {          // 2D-TP，含x轴的通信和y轴通信
    int32_t agDim = -1;        // 表示ag轴卡数，规定x轴方向是非连续卡号
    int32_t rsDim = -1;        // 表示rs轴卡数，规定y轴方向是连续卡号
    bool innerDimIsAg = true;  // 是否沿着内轴进行allgather通信
};

struct QuantInfo {
    // 反量化（包括Matmul前置伪量化和后置反量化）粒度
    QuantGranularity dequantGranularity = QuantGranularity::QUANT_GRANULARITY_UNDEFINED;
    int32_t dequantGroupSize = -1;

    QuantGranularity quantGranularity = QuantGranularity::QUANT_GRANULARITY_UNDEFINED;  // 量化粒度
    int32_t quantGroupSize = -1;
};

struct PostInfo {
    int32_t withRmsNorm = 0;
};

struct MoeInfo {
    int16_t local_expert_nums = 0;
    int8_t EP = 0;
    int8_t TP = 0;
    int32_t maxOutputSize = -1;
    int8_t isMoe = 0;
};
}
#endif  // LCAL_LCOC_BASE_H
