/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_OPS_LASER_ATTENTION_TILING_DATA_H
#define ASCEND_OPS_LASER_ATTENTION_TILING_DATA_H

#include <cstdint>

namespace AtbOps {
struct LaserAttentionTilingData {
    int32_t batchSize;       // B
    int32_t headNum;         // N
    int32_t seqSize;         // S
    int32_t headDim;         // D
    int32_t coreNumPerGroup; // Y
    int32_t coreGroupNum;    // F
    int32_t qSeqLength;      // qkv不等长
    int32_t kSeqLength;      // qkv不等长
    int32_t vSeqLength;      // qkv不等长
    int32_t maskSeqLength;   // 预留
    float scale;             // 预留
    float keepProb;          // 预留
    int32_t preTokens;       // 预留
    int32_t nextTokens;      // 预留
    int32_t attenType;       // 0:MHA/1:GQA
    int32_t sparseMode;      // 0:dense/1:sparse
    int32_t headGroupSize;   // N/G
    int32_t windowLen;       // sparse的滑动窗口
    int32_t isTriangle;      // 是否倒三角
    int32_t isHighPrecision; // 高性能
    int32_t inputLayout;     // transpose
};
} // namespace AtbOps
#endif // ASCEND_OPS_LASER_ATTENTION_TILING_DATA_H
