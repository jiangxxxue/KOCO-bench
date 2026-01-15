/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_OPS_LASER_ATTENTION_GRAD_TILING_DATA_H
#define ASCEND_OPS_LASER_ATTENTION_GRAD_TILING_DATA_H

#include <cstdint>

namespace AtbOps {
struct LaserAttentionGradTilingData {
    int32_t batchSize;
    int32_t headNum;
    int32_t seqSize;
    int32_t headDim;
    int32_t blockDim;
    int32_t blockNumPerCore;

    int32_t isTriangle;      // 是否非倒三角
    int32_t attenType;       // 0:MHA/1:GQA
    int32_t sparseMode;      // 0:dense/1:sparse
    int32_t headGroupSize;   // N/G
    int32_t windowLen;       // V
    int32_t qSeqLength;      // qkv不等长
    int32_t kSeqLength;      // qkv不等长
    int32_t vSeqLength;      // qkv不等长
    int32_t isHighPrecision; // 高性能

    float scale;           // 预留
    float keepProb;        // 预留
    int32_t preTokens;     // 预留
    int32_t nextTokens;    // 预留
    int32_t maskSeqLength; // 预留

    int64_t queryShape;
    int64_t keyShape;
    int64_t valueShape;

    // tiling for post kernel
    int64_t postBaseNum;
    int64_t postDqFrontCoreNum;
    int64_t postDqTailCoreNum;
    int64_t postDqFrontDataNum;
    int64_t postDqTailDataNum;

    int64_t postDkFrontCoreNum;
    int64_t postDkTailCoreNum;
    int64_t postDkFrontDataNum;
    int64_t postDkTailDataNum;

    int64_t postDvFrontCoreNum;
    int64_t postDvTailCoreNum;
    int64_t postDvFrontDataNum;
    int64_t postDvTailDataNum;

    int64_t outputWorkspaceOffset;
    int32_t inputLayout;

    int32_t coreSeqOuterBlkIdx[24];
    int32_t coreSeqOuterBlkNum[24];
    int32_t coreBlkNum[24];
};
} // namespace AtbOps
#endif // ASCEND_OPS_LASER_ATTENTION_GRAD_TILING_DATA_H
