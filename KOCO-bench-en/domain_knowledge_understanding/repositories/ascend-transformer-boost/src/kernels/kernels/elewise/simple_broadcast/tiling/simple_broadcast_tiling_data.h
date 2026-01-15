/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_SIMPLE_BROADCAST_TILING_DATA_H
#define ASCEND_OPS_SIMPLE_BROADCAST_TILING_DATA_H

#include <cstdint>

namespace AsdOps {
struct BroadcastTilingData {
    int64_t dimB{1}; // b维度大小
    int64_t dimN{1}; // n维度大小
    int64_t dimBBlockNum{1}; // b维度分块个数
    int64_t dimNBlockNum{1}; // n维度分块个数
    int64_t dimBBlockSize{1}; // b维度分块元素个数
    int64_t dimNBlockSize{1}; // n维度分块元素个数
    int64_t dimBLoop{1}; // 单次循环操作的b维度元素个数
    int64_t dimNLoop{1}; // 单次循环操作的n维度元素个数
};
} // namespace AsdOps
#endif