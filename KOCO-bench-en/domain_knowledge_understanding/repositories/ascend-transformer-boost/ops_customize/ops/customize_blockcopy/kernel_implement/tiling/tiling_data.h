/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CUSTOMIZE_OPS_BLOCKCOPY_TILING_DATA
#define CUSTOMIZE_OPS_BLOCKCOPY_TILING_DATA

#include <cstdint>
constexpr uint32_t TILING_DTYPE_IDX = 100000000;
namespace AtbOps {
struct CustomizeBlockCopyTilingData {
    uint32_t blockCount;
    uint32_t blockSize;
    uint32_t numHead;
    uint32_t headSizeK;
    uint32_t headSizeV;
    uint32_t sourceCount;
    uint32_t destinationCount;
    uint32_t typeByte;
    uint32_t blockDim;
    uint32_t perCoreCopyCount;
    uint32_t tailCoreCopyCount;
};
} // namespace AtbOps

#endif // ASCEND_OPS_GATING_TILING_DATA