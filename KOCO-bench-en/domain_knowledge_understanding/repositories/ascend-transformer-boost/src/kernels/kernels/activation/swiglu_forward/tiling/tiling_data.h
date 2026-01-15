/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_OPS_SWIGLU_TILING_DATA_H
#define ASCEND_OPS_SWIGLU_TILING_DATA_H

#include <cstdint>

namespace AsdOps {
struct SwiGluForwardTilingData {
    uint32_t rowLen; // 行数
    uint32_t colLen; // 列数, 输入x的一半
    uint32_t rowLenPerCore; // 每核处理行数
    uint32_t colLenPerCore; // 每核处理列数
    uint32_t coreNumPerLine; // 用于一行多核处理情况，处理每行的核数
    uint16_t basicRowLen; // 每次计算的行数
    uint16_t basicColLen; // 每次计算的列数，是32B对齐
};
}
#endif // ASCEND_OPS_SWIGLU_TILING_DATA_H