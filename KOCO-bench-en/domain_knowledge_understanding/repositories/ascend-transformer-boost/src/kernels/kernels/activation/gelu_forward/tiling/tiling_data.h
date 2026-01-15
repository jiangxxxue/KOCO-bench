/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GELU_FORWARD_TILING_H
#define GELU_FORWARD_TILING_H

#include <cstdint>

namespace AsdOps {
constexpr uint32_t GELU_FORWARD_BUFF_NUM = 2;
struct GeluForwardTilingData {
    uint32_t blockLength{0};        // 每个核上分配的数据量
    uint32_t tileNum{0};            // 搬运次数
    uint32_t tileLength{0};         // 最大搬运长度，用于初始化buffer
    uint32_t tailLength{0};         // 尾块长度
    uint32_t bufferNum{2};          // doublebuffer
};
} // namespace AsdOps
#endif // GELU_FORWARD_TILING_H