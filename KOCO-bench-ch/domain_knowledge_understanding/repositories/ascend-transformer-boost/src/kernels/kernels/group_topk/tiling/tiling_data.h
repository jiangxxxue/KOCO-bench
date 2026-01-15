/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_OPS_GROUPTOPK_TILING_DATA
#define ASCEND_OPS_GROUPTOPK_TILING_DATA

#include <cstdint>

namespace AsdOps {
struct GroupTopkTilingData {
    uint32_t groupNum{0};                // 分组数量
    uint32_t k{0};                       // 选k个组
    uint32_t kInner{0};                  // 组内选取kInner个最大值求和
    uint32_t expertNumPerGroup{0};       // 每个组的专家数量
    uint32_t expertNumPerGroupPadded{0}; // group填充至32个对齐后的专家数量
    uint32_t expertNumPerToken{0};       // 每个token的专家总数量
    uint32_t tokenNumPerCore{0};         // 平均每个核处理的token数量，不含尾块
    uint32_t tailTokenNum{0};            // 尾块token数量，从0核开始逐一分配
};
} // namespace AsdOps

#endif // ASCEND_OPS_GROUPTOPK_TILING_DATA
