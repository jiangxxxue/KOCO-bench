/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDOPS_PARAMS_SCATTER_ELEMENTS_V2_H
#define ASDOPS_PARAMS_SCATTER_ELEMENTS_V2_H

#include <string>
#include <sstream>
#include <mki/utils/SVector/SVector.h>

namespace AsdOps {
namespace OpParam {

/**
 * @brief ScatterElementsV2 算子的参数结构体
 */
struct ScatterElementsV2 {
    enum ReductionType {
        NONE = 0,
        ADD,
    };
    ReductionType reduction; // 指定更新的方式（none,add）
    int32_t axis; // 指定更新的轴

    /**
     * @brief 重载 == 运算符，用于比较两个 Scatter 参数是否相等
     */
    bool operator==(const ScatterElementsV2 &other) const
    {
        return this->axis == other.axis && this->reduction == other.reduction;
    }
};

} // namespace OpParam
} // namespace AsdOps

#endif // ASDOPS_PARAMS_SCATTER_ELEMENTS_H