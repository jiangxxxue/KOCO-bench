/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDOPS_PARAMS_GROUPTOPK_H
#define ASDOPS_PARAMS_GROUPTOPK_H

#include <cstdint>

namespace AsdOps {
namespace OpParam {
struct GroupTopk {
    int32_t groupNum = 0;
    int32_t k = 0;
    int32_t kInner = 1;

    bool operator==(const GroupTopk &other) const
    {
        return (this->groupNum == other.groupNum) && (this->k == other.k) && (this->kInner == other.kInner);
    };
};
} // namespace OpParam
} // namespace AsdOps

#endif // ASDOPS_PARAMS_GROUPTOPK_H
