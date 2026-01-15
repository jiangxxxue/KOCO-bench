/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDOPS_PARAMS_CUMSUM_H
#define ASDOPS_PARAMS_CUMSUM_H

#include <cstdint>
#include <string>
#include <sstream>
#include <mki/utils/SVector/SVector.h>

namespace AsdOps {
namespace OpParam {
struct Cumsum {
    Mki::SVector<int64_t> axis;
    bool exclusive = false;
    bool reverse = false;
    bool deterministic = false;

    bool operator==(const Cumsum &other) const
    {
        return (this->axis == other.axis) && (this->exclusive == other.exclusive) && (this->reverse == other.reverse);
    };
};
} // namespace OpParam
} // namespace AsdOps

#endif // ASDOPS_PARAMS_CUMSUM_H