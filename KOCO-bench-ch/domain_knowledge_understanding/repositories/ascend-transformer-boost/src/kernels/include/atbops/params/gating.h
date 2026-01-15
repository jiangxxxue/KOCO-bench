/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATBOPS_PARAMS_GATING_H
#define ATBOPS_PARAMS_GATING_H

#include <cstdint>
#include <string>
#include <sstream>
#include <mki/utils/SVector/SVector.h>

namespace AtbOps {
namespace OpParam {
struct Gating {
    int32_t headSize = 0;
    int32_t headNum = 0;
    bool cumSumInt64 = false;
    std::vector<int32_t> deviceExpert;

    bool operator==(const Gating &other) const
    {
        return this->headSize == other.headSize && this->headNum == other.headNum &&
               this->deviceExpert == other.deviceExpert && this->cumSumInt64 == other.cumSumInt64;
    }
};

} // namespace OpParam
} // namespace AtbOps

#endif // ATBOPS_PARAMS_GATING_H