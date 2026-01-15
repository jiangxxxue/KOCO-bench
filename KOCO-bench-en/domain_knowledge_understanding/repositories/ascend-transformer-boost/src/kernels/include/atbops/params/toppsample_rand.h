/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATBOPS_PARAMS_TOPPSAMPLE_RAND_H
#define ATBOPS_PARAMS_TOPPSAMPLE_RAND_H

#include <cstdint>
#include <string>
#include <sstream>
#include <vector>

namespace AtbOps {
namespace OpParam {
struct ToppsampleRand {
    // TOPP_SAMPLE_RAND
    bool operator==(const ToppsampleRand &other) const
    {
        (void)other;
        return true;
    }
};
} // namespace OpParam
} // namespace AtbOps

#endif // ATBOPS_PARAMS_TOPPSAMPLE_RAND_H