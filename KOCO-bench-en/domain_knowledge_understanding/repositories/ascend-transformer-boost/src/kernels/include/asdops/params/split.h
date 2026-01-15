/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDOPS_PARAMS_SPLIT_H
#define ASDOPS_PARAMS_SPLIT_H

#include <cstdint>
#include <mki/utils/SVector/SVector.h>

namespace AsdOps {
namespace OpParam {
struct Split {
    int splitDim = 0;
    int splitNum = 2;
    Mki::SVector<int32_t> splitVDim = {0};
    Mki::SVector<int32_t> splitSize = {};

    bool operator==(const Split &other) const
    {
        return this->splitDim == other.splitDim && this->splitNum == other.splitNum &&
            this->splitVDim == other.splitVDim && this->splitSize == other.splitSize;
    }
};
} // namespace OpParam
} // namespace AsdOps

#endif