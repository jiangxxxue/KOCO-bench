/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATBOPS_PARAMS_LASER_ATTENTION_H
#define ATBOPS_PARAMS_LASER_ATTENTION_H

#include <string>
#include <sstream>
#include "mki/tensor.h"
#include "mki/utils/compare/compare.h"

namespace AtbOps {
namespace OpParam {
struct LaserAttention {
    int headNum = 0;
    std::string inputLayout = "BNSD";
    float scaleValue = 0.088;
    float keepProb = 1.0f;
    int preTokens = 2147483647;
    int nextTokens = 1;
    int sparseMode = 0;
    int innerPrecise = 1;
    bool operator==(const LaserAttention &other) const
    {
        return this->headNum == other.headNum && this->inputLayout == other.inputLayout &&
               Mki::Utils::Compare<float>::IsEqual(this->scaleValue, other.scaleValue) &&
               Mki::Utils::Compare<float>::IsEqual(this->keepProb, other.keepProb) &&
               this->preTokens == other.preTokens && this->nextTokens == other.nextTokens &&
               this->sparseMode == other.sparseMode && this->innerPrecise == other.innerPrecise;
    }
};
} // namespace OpParam
} // namespace AtbOps

#endif // ATBOPS_PARAMS_LASER_ATTENTION_H
