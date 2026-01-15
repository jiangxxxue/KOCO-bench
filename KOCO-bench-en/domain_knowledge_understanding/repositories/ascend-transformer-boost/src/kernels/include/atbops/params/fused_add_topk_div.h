/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATBOPS_PARAMS_FUSED_ADD_TOPK_DIV_H
#define ATBOPS_PARAMS_FUSED_ADD_TOPK_DIV_H
 
#include <cstdint>
#include <string>
#include <sstream>
#include <mki/utils/SVector/SVector.h>
 
namespace AtbOps {
namespace OpParam {
struct FusedAddTopkDiv {
    uint32_t groupNum = 1;
    uint32_t groupTopk = 1;
    uint32_t n = 1;
    uint32_t k = 1;
    uint32_t activateType = 0;
    bool isNorm = true;
    float scale = 1.0;
    bool enableExpertMapping = false;
 
    bool operator==(const FusedAddTopkDiv &other) const
    {
        return this->groupNum == other.groupNum && this->groupTopk == other.groupTopk &&
               this->n == other.n && this->k == other.k && this->activateType == other.activateType &&
               this->isNorm == other.isNorm && Mki::Utils::Compare<float>::IsEqual(this->scale, other.scale) &&
               this->enableExpertMapping == other.enableExpertMapping;
    }
};
 
} // namespace OpParam
} // namespace AtbOps
 
#endif // ATBOPS_PARAMS_FUSED_ADD_TOPK_DIV_H
