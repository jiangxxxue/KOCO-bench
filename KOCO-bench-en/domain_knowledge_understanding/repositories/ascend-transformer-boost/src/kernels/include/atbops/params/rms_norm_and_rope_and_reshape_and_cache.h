/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATBOPS_PARAMS_RMS_NORM_AND_ROPE_AND_RESHAPE_AND_CACHE_H
#define ATBOPS_PARAMS_RMS_NORM_AND_ROPE_AND_RESHAPE_AND_CACHE_H
 
#include <cstdint>
namespace AtbOps {
namespace OpParam {
struct RmsNormAndRopeAndReshapeAndCache {
    uint32_t precisionMode = 0;
    float epsilon = 1e-5f;
    uint32_t rotaryCoeff = 2;
 
    bool operator==(const RmsNormAndRopeAndReshapeAndCache &other) const
    {
        return (this->precisionMode == other.precisionMode) &&
               Mki::Utils::Compare<float>::IsEqual(this->epsilon, other.epsilon) &&
               (this->rotaryCoeff == other.rotaryCoeff);
    };
};
} // namespace OpParam
} // namespace AtbOps
 
#endif // ATBOPS_PARAMS_RMS_NORM_AND_ROPE_AND_RESHAPE_AND_CACHE_H