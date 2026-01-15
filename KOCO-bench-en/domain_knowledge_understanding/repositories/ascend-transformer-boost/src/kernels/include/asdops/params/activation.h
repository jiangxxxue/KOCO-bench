/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDOPS_PARAMS_ACTIVATION_H
#define ASDOPS_PARAMS_ACTIVATION_H
#include <string>
#include <sstream>
#include <mki/utils/compare/compare.h>

namespace AsdOps {
namespace OpParam {
struct Activation {
    enum ActivationType {
        ACTIVATION_UNDEFINED = 0,
        ACTIVATION_RELU,
        ACTIVATION_GELU,
        ACTIVATION_FAST_GELU,
        ACTIVATION_SWISH,
        ACTIVATION_LOG,
        ACTIVATION_SWIGLU_FORWARD,
        ACTIVATION_SWIGLU_BACKWARD,
	    ACTIVATION_SIGMOID,
        ACTIVATION_FASTER_GELU_FORWARD,
    };
    ActivationType activationType;

    float scale = 1.0f;       // for Swish
    int32_t dim = -1;         // SWIGLU
    int32_t approx = 0;       // GeLU approx mode: 0:tanh; 1:none;

    bool operator==(const Activation &other) const
    {
        return this->activationType == other.activationType &&
               Mki::Utils::Compare<float>::IsEqual(this->scale, other.scale) && this->dim == other.dim &&
               this->approx == other.approx;
    }
};
} // namespace OpParam
} // namespace AsdOps

#endif
