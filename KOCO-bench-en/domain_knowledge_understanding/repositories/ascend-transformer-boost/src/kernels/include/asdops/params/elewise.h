/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDOPS_PARAMS_ELEWISE_H
#define ASDOPS_PARAMS_ELEWISE_H

#include <mki/types.h>
#include <mki/utils/compare/compare.h>

namespace AsdOps {
namespace OpParam {
struct Elewise {
    enum ElewiseType {
        ELEWISE_UNDEFINED = 0,
        ELEWISE_CAST,
        ELEWISE_MULS,
        ELEWISE_COS,
        ELEWISE_SIN,
        ELEWISE_NEG,
        ELEWISE_QUANT,
        ELEWISE_LOGICAL_NOT,
        ELEWISE_ADD,
        ELEWISE_MUL,
        ELEWISE_REALDIV,
        ELEWISE_LOGICAL_AND,
        ELEWISE_LOGICAL_OR,
        ELEWISE_LESS,
        ELEWISE_GREATER,
        ELEWISE_SUB,
        ELEWISE_TANH,
        ELEWISE_EQUAL,
        ELEWISE_QUANT_PER_CHANNEL,
        ELEWISE_DEQUANT_PER_CHANNEL,
        ELEWISE_DYNAMIC_QUANT,
    };
    ElewiseType elewiseType;

    float varAttr = 0.0f;    // MULS
    float inputScale = 1.0f; // QUANT
    int inputOffset = 0;     // QUANT
    bool asymmetric = false; // DynamicQuant false : symmetric，true : asymmetric
    Mki::TensorDType outTensorType = Mki::TENSOR_DTYPE_UNDEFINED;

    bool operator==(const Elewise &other) const
    {
        return this->elewiseType == other.elewiseType &&
               Mki::Utils::Compare<float>::IsEqual(this->varAttr, other.varAttr) &&
               Mki::Utils::Compare<float>::IsEqual(this->inputScale, other.inputScale) &&
               this->inputOffset == other.inputOffset &&
               this->asymmetric == other.asymmetric &&
               this->outTensorType == other.outTensorType;
    }
};
} // namespace OpParam
} // namespace AsdOps

#endif
