/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDOPS_PARAMS_FUSION_H
#define ASDOPS_PARAMS_FUSION_H

#include <mki/types.h>
#include <mki/utils/compare/compare.h>

namespace AtbOps {
namespace OpParam {
struct Fusion {
    enum FusionType : int {
        NON_FUSION = 0,
        MATMUL_ADD = 1,
        MATMUL_GELU = 2,
        MATMUL_SIGMOID = 3,
        MATMUL_SWIGLU = 4,
    };
    FusionType fusionType = FusionType::NON_FUSION;
    Mki::TensorDType outTensorType = Mki::TENSOR_DTYPE_UNDEFINED;

    bool operator==(const Fusion &other) const
    {
        return this->fusionType == other.fusionType &&
               this->outTensorType == other.outTensorType;
    }
};
} // namespace OpParam
} // namespace AtbOps

#endif
