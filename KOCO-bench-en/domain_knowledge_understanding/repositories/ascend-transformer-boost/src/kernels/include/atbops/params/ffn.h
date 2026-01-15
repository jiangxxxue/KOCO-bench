/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATBOPS_PARAMS_FFN_H
#define ATBOPS_PARAMS_FFN_H

namespace AtbOps {
namespace OpParam {
struct FFN {
    bool transX;
    bool transW1;
    bool transW2;

    enum ActivationType {
        GELU = 0,
        FASTGELU,
        FASTGELUV2,
        INVALID_ACTIVATION_TYPE
    };
    ActivationType activationType = FASTGELUV2;

    enum ImplMode {
        HIGH_PRECISION = 0,
        HIGH_PERFORMANCE,
        INVALID_IMPL_MODE
    };
    ImplMode implMode = HIGH_PRECISION;

    bool operator==(const FFN &other) const
    {
        return this->transX == other.transX && this->transW1 == other.transW1 && this->transW2 == other.transW2 &&
               this->activationType == other.activationType && this->implMode == other.implMode;
    }
};

} // namespace OpParam
} // namespace AtbOps

#endif // ATBOPS_PARAMS_FFN_H