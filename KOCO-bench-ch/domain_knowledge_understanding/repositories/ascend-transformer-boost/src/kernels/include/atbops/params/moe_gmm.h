/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATBOPS_PARAMS_MOE_GMM_H
#define ATBOPS_PARAMS_MOE_GMM_H

#include <cstdint>
#include <mki/utils/SVector/SVector.h>

namespace AtbOps {
namespace OpParam {
struct MoeGmm {
    enum MoeGmmMode : int32_t {
        MOE_GMM_UNDEFINED = -1,
        MOE_GMM_UP = 0,
        MOE_GMM_DOWN
    };
    enum MoeGmmDequantType : int32_t {
        NO_DEQUANT = -1,
        DEQ_FP16 = 0,
        DEQ_BF16
    };

    bool transposeB = false;
    uint32_t topK = 0;
    MoeGmmMode moeGmmMode = MOE_GMM_UNDEFINED;
    MoeGmmDequantType moeGmmDequantType = NO_DEQUANT;
    Mki::SVector<int64_t> hiddenSize = {0, 0}; // hiddenSizeIn, hiddenSizeOut

    bool operator==(const MoeGmm &other) const
    {
        return (transposeB == other.transposeB) && (topK == other.topK) && (moeGmmMode == other.moeGmmMode) &&
               (moeGmmDequantType == other.moeGmmDequantType);
    }
};

} // namespace OpParam
} // namespace AtbOps

#endif // ATBOPS_PARAMS_MOE_GMM_H