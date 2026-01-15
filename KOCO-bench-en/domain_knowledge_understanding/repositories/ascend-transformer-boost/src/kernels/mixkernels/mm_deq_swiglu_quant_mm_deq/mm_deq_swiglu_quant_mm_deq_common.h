/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_MM_DEQ_SWIGLU_QUANT_MM_DEQ_H
#define ASCEND_OPS_MM_DEQ_SWIGLU_QUANT_MM_DEQ_H

constexpr int64_t MAX_M = 131072;
constexpr int64_t SUPPORTED_K1 = 7168;
constexpr int64_t SUPPORTED_N1 = 4096;
constexpr int64_t SUPPORTED_K2 = 2048;
constexpr int64_t SUPPORTED_N2 = 7168;

constexpr size_t INPUT_NUM = 6;
constexpr size_t OUTPUT_NUM = 1;

// 该算子有前后两个反量化 MM 算子，使用 1 和 2 分别标识
struct InTensorIndex {
    static constexpr size_t X1 = 0;
    static constexpr size_t WEIGHT1 = 1;
    static constexpr size_t SCALE1 = 2;
    static constexpr size_t PER_TOKEN_SCALE1 = 3;
    static constexpr size_t WEIGHT2 = 4;
    static constexpr size_t SCALE2 = 5;
};

constexpr size_t X_DIMS = 2;
struct XDimIndex {
    static constexpr size_t M = 0;
    static constexpr size_t K = 1;
};

constexpr size_t WEIGHT_DIMS = 2;
template <bool IsTrans = false>
struct WeightDimIndex {
    static constexpr size_t K = IsTrans ? 1 : 0;
    static constexpr size_t N = IsTrans ? 0 : 1;
};

constexpr size_t SCALE_DIMS = 1;
struct ScaleDimIndex {
    static constexpr size_t N = 0;
};

constexpr size_t PER_TOKEN_SCALE_DIMS = 1;
struct PerTokenScaleDimIndex {
    static constexpr size_t M = 0;
};

#endif