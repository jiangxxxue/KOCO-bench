/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDOPS_PARAMS_MATMUL_H
#define ASDOPS_PARAMS_MATMUL_H

#include <mki/types.h>
#include <mki/utils/SVector/SVector.h>
#include <sstream>
#include <string>

namespace AsdOps {
namespace OpParam {
struct MatMul {
    enum class MatMulType : uint32_t {
        MATMUL_DEFAULT = 0,  // C = op(A) * op(B)
        MATMUL_DEQUANT,      //
        MATMUL_ACCUM_ATOMIC, // C += op(A) * op(B)
        MATMUL_WITH_BIAS,    // C = op(A) * op(B) + Bias, where Bias is a vector.
        MATMUL_EIN_SUM
    };
    enum class QuantMode : uint32_t {
        PER_CHANNEL_SYMM = 0,
        PER_CHANNEL_ASYMM,
        PER_TOKEN_SYMM
    };
    bool transposeA = false;
    bool transposeB = false;
    Mki::SVector<int64_t> oriShape = {0, 0, 0}; // original shape: m,k,n - (m,k) * (k,n)
    bool withBias = false;
    bool enDequant = false;
    uint32_t tilingN = 0;    // 压缩算法透传参数, 单压缩块 n 方向的基块数
    uint32_t tilingK = 0;    // 压缩算法透传参数, 单压缩块 k 方向的基块数
    bool enShuffleK = false; // Shuffle-K使能，默认关。
    Mki::TensorDType outDtype = Mki::TENSOR_DTYPE_FLOAT16; // 只有量化能用， 可选FLOAT16：1  BFLOAT16:27
    QuantMode quantMode = QuantMode::PER_CHANNEL_SYMM; // 仅量化使用, 量化模式
    MatMulType matmulType = MatMulType::MATMUL_DEFAULT;
    bool operator==(const MatMul &other) const
    {
        return this->transposeA == other.transposeA && this->transposeB == other.transposeB &&
               this->oriShape == other.oriShape && this->withBias == other.withBias &&
               this->enDequant == other.enDequant && this->tilingN == other.tilingN && this->tilingK == other.tilingK &&
               this->outDtype == other.outDtype && this->matmulType == other.matmulType &&
               this->quantMode == other.quantMode;
    }
};
} // namespace OpParam
} // namespace AsdOps

#endif // ASDOPS_PARAMS_MATMUL_H