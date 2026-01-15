/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATBOPS_PARAMS_STRIDEDBATCHMATMUL_H
#define ATBOPS_PARAMS_STRIDEDBATCHMATMUL_H

#include <cstdint>
#include <string>
#include <sstream>
#include <vector>
#include "mki/utils/compare/compare.h"

namespace AtbOps {
namespace OpParam {
struct StridedBatchMatmul {
    // UNPAD_GEN_ATTENTION_MASK
    int32_t headNum = 0;
    // UNPAD_STRIDEDBATCHMATMUL
    int32_t batch = 1;
    int32_t transA = 0;
    int32_t transB = 0;
    // UNPAD_STRIDEDBATCHMATMUL
    std::vector<int32_t> m;
    std::vector<int32_t> k;
    std::vector<int32_t> n;
    std::vector<int32_t> lda;
    std::vector<int32_t> ldb;
    std::vector<int32_t> ldc;
    std::vector<int32_t> strideA;
    std::vector<int32_t> strideB;
    std::vector<int32_t> strideC;
    bool operator==(const StridedBatchMatmul &other) const
    {
        return this->headNum == other.headNum && this->batch == other.batch && this->transA == other.transA &&
               this->transB == other.transB && this->m == other.m && this->k == other.k && this->n == other.n &&
               this->lda == other.lda && this->ldb == other.ldb && this->ldc == other.ldc &&
               this->strideA == other.strideA && this->strideB == other.strideB && this->strideC == other.strideC;
    }
};
} // namespace OpParam
} // namespace AtbOps

#endif // ATBOPS_PARAMS_STRIDEDBATCHMATMUL_H