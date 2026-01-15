/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ASCEND_OPS_FFN_COMMON_H
#define ASCEND_OPS_FFN_COMMON_H

#include "kernel_operator.h"
#include "lib/matrix/matmul/matmul.h"
#include "lib/matmul_intf.h"
#include "kernels/utils/kernel/hardware.h"

namespace FFN {
constexpr uint32_t BLOCK_SIZE_16 = 16;
constexpr uint32_t BLOCK_SIZE_32 = 32;
constexpr uint32_t CONST_1 = 1;
constexpr uint32_t CONST_2 = 2;
constexpr uint32_t CONST_3 = 3;
constexpr uint32_t CONST_4 = 4;
constexpr uint32_t CONST_8 = 8;
constexpr uint32_t CONST_16 = 16;
constexpr uint32_t CONST_32 = 32;
constexpr uint32_t CONST_64 = 64;
constexpr uint32_t CONST_128 = 128;
constexpr uint32_t CONST_256 = 256;
constexpr uint32_t BLOCK_NUM_PER_FRACTAL = 16;
constexpr uint32_t BLOCK_NUM_PER_VEC = 8;
constexpr uint32_t FRACTAL_SIZE = 512;
constexpr uint32_t MAX_REPEAT_LIMIT = 255;

enum class ActivationType {
    GELU = 0,
    FASTGELU,
    FASTGELUV2,
    INVALID_TYPE
};

template <ArchType ARCH_TAG, typename DTYPE_A, CubeFormat FORMAT_A, typename DTYPE_B, CubeFormat FORMAT_B,
    typename DTYPE_C, CubeFormat FORMAT_C, typename DTYPE_ACCUMULATOR, typename DTYPE_BIAS, typename DTYPE_SCALE,
    bool TRANS_A, bool TRANS_B, bool WITH_BIAS, bool EN_L0C_DB
>
struct DefaultMatmul;

template <
    ArchType ARCH_TAG,
    typename IN_DTYPE,
    typename DESCALE_DTYPE,
    typename BIAS_DTYPE,
    typename ACCUMULATOR_DTYPE,
    typename OUT_DTYPE,
    CubeFormat IN_FORMAT,
    CubeFormat OUT_FORMAT,
    bool TRANSPOSE_X,
    bool TRANSPOSE_W1,
    bool TRANSPOSE_W2,
    bool HIGH_PRECISION,
    bool HIGH_PERFORMANCE
>
struct DefaultFFN;
} // namespace FFN

#endif