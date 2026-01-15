/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PP_MATMUL_ND_TILING_H
#define PP_MATMUL_ND_TILING_H

#include <mki/launch_param.h>
#include <mki/kernel_info.h>
#include <mki/utils/status/status.h>

namespace AsdOps {
using namespace Mki;
constexpr uint32_t CONST_16 = 16;
constexpr uint32_t CONST_128 = 128;
constexpr uint32_t CONST_256 = 256;
constexpr uint32_t L0C_SIZE = 128 * 1024;

template <typename T = uint32_t> inline T Max(const T a, const T b) { return a > b ? a : b; }

template <typename T = uint32_t> inline T Min(const T a, const T b) { return a < b ? a : b; }

struct MatMulInfoNd {
    uint32_t batchSize{0};
    uint32_t m{0};                  // 实际输入的 m
    uint32_t n{0};                  // 实际输入的 n
    uint32_t k{0};                  // 实际输入的 k
    bool transA{0};                 // false: 0, true: 1
    bool transB{0};                 // false: 0, true: 1
    bool biasFlag{0};               // false: 0, true: 1
    bool isInt8{0};                 // 是否 int8融合
    float inDtype{2};
    float outDtype{4};
    uint32_t formatSema{0};         // "FRACTAL_NZ": 0, "ND": 1
};

struct OpShapeNd {
    uint32_t batchSize{0};
    uint32_t m{0};
    uint32_t k{0};
    uint32_t n{0};
    uint32_t m0{0};
    uint32_t k0{0};
    uint32_t n0{0};
};

struct PpTilingDataNd {
    uint32_t batchSize{0};
    uint32_t m{0};
    uint32_t k{0};
    uint32_t n{0};
    uint32_t m0{0};
    uint32_t k0{0};
    uint32_t n0{0};
    uint32_t mLoop{1};
    uint32_t kLoop{1};
    uint32_t nLoop{1};
    uint32_t coreLoop{1};
    uint32_t tilingKey{0};

    void SetTilingKey(bool transB);
};

Status PpTilingNd(const LaunchParam &launchParam, KernelInfo &kernelInfo);
}
#endif // PP_MATMUL_ND_TILING_H
