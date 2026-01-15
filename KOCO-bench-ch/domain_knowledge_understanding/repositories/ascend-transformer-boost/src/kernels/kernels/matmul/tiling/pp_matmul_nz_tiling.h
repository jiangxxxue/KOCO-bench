/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_OPS_MATMUL_PP_310P_TILING_H
#define ASCEND_OPS_MATMUL_PP_310P_TILING_H

#include <mki/launch_param.h>
#include <mki/kernel_info.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/status/status.h>

namespace AsdOps {
using namespace Mki;

constexpr uint32_t PP_MATMUL_TILING_SIZE = 512;
constexpr uint32_t NUM_TILING_FIELD = 9;
struct MatMulInfoNz {
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

struct HardwareInfoNz {
    uint32_t coreNum{0};
    uint32_t l2Size{0};
    uint32_t l1Size{0};
    uint32_t l0aSize{0};
    uint32_t l0bSize{0};
    uint32_t l0cSize{0};
    uint32_t hbmBandWidth{0};
    uint32_t l2BandWidth{0};
    explicit HardwareInfoNz(PlatformInfo &platInfo);
};

struct OpShapeNz {
    uint32_t batchSize{0};
    uint32_t m{0};
    uint32_t k{0};
    uint32_t n{0};
    uint32_t m0{0};
    uint32_t k0{0};
    uint32_t n0{0};
};

struct PpTilingDataNz {
    OpShapeNz opShape{};
    uint32_t mLoop{1};
    uint32_t kLoop{1};
    uint32_t nLoop{1};
    uint32_t coreLoop{1};
    uint32_t swizzlCount{1};
    uint32_t tilingKey{0};
    uint32_t blockDim{1};
    uint32_t swizzlDirect{0};
    uint32_t splitk{0};

    uint8_t reserved[PP_MATMUL_TILING_SIZE - sizeof(OpShapeNz) - sizeof(uint32_t) * NUM_TILING_FIELD];

    void SetBaseShape(uint32_t batchSize, uint32_t m, uint32_t k, uint32_t n);
    void SetBaseOp(uint32_t coreNum, uint32_t mBase, uint32_t nBase);
    void SetTilingKey(const MatMulInfoNz &mmInfo, uint32_t isSwizlDirect, uint32_t isSplitk);
    uint32_t End();
};

Status PpTilingNz(const LaunchParam &launchParam, KernelInfo &kernelInfo);
} // namespace AsdOps

#endif
