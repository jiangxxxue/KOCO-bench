/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_RMS_NORM_TILING_H
#define ASCEND_OPS_RMS_NORM_TILING_H

#include <mki/launch_param.h>
#include <mki/kernel_info.h>
#include <mki/utils/status/status.h>

namespace AsdOps {
using namespace Mki;
constexpr uint32_t TENSOR_X_IDX = 0;
constexpr uint32_t TENSOR_BIAS_IDX = 1;
constexpr uint32_t TENSOR_RES_IN_IDX = 2;
constexpr uint32_t TENSOR_GAMMA_IDX = 3;
constexpr uint32_t TENSOR_Y_IDX = 0;
constexpr uint32_t TENSOR_RES_OUT_IDX = 1;
Status PostRmsNormTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo);
} // namespace AsdOps
#endif