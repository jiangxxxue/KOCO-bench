/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDOPS_LOGPROBS_SAMPLE_TILING_H
#define ASDOPS_LOGPROBS_SAMPLE_TILING_H

#include <mki/utils/status/status.h>
#include <mki/launch_param.h>
#include <mki/kernel_info.h>
#include <mki/launch_param.h>

namespace AsdOps {
using namespace Mki;

constexpr uint32_t INPUT_SORTED_PROBS_INDEX = 0;
constexpr uint32_t INPUT_LOGPROBS_SIZE_INDEX = 3;

Status LogprobsSampleTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo);
} // namespace AsdOps

#endif // ASDOPS_LOGPROBS_SAMPLE_TILING_H
