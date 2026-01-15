/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_OPS_MATMUL_PP_MIX_TILING_H
#define ASCEND_OPS_MATMUL_PP_MIX_TILING_H
#include <mki/launch_param.h>
#include <mki/kernel_info.h>
#include <mki/utils/status/status.h>

namespace AsdOps {
using namespace Mki;
Status PpMatmulMixTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo);
} // namespace AsdOps

#endif // ASCEND_OPS_MATMUL_PP_MIX_TILING_H
