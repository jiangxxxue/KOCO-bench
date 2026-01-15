/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef MLA_TILING_H
#define MLA_TILING_H

#include <mki/kernel_info.h>
#include <mki/launch_param.h>
#include <mki/utils/log/log.h>
#include <mki/utils/status/status.h>
#include "atbops/params/params.h"
#include "ring_mla_tiling_dependency.h"

namespace AtbOps {
using namespace Mki;
Status RINGMLATiling(const LaunchParam &launchParam, KernelInfo &kernelInfo);
Status RINGMLAPrefillTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo);
Status GetRINGMLATilingParam(const LaunchParam &launchParam, const RINGMLAInfo &mmInfo,
    uint32_t &blockDim, uint32_t *tilingParam, uint64_t tilingParamSize);
Status GetRINGMLAPrefillTilingParam(const RINGMLAInfo &mmInfo, uint32_t &blockDim,
                                    uint32_t *tilingParam, uint32_t tilingParamSize);
}

#endif // MLA_TILING_H
