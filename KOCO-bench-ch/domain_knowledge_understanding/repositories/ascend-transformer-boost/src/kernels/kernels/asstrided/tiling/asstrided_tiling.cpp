/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asstrided_tiling.h"

#include <mki/types.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/SVector/SVector.h>
#include "asdops/params/asstrided.h"
#include "tbe_tiling_runner.h"

namespace AsdOps {
Status AsStridedTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                       const BinHandle &binHandle)
{
    const auto &tensorDesc0 = launchParam.GetInTensor(0).desc;
    const auto &tensorDescOut = launchParam.GetOutTensor(0).desc;
    const auto &param = AnyCast<OpParam::AsStrided>(launchParam.GetParam());

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetName("AsStrided")
        .SetKernelName(kernelName)
        .AddInput(tensorDesc0.dtype, tensorDesc0.format, tensorDesc0.dims)
        .AddConstInput(TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {static_cast<int64_t>(param.size.size())},
                       param.size.data(), param.size.size() * sizeof(int64_t))
        .AddConstInput(TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {static_cast<int64_t>(param.stride.size())},
                       param.stride.data(), param.stride.size() * sizeof(int64_t))
        .AddConstInput(TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {static_cast<int64_t>(param.offset.size())},
                       param.offset.data(), param.offset.size() * sizeof(int64_t))
        .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims);

    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}
} // namespace AsdOps
