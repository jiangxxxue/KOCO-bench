/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "activation_tiling.h"

#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include "asdops/params/params.h"
#include "tbe_tiling_runner.h"

namespace AsdOps {
using namespace Mki;
Status ActivationCommonTiling(
    const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo, const BinHandle &handle)
{
    const auto &tensorDescOut = launchParam.GetOutTensor(0).desc;
    const auto &tensorDesc = launchParam.GetInTensor(0).desc;

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetKernelName(kernelName)
        .AddInput(tensorDesc.dtype, tensorDesc.format, tensorDesc.dims)
        .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims);

    return GetTilingFromRunner(kernelInfo, runner, handle);
}

Status SwishTiling(
    const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo, const BinHandle &handle)
{
    const auto &tensorDesc = launchParam.GetInTensor(0).desc;
    const auto &tensorDescOut = launchParam.GetOutTensor(0).desc;
    const auto &param = AnyCast<OpParam::Activation>(launchParam.GetParam());

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetKernelName(kernelName)
        .AddInput(tensorDesc.dtype, tensorDesc.format, tensorDesc.dims)
        .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims)
        .AddAttrFloat(param.scale);

    return GetTilingFromRunner(kernelInfo, runner, handle);
}
}