/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "layernorm_tiling.h"
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include "asdops/params/params.h"
#include "tbe_tiling_runner.h"

namespace AsdOps {
Status LayerNormTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                       const BinHandle &binHandle)
{
    const auto &tensorDesc0 = launchParam.GetInTensor(0).desc;
    const auto &tensorDesc1 = launchParam.GetInTensor(1).desc;
    const auto &tensorDesc2 = launchParam.GetInTensor(2).desc;
    const auto &tensorDescOut0 = launchParam.GetOutTensor(0).desc;
    const auto &tensorDescOut1 = launchParam.GetOutTensor(1).desc;
    const auto &tensorDescOut2 = launchParam.GetOutTensor(2).desc;
    const auto &param = AnyCast<OpParam::Norm>(launchParam.GetParam());

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetKernelName(kernelName)
        .SetName("LayerNormV3")
        .AddInput(tensorDesc0.dtype, tensorDesc0.format, tensorDesc0.dims)
        .AddInput(tensorDesc1.dtype, tensorDesc1.format, tensorDesc1.dims)
        .AddInput(tensorDesc2.dtype, tensorDesc2.format, tensorDesc2.dims)
        .AddOutput(tensorDescOut0.dtype, tensorDescOut0.format, tensorDescOut0.dims)
        .AddOutput(tensorDescOut1.dtype, tensorDescOut1.format, tensorDescOut1.dims)
        .AddOutput(tensorDescOut2.dtype, tensorDescOut2.format, tensorDescOut2.dims)
        .AddAttrInt64(param.beginNormAxis)
        .AddAttrInt64(param.beginParamsAxis)
        .AddAttrFloat(param.epsilon);

    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}
} // namespace AsdOps