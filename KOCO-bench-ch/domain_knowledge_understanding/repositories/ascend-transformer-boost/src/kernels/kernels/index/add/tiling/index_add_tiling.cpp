/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "index_add_tiling.h"

#include "asdops/params/index.h"
#include "tbe_tiling_runner.h"

namespace AsdOps {
Status IndexAddTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                      const BinHandle &binHandle)
{
    const auto &tensorDesc0 = launchParam.GetInTensor(0).desc;
    const auto &tensorDesc1 = launchParam.GetInTensor(1).desc;
    const auto &tensorDesc2 = launchParam.GetInTensor(2).desc;
    const auto &tensorDesc3 = launchParam.GetInTensor(3).desc;

    const auto &tensorDescOut = launchParam.GetOutTensor(0).desc;

    const auto &param = AnyCast<OpParam::Index>(launchParam.GetParam());

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetName("InplaceIndexAdd")
        .SetKernelName(kernelName)
        .AddInput(tensorDesc0.dtype, tensorDesc0.format, tensorDesc0.dims)
        .AddInput(tensorDesc1.dtype, tensorDesc1.format, tensorDesc1.dims)
        .AddInput(tensorDesc2.dtype, tensorDesc2.format, tensorDesc2.dims)
        .AddInput(tensorDesc3.dtype, tensorDesc3.format, tensorDesc3.dims)
        .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims)
        .AddAttrInt(param.axis);

    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}
} // namespace AsdOps