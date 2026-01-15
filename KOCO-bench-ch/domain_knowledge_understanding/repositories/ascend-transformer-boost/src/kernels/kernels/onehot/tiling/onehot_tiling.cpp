/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "onehot_tiling.h"
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include "asdops/params/onehot.h"
#include "tbe_tiling_runner.h"

namespace AsdOps {
Status OnehotTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                    const BinHandle &binHandle)
{
    const TensorDesc &tensorDesc0 = launchParam.GetInTensor(0).desc;
    const TensorDesc &tensorDesc1 = launchParam.GetInTensor(1).desc;
    const TensorDesc &tensorDesc2 = launchParam.GetInTensor(2).desc;
    const auto &tensorDescOut = launchParam.GetOutTensor(0).desc;
    const auto &param = AnyCast<OpParam::Onehot>(launchParam.GetParam());
    SVector<int64_t> depthShape = {1};

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetName("OneHot")
        .SetKernelName(kernelName)
        .AddInput(tensorDesc0.dtype, tensorDesc0.format, tensorDesc0.dims)
        .AddConstInput(TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND,
                       depthShape, param.depth.data(), param.depth.size() * sizeof(int64_t))
        .AddInput(tensorDesc1.dtype, tensorDesc1.format, tensorDesc1.dims)
        .AddInput(tensorDesc2.dtype, tensorDesc2.format, tensorDesc2.dims)
        .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims)
        .AddAttrInt64(param.axis);

    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}
}