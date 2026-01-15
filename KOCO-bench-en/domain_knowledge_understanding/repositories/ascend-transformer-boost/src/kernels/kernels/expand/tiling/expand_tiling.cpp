/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "expand_tiling.h"
#include "asdops/params/expand.h"
#include "tbe_tiling_runner.h"

namespace AsdOps {
Status ExpandTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                    const BinHandle &binHandle)
{
    const TensorDesc &tensorDesc0 = launchParam.GetInTensor(0).desc;
    const TensorDesc &tensorDescOut = launchParam.GetOutTensor(0).desc;
    auto param = AnyCast<OpParam::Expand>(launchParam.GetParam());
    SVector<int64_t> shape = param.shape;
    SVector<int64_t> shapeDim = {static_cast<int64_t>(shape.size())};

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetName("Expand")
        .SetKernelName(kernelName)
        .AddInput(tensorDesc0.dtype, tensorDesc0.format, tensorDesc0.dims)
        .AddConstInput(TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND,
                       shapeDim, shape.data(), shape.size() * sizeof(int64_t))
        .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims);
    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}

}