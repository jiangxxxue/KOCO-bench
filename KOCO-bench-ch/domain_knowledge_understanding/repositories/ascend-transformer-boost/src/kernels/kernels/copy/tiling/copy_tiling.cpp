/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "copy_tiling.h"
#include <mki/types.h>
#include <mki/utils/log/log.h>
#include <mki/utils/SVector/SVector.h>
#include "asdops/params/copy.h"
#include "tbe_tiling_runner.h"

namespace AsdOps {
Status  ViewCopyTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                       const BinHandle &binHandle)
{
    const TensorDesc &tensorDesc0 = launchParam.GetInTensor(0).desc;
    const TensorDesc &tensorDesc1 = launchParam.GetInTensor(1).desc;
    const TensorDesc &tensorDescOut = launchParam.GetOutTensor(0).desc;

    auto param = AnyCast<OpParam::Copy>(launchParam.GetParam());

    /* Attention: now, src size/stride/offset is useless in tiling, placeholder by dst. */
    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetName("ViewCopy")
        .SetKernelName(kernelName)
        .AddInput(tensorDesc0.dtype, tensorDesc0.format, tensorDesc0.dims)
        .AddConstInput(TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {static_cast<int64_t>(param.dstSize.size())},
                       param.dstSize.data(), param.dstSize.size() * sizeof(int64_t))
        .AddConstInput(TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {static_cast<int64_t>(param.dstStride.size())},
                       param.dstStride.data(), param.dstStride.size() * sizeof(int64_t))
        .AddConstInput(TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {static_cast<int64_t>(param.dstOffset.size())},
                       param.dstOffset.data(), param.dstOffset.size() * sizeof(int64_t))
        .AddInput(tensorDesc1.dtype, tensorDesc1.format, tensorDesc1.dims)
        .AddConstInput(TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {static_cast<int64_t>(param.dstSize.size())},
                       param.dstSize.data(), param.dstSize.size() * sizeof(int64_t))
        .AddConstInput(TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {static_cast<int64_t>(param.dstStride.size())},
                       param.dstStride.data(), param.dstStride.size() * sizeof(int64_t))
        .AddConstInput(TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, {static_cast<int64_t>(param.dstOffset.size())},
                       param.dstOffset.data(), param.dstOffset.size() * sizeof(int64_t))
        .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims);

        return GetTilingFromRunner(kernelInfo, runner, binHandle);
}
} // namespace AsdOps
