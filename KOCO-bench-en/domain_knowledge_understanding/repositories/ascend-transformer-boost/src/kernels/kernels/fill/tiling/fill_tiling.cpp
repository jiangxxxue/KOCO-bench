/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fill_tiling.h"
#include <mki/utils/fp16/fp16_t.h>
#include <mki/utils/log/log.h>
#include "asdops/params/params.h"
#include "tbe_tiling_runner.h"

namespace AsdOps {
using namespace Mki;
Status MaskedFillTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                        const BinHandle &binHandle)
{
    const TensorDesc &tensorDesc0 = launchParam.GetInTensor(0).desc;
    const TensorDesc &tensorDesc1 = launchParam.GetInTensor(1).desc;
    const TensorDesc &tensorDescOut = launchParam.GetOutTensor(0).desc;
    SVector<int64_t> tensorDesc2dims = {1};

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetKernelName(kernelName)
        .AddInput(tensorDesc0.dtype, tensorDesc0.format, tensorDesc0.dims)
        .AddInput(tensorDesc1.dtype, tensorDesc1.format, tensorDesc1.dims)
        .AddInput(TensorDType::TENSOR_DTYPE_FLOAT16, TensorFormat::TENSOR_FORMAT_ND, tensorDesc2dims)
        .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims);

    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}

Status FillTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                  const BinHandle &binHandle)
{
    const TensorDesc &tensorDescOut = launchParam.GetOutTensor(0).desc;
    auto param = AnyCast<OpParam::Fill>(launchParam.GetParam());
    SVector<int64_t> outDim = param.outDim;
    SVector<int64_t> outDimShape = {static_cast<int64_t>(outDim.size())};
    SVector<fp16_t> value;
    for (size_t i = 0; i < param.value.size(); i++) {
        value.push_back(static_cast<fp16_t>(param.value.at(i)));
    }
    SVector<int64_t> valueShape = {static_cast<int64_t>(value.size())};
    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetName("Fill")
        .SetKernelName(kernelName)
        .AddConstInput(TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, outDimShape, outDim.data(),
        outDim.size() * sizeof(int64_t))
        .AddConstInput(TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, valueShape, value.data(),
        value.size() * sizeof(float) / 2) // 2 the half of float
        .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims);

    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}
}