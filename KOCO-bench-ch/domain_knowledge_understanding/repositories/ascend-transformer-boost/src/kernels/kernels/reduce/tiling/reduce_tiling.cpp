/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reduce_tiling.h"

#include <mki/utils/log/log.h>
#include "asdops/params/reduce.h"
#include "tbe_tiling_runner.h"

namespace AsdOps {
Status ReduceCommonTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                          const BinHandle &binHandle)
{
    const auto &tensorDesc0 = launchParam.GetInTensor(0).desc;
    const auto &tensorDescOut = launchParam.GetOutTensor(0).desc;

    auto &param = AnyCast<OpParam::Reduce>(launchParam.GetParam());
    auto &axis = param.axis;
    SVector<int64_t> axisDim = {static_cast<int64_t>(axis.size())};

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetKernelName(kernelName)
        .AddInput(tensorDesc0.dtype, tensorDesc0.format, tensorDesc0.dims)
        .AddConstInput(TENSOR_DTYPE_INT64, TENSOR_FORMAT_ND, axisDim, axis.data(), axis.size() * sizeof(int64_t))
        .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims)
        .AddAttrBool(false);

    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}
} // namespace AsdOps