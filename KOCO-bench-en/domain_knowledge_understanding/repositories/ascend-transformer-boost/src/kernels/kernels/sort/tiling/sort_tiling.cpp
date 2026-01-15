/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sort_tiling.h"

#include <mki/types.h>
#include <mki/utils/any/any.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include <mki/utils/SVector/SVector.h>
#include "asdops/params/sort.h"
#include "tbe_tiling_runner.h"

namespace AsdOps {
Status TopKDescTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                      const BinHandle &binHandle)
{
    const auto &tensorDesc0 = launchParam.GetInTensor(0).desc;
    const auto &tensorDescOut0 = launchParam.GetOutTensor(0).desc;
    const auto &tensorDescOut1 = launchParam.GetOutTensor(1).desc;
    const auto &opParam = AnyCast<OpParam::Sort>(launchParam.GetParam());
    const auto &num = opParam.num;
    SVector<int64_t> numShape = {static_cast<int64_t>(num.size())};

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetName("TopKV2")
        .SetKernelName(kernelName)
        .AddInput(tensorDesc0.dtype, tensorDesc0.format, tensorDesc0.dims)
        .AddConstInput(TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, numShape, num.data(), num.size() * sizeof(int32_t))
        .AddOutput(tensorDescOut0.dtype, tensorDescOut0.format, tensorDescOut0.dims)
        .AddOutput(tensorDescOut1.dtype, tensorDescOut1.format, tensorDescOut1.dims)
        .AddAttrBool(true)
        .AddAttrInt64(-1)
        .AddAttrBool(true);

    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}
} // namespace AsdOps