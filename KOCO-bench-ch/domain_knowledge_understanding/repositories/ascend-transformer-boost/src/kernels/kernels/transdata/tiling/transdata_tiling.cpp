/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "transdata_tiling.h"
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>
#include "asdops/params/transdata.h"
#include "tbe_tiling_runner.h"

namespace AsdOps {
Status TransdataCommonTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                             const BinHandle &binHandle)
{
    const auto &tensorDescIn = launchParam.GetInTensor(0).desc;
    const auto &tensorDescOut = launchParam.GetOutTensor(0).desc;

    auto runner = AsdOpsGeRt::TbeTilingRunner()
        .SetName("TransData")
        .SetKernelName(kernelName)
        .AddInput(tensorDescIn.dtype, tensorDescIn.format, tensorDescIn.dims)
        .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims);

    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}
} // namespace AsdOps