/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * AscendOpCommonLib is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include "zeroslike_tiling.h"

#include "asdops/params/zeroslike.h"
#include "mki/types.h"
#include "mki/utils/SVector/SVector.h"
#include "mki/utils/assert/assert.h"
#include "mki/utils/log/log.h"
#include "tbe_tiling_runner.h"

namespace AsdOps {
using namespace Mki;
Status ZerosLikeTiling(const std::string &kernelName, const LaunchParam &launchParam, KernelInfo &kernelInfo,
                       const BinHandle &binHandle)
{
    const auto &tensorDesc0 = launchParam.GetInTensor(0).desc;
    const auto &tensorDescOut = launchParam.GetOutTensor(0).desc;

    auto runner = AsdOpsGeRt::TbeTilingRunner()
                      .SetKernelName(kernelName)
                      .AddInput(tensorDesc0.dtype, tensorDesc0.format, tensorDesc0.dims)
                      .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims);

    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}
} // namespace AsdOps
