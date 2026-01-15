/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "scatter_elements_v2_tiling.h"

#include <mki/utils/SVector/SVector.h>
#include <mki/utils/assert/assert.h>
#include <mki/utils/log/log.h>

#include "asdops/params/scatter_elements_v2.h"
#include "tbe_tiling_runner.h"

namespace AsdOps {
Status ScatterElementsV2CommonTiling(const std::string &kernelName, const LaunchParam &launchParam,
                                     KernelInfo &kernelInfo, const BinHandle &binHandle)
{
    // 获取输入张量和输出张量的描述信息
    const auto &tensorDesc0 = launchParam.GetInTensor(0).desc;    // 输入张量 0
    const auto &tensorDesc1 = launchParam.GetInTensor(1).desc;    // 输入张量 1（例如索引张量）
    const auto &tensorDesc2 = launchParam.GetInTensor(2).desc;    // 输入张量 1（例如索引张量）
    const auto &tensorDescOut = launchParam.GetOutTensor(0).desc; // 输出张量

    // 获取 scatter_elements 算子的参数
    const auto &param = AnyCast<OpParam::ScatterElementsV2>(launchParam.GetParam());
    std::string reductionStr = "";

    if (param.reduction == OpParam::ScatterElementsV2::ReductionType::NONE) {
        reductionStr = "none";
    } else if (param.reduction == OpParam::ScatterElementsV2::ReductionType::ADD) {
        reductionStr = "add";
    } else {
        MKI_LOG(ERROR) << "reduction only support none or add";
    }

    // 创建 TbeTilingRunner 对象
    auto runner = AsdOpsGeRt::TbeTilingRunner()
                      .SetKernelName(kernelName) // 设置内核名称
                      .SetName("ScatterElementsV2")
                      .AddInput(tensorDesc0.dtype, tensorDesc0.format, tensorDesc0.dims)        // 添加输入张量 0
                      .AddInput(tensorDesc1.dtype, tensorDesc1.format, tensorDesc1.dims)        // 添加输入张量 1
                      .AddInput(tensorDesc2.dtype, tensorDesc2.format, tensorDesc2.dims)        // 添加输入张量 2
                      .AddOutput(tensorDescOut.dtype, tensorDescOut.format, tensorDescOut.dims) // 添加输出张量
                      .AddAttrInt(param.axis)               // 添加 ScatterElementsV2 的 axis 属性
                      .AddAttrStr(reductionStr.c_str()); // 添加 ScatterElementsV2 的 reduction 属性

    // 获取 tiling 结果并返回状态
    return GetTilingFromRunner(kernelInfo, runner, binHandle);
}
} // namespace AsdOps
