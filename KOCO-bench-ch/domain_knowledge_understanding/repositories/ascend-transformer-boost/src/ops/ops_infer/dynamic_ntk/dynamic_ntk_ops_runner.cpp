/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dynamic_ntk_ops_runner.h"
#include <asdops/params/params.h>


static const uint32_t ONE_CUSTOM_INPUT_IN_TENSOR_NUM = 2;
static const uint32_t TWO_CUSTOM_INPUT_IN_TENSOR_NUM = 3;
static const uint32_t OUT_TENSOR_NUM = 2;

namespace atb {

DynamicNTKOpsRunner::DynamicNTKOpsRunner(const infer::DynamicNTKParam &param)
    : OpsRunner("DynamicNTKOpsRunner", RUNNER_TYPE_DYNAMICNTK), param_(param), asdopsParam_({})
{
    kernelGraph_.nodes.resize(1);
    asdopsParam_.outType = param_.outDataType == ACL_FLOAT16 ? 0 : 1; // 0: 输出fp16, 1: 输出bf16

    kernelGraph_.inTensors.resize(TWO_CUSTOM_INPUT_IN_TENSOR_NUM);
    size_t inId = 0;
    Mki::Tensor &positionIds = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &invfreqIn = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &seqlens = kernelGraph_.inTensors.at(inId++);

    kernelGraph_.outTensors.resize(OUT_TENSOR_NUM);
    size_t outId = 0;
    Mki::Tensor &sinTensor = kernelGraph_.outTensors.at(outId++);
    Mki::Tensor &cosTensor = kernelGraph_.outTensors.at(outId++);

    auto &dynamicNTKNode = kernelGraph_.nodes[0];
    dynamicNTKNode.opDesc = {0, "DynamicNTKOperation", asdopsParam_};
    dynamicNTKNode.inTensors = {&positionIds, &invfreqIn, &seqlens};
    dynamicNTKNode.outTensors = {&sinTensor, &cosTensor};
}

DynamicNTKOpsRunner::~DynamicNTKOpsRunner() {}
} // namespace atb
