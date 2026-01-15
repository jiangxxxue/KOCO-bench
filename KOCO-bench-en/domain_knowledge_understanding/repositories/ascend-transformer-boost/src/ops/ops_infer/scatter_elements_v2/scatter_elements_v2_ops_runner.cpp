/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "scatter_elements_v2_ops_runner.h"
#include <asdops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"

 
namespace atb {
ScatterElementsV2OpsRunner::ScatterElementsV2OpsRunner(const infer::ScatterElementsV2Param &param)
    : OpsRunner("ScatterElementsV2OpsRunner", RUNNER_TYPE_GATHER), param_(param)
{
    ATB_LOG(INFO) << "ScatterElementsV2OpsRunner::ScatterElementsV2OpsRunner called";
    kernelGraph_.inTensors.resize(3); // intersorNum:3
    Mki::Tensor &inputTensor = kernelGraph_.inTensors.at(0);
    Mki::Tensor &indiceTensor = kernelGraph_.inTensors.at(1);
    Mki::Tensor &updateTensor = kernelGraph_.inTensors.at(2);
 
    // 原地写算子，无须创建outTensors
    kernelGraph_.outTensors.resize(0);
 
    kernelGraph_.nodes.resize(1);
    auto &scatterElementsV2Node = kernelGraph_.nodes[0];

    AsdOps::OpParam::ScatterElementsV2::ReductionType reduction = AsdOps::OpParam::ScatterElementsV2::ReductionType::NONE;
    if (param_.reduction == atb::infer::ScatterElementsV2Param::ReductionType::NONE) {
        reduction = AsdOps::OpParam::ScatterElementsV2::ReductionType::NONE;
    } else if (param_.reduction == atb::infer::ScatterElementsV2Param::ReductionType::ADD) {
        reduction = AsdOps::OpParam::ScatterElementsV2::ReductionType::ADD;
    } else {
        MKI_LOG(ERROR) << "reduction only support none or add";
    }
 
    AsdOps::OpParam::ScatterElementsV2 scatterElementsV2NodeParam = {reduction, param_.axis};
 
    scatterElementsV2Node.opDesc = {0, "ScatterElementsV2Operation", scatterElementsV2NodeParam};
    scatterElementsV2Node.inTensors = {&inputTensor, &indiceTensor, &updateTensor};
 
    // 原地写算子，无须创建outTensors指向输入tensor
    scatterElementsV2Node.outTensors = {&inputTensor};
}
 
ScatterElementsV2OpsRunner::~ScatterElementsV2OpsRunner() {}
 
} // namespace atb