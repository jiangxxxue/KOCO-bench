/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "swiglu_quant_ops_runner.h"
#include <atbops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"
 
namespace atb {
static const int32_t IN_TENSOR_NUM = 1;
static const int32_t OUT_TENSOR_NUM = 2;
static const int32_t TENSOR_ONE_INDEX = 1;
 
SwigluQuantOpsRunner::SwigluQuantOpsRunner(const infer::SwigluQuantParam &param)
    : OpsRunner("SwigluQuantOpsRunner", RUNNER_TYPE_SWIGLU_QUANT), param_(param)
{
    ATB_LOG(INFO) << "SwigluQuantOpsRunner::SwigluQuantOpsRunner called";
    kernelGraph_.inTensors.resize(IN_TENSOR_NUM);
    kernelGraph_.outTensors.resize(OUT_TENSOR_NUM);
 
    Mki::Tensor &intensor = kernelGraph_.inTensors.at(0);
    
    Mki::Tensor &operationOutTensor = kernelGraph_.outTensors.at(0);
    Mki::Tensor &operationOutTensorTwo = kernelGraph_.outTensors.at(TENSOR_ONE_INDEX);
 
    kernelGraph_.nodes.resize(1);
    auto &swigluQuantNode = kernelGraph_.nodes.at(0);
    AtbOps::OpParam::SwigluQuant swiGluQuantParam;
    swigluQuantNode.opDesc = {0, "SwiGluQuantOperation", swiGluQuantParam};
    swigluQuantNode.inTensors = {&intensor};
    swigluQuantNode.outTensors = {&operationOutTensor, &operationOutTensorTwo};
}
 
SwigluQuantOpsRunner::~SwigluQuantOpsRunner() {}
} // namespace atb