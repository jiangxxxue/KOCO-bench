/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fill_ops_runner.h"
#include <asdops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"
namespace atb {
FillOpsRunner::FillOpsRunner(const infer::FillParam &param)
    : OpsRunner("FillOpsRunner", RUNNER_TYPE_FILL), param_(param)
{
    ATB_LOG(INFO) << "FillOpsRunner::FillOpsRunner called";
    kernelGraph_.outTensors.resize(1);
    kernelGraph_.nodes.resize(1);

    if (param_.withMask) {
        SetMaskedFillGraph();
    } else {
        SetFillGraph();
    }
}

void FillOpsRunner::SetFillGraph()
{
    kernelGraph_.inTensors.resize(0);
    auto &fillNode = kernelGraph_.nodes.at(0);
    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(0);

    AsdOps::OpParam::Fill fillParam;
    fillParam.withMask = false;
    TensorUtil::AtbSVector2OpsSVector(param_.value, fillParam.value);
    TensorUtil::AtbSVector2OpsSVector(param_.outDim, fillParam.outDim);

    fillNode.opDesc = {0, "FillOperation", fillParam};
    fillNode.outTensors = {&outTensor};
}

void FillOpsRunner::SetMaskedFillGraph()
{
    kernelGraph_.inTensors.resize(2); // 2: maskedFill算子的intensor数为2
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(0);
    Mki::Tensor &maskTensor = kernelGraph_.inTensors.at(1);
    auto &maskedFillNode = kernelGraph_.nodes.at(0);
    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(0);

    AsdOps::OpParam::Fill maskedFillParam;
    maskedFillParam.withMask = true;
    TensorUtil::AtbSVector2OpsSVector(param_.value, maskedFillParam.value);
    maskedFillNode.opDesc = {0, "FillOperation", maskedFillParam};
    maskedFillNode.inTensors = {&xTensor, &maskTensor};
    maskedFillNode.outTensors = {&outTensor};
}

FillOpsRunner::~FillOpsRunner() {}

} // namespace atb
