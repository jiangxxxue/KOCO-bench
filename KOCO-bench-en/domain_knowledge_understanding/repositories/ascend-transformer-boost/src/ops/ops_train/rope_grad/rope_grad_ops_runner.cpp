/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rope_grad_ops_runner.h"
#include <asdops/params/params.h>
#include <atb/utils/log.h>
#include <atb/utils/tensor_util.h>
#include <atbops/params/params.h>


namespace atb {
static const uint64_t IN_TENSOR_COUNT = 4;
static const uint64_t OUT_TENSOR_COUNT = 2;

RopeGradOpsRunner::RopeGradOpsRunner(const train::RopeGradParam &param)
    : OpsRunner("RopeGradOpsRunner", RUNNER_TYPE_ROPE_GRAD), param_(param)
{
}
RopeGradOpsRunner::~RopeGradOpsRunner() {}

Status RopeGradOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    (void)opsTensorPack;
    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT);
    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT);

    int64_t inTensorNum = 0;
    Mki::Tensor &qEmbeddedGrad = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &kEmbeddedGrad = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &cos = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &sin = kernelGraph_.inTensors.at(inTensorNum++);

    int64_t outTensorNum = 0;
    Mki::Tensor &qGrad = kernelGraph_.outTensors.at(outTensorNum++);
    Mki::Tensor &kGrad = kernelGraph_.outTensors.at(outTensorNum++);

    kernelGraph_.nodes.resize(1);
    auto &ropeGradNode = kernelGraph_.nodes.at(0);

    AtbOps::OpParam::RopeGrad ropeGradParam;
    ropeGradParam.qSeqLen = param_.qSeqLen;
    ropeGradNode.opDesc = {0, "RopeGradOperation", ropeGradParam};
    ropeGradNode.inTensors = {&qEmbeddedGrad, &kEmbeddedGrad, &cos, &sin};
    ropeGradNode.outTensors = {&qGrad, &kGrad};

    return NO_ERROR;
}

void RopeGradOpsRunner::SetParam(const Mki::Any &param)
{
    train::RopeGradParam newParam = Mki::AnyCast<train::RopeGradParam>(param);
    if (!(newParam == param_)) {
        ATB_LOG(DEBUG) << GetLogPrefix() << "RopeGradOpsRunner Param Changed!";
        param_ = newParam;
        isParamUpdated_ = true;
    }
}
} // namespace atb