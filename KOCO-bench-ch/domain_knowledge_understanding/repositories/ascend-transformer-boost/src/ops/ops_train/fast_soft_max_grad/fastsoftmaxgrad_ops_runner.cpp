/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fastsoftmaxgrad_ops_runner.h"
#include <atbops/params/params.h>
#include "atb/utils/log.h"

namespace atb {

FastSoftMaxGradOpsRunner::FastSoftMaxGradOpsRunner(const train::FastSoftMaxGradParam &param)
    : OpsRunner("FastSoftMaxGradOpsRunner", RUNNER_TYPE_FASTSOFTMAXGRAD), param_(param)
{
}
FastSoftMaxGradOpsRunner::~FastSoftMaxGradOpsRunner() {}

Status FastSoftMaxGradOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    (void)opsTensorPack;
    const size_t inTensorSize = 2;
    const size_t outTensorSize = 1;
    kernelGraph_.inTensors.resize(inTensorSize);
    kernelGraph_.outTensors.resize(outTensorSize);

    Mki::Tensor &yInputTensor = kernelGraph_.inTensors.at(0);
    Mki::Tensor &yGradTensor = kernelGraph_.inTensors.at(1);
    Mki::Tensor &xGradTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &fastSoftMaxGradNode = kernelGraph_.nodes.at(0);
    AtbOps::OpParam::FastSoftMaxGrad fastSoftMaxGradParam;
    fastSoftMaxGradParam.headNum = param_.headNum;
    fastSoftMaxGradParam.qSeqLen = param_.qSeqLen;
    fastSoftMaxGradNode.opDesc = {0, "FastSoftMaxGradOperation", fastSoftMaxGradParam};
    fastSoftMaxGradNode.inTensors = {&yInputTensor, &yGradTensor};
    fastSoftMaxGradNode.outTensors = {&xGradTensor};

    return NO_ERROR;
}

void FastSoftMaxGradOpsRunner::SetParam(const Mki::Any &param)
{
    train::FastSoftMaxGradParam newParam = Mki::AnyCast<train::FastSoftMaxGradParam>(param);
    if (!(newParam == param_)) {
        ATB_LOG(DEBUG) << GetLogPrefix() << "FastSoftMaxGradOpsRunner Param Changed!";
        param_ = newParam;
        isParamUpdated_ = true;
    }
}
} // namespace atb
