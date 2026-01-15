/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "unpad_with_hidden_state_ops_runner.h"
#include <atbops/params/params.h>
#include "atb/utils/log.h"


namespace atb {

UnpadWithHiddenStateOpsRunner::UnpadWithHiddenStateOpsRunner(const train::UnpadWithHiddenStateParam &param)
    : OpsRunner("UnpadWithHiddenStateOpsRunner", RUNNER_TYPE_UNPAD_WITH_HIDDEN_STATE), param_(param)
{
}
UnpadWithHiddenStateOpsRunner::~UnpadWithHiddenStateOpsRunner() {}

Status UnpadWithHiddenStateOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    (void)opsTensorPack;
    const size_t inTensorSize = 1;
    const size_t outTensorSize = 1;
    kernelGraph_.inTensors.resize(inTensorSize);
    kernelGraph_.outTensors.resize(outTensorSize);

    Mki::Tensor &dataInputTensor = kernelGraph_.inTensors.at(0);
    Mki::Tensor &dataOutputTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &trainUnpadNode = kernelGraph_.nodes.at(0);
    AtbOps::OpParam::UnpadWithHiddenState unpadWithHiddenStateParam;
    unpadWithHiddenStateParam.qSeqLen = param_.qSeqLen;
    unpadWithHiddenStateParam.maxSeqLen = static_cast<uint32_t>(param_.maxSeqLen);
    trainUnpadNode.opDesc = {0, "UnpadWithHiddenStateOperation", unpadWithHiddenStateParam};
    trainUnpadNode.inTensors = {&dataInputTensor};
    trainUnpadNode.outTensors = {&dataOutputTensor};

    return NO_ERROR;
}

void UnpadWithHiddenStateOpsRunner::SetParam(const Mki::Any &param)
{
    train::UnpadWithHiddenStateParam newParam = Mki::AnyCast<train::UnpadWithHiddenStateParam>(param);
    if (!(newParam == param_)) {
        ATB_LOG(DEBUG) << GetLogPrefix() << "UnpadWithHiddenStateOpsRunner Param Changed!";
        param_ = newParam;
        isParamUpdated_ = true;
    }
}
} // namespace atb
