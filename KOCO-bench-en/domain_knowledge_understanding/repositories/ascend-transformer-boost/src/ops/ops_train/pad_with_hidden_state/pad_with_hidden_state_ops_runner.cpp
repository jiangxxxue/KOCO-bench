/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pad_with_hidden_state_ops_runner.h"
#include <atbops/params/params.h>
#include "atb/utils/log.h"

namespace atb {

PadWithHiddenStateOpsRunner::PadWithHiddenStateOpsRunner(const train::PadWithHiddenStateParam &param)
    : OpsRunner("PadWithHiddenStateOpsRunner", RUNNER_TYPE_PAD_WITH_HIDDEN_STATE), param_(param)
{
}
PadWithHiddenStateOpsRunner::~PadWithHiddenStateOpsRunner() {}

Status PadWithHiddenStateOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    (void)opsTensorPack;
    kernelGraph_.inTensors.resize(1);
    kernelGraph_.outTensors.resize(1);

    Mki::Tensor &dataInputTensor = kernelGraph_.inTensors.at(0);
    Mki::Tensor &dataOutputTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &trainPadNode = kernelGraph_.nodes.at(0);
    AtbOps::OpParam::PadWithHiddenState padWithHiddenStateParam;
    padWithHiddenStateParam.qSeqLen = param_.qSeqLen;
    padWithHiddenStateParam.maxSeqLen = static_cast<uint32_t>(param_.maxSeqLen);
    trainPadNode.opDesc = {0, "PadWithHiddenStateOperation", padWithHiddenStateParam};
    trainPadNode.inTensors = {&dataInputTensor};
    trainPadNode.outTensors = {&dataOutputTensor};

    return NO_ERROR;
}

void PadWithHiddenStateOpsRunner::SetParam(const Mki::Any &param)
{
    train::PadWithHiddenStateParam newParam = Mki::AnyCast<train::PadWithHiddenStateParam>(param);
    if (!(newParam == param_)) {
        ATB_LOG(DEBUG) << GetLogPrefix() << "PadWithHiddenStateOpsRunner Param Changed!";
        param_ = newParam;
        isParamUpdated_ = true;
    }
}
} // namespace atb
