/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "genattentionmask_ops_runner.h"
#include <atb/utils/log.h>
#include <asdops/params/params.h>
#include <atbops/params/params.h>

namespace atb {
GenAttentionMaskOpsRunner::GenAttentionMaskOpsRunner(const train::GenAttentionMaskParam &param)
    : OpsRunner("GenAttentionMaskOpsRunner", RUNNER_TYPE_GENATTENTIONMASK), param_(param)
{
}
GenAttentionMaskOpsRunner::~GenAttentionMaskOpsRunner() {}

Status GenAttentionMaskOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    (void)opsTensorPack;
    kernelGraph_.inTensors.resize(1);
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(0);

    kernelGraph_.outTensors.resize(1);
    Mki::Tensor &resultTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &genAttentionMaskNode = kernelGraph_.nodes.at(0);

    AtbOps::OpParam::GenAttentionMask genAttentionMaskParam;
    genAttentionMaskParam.headNum = param_.headNum;
    for (std::size_t i = 0; i < param_.seqLen.size(); ++i) {
        genAttentionMaskParam.qSeqLen.push_back(param_.seqLen[i]);
    }
    genAttentionMaskNode.opDesc = {0, "GenAttentionMaskOperation", genAttentionMaskParam};
    genAttentionMaskNode.inTensors = {&xTensor};
    genAttentionMaskNode.outTensors = {&resultTensor};

    return NO_ERROR;
}

void GenAttentionMaskOpsRunner::SetParam(const Mki::Any &param)
{
    train::GenAttentionMaskParam newParam = Mki::AnyCast<train::GenAttentionMaskParam>(param);
    if (!(newParam == param_)) {
        ATB_LOG(DEBUG) << GetLogPrefix() << "GenAttentionMaskOpsRunner Param Changed!";
        param_ = newParam;
        isParamUpdated_ = true;
    }
}
} // namespace atb