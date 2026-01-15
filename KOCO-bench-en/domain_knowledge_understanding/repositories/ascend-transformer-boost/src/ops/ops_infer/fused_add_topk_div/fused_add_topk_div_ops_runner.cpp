/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fused_add_topk_div_ops_runner.h"
#include <asdops/params/params.h>
#include <atbops/params/params.h>
#include "atb/utils/log.h"

static constexpr size_t SIZE_2 = 2;
static constexpr size_t SIZE_3 = 3;
static constexpr size_t SIZE_4 = 4;

namespace atb {
FusedAddTopkDivOpsRunner::FusedAddTopkDivOpsRunner(const infer::FusedAddTopkDivParam &param)
    : OpsRunner("FusedAddTopkDivOpsRunner", RUNNER_TYPE_FUSED_ADD_TOPK_DIV), param_(param)
{
}

FusedAddTopkDivOpsRunner::~FusedAddTopkDivOpsRunner() {}

void FusedAddTopkDivOpsRunner::SetParam(const Mki::Any &param)
{
    infer::FusedAddTopkDivParam newParam = Mki::AnyCast<infer::FusedAddTopkDivParam>(param);
    if (!(newParam == param_)) {
        ATB_LOG(DEBUG) << GetLogPrefix() << "FusedAddTopkDivOpsRunner param changed!";
        param_ = newParam;
        isParamUpdated_ = true;
    }
}

Status FusedAddTopkDivOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    kernelGraph_.inTensors.resize(SIZE_4);
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(0);
    Mki::Tensor &addNumTensor = kernelGraph_.inTensors.at(1);
    Mki::Tensor &mappingNumTensor =
        param_.enableExpertMapping ? kernelGraph_.inTensors.at(2) : nullTensor_; // 2: mappingNum
    Mki::Tensor &mappingTableTensor =
        param_.enableExpertMapping ? kernelGraph_.inTensors.at(3) : nullTensor_; // 3: mappingTable
    kernelGraph_.outTensors.resize(SIZE_2);
    Mki::Tensor &yTensor = kernelGraph_.outTensors.at(0);
    Mki::Tensor &indicesTensor = kernelGraph_.outTensors.at(1);

    kernelGraph_.internalTensors.resize(0);

    kernelGraph_.nodes.resize(1);
    KernelGraphNode &fusedAddTopkDivNode = kernelGraph_.nodes.at(0);

    AtbOps::OpParam::FusedAddTopkDiv fusedAddTopkDivParam;
    fusedAddTopkDivParam.groupNum = param_.groupNum;
    fusedAddTopkDivParam.groupTopk = param_.groupTopk;
    fusedAddTopkDivParam.n = param_.n;
    fusedAddTopkDivParam.k = param_.k;
    fusedAddTopkDivParam.activateType = 0;
    fusedAddTopkDivParam.isNorm = param_.isNorm;
    fusedAddTopkDivParam.scale = param_.scale;
    fusedAddTopkDivParam.enableExpertMapping = param_.enableExpertMapping;
    fusedAddTopkDivNode.opDesc = {0, "FusedAddTopkDivOperation", fusedAddTopkDivParam};
    fusedAddTopkDivNode.inTensors = {&xTensor, &addNumTensor, &mappingNumTensor, &mappingTableTensor};
    fusedAddTopkDivNode.outTensors = {&yTensor, &indicesTensor};

    (void)opsTensorPack;
    return NO_ERROR;
}
} // namespace atb