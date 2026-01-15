/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "group_topk_ops_runner.h"
#include <asdops/params/params.h>
#include <atb/utils/log.h>

namespace atb {
GroupTopkOpsRunner::GroupTopkOpsRunner(const infer::GroupTopkParam &param)
    : OpsRunner("GroupTopkOpsRunner", RUNNER_TYPE_GROUP_TOPK), param_(param)
{
    ATB_LOG(INFO) << "GroupTopkOpsRunner::GroupTopkOpsRunner";
}

GroupTopkOpsRunner::~GroupTopkOpsRunner() {}

Status GroupTopkOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    (void)opsTensorPack;

    size_t inTensorNum = 2;
    size_t outTensorNum = 1;
    size_t internalTensorNum = 0;
    size_t nodeNum = 1;
    kernelGraph_.inTensors.resize(inTensorNum);
    kernelGraph_.outTensors.resize(outTensorNum);
    kernelGraph_.internalTensors.resize(internalTensorNum);
    kernelGraph_.nodes.resize(nodeNum);

    ATB_LOG(INFO) << GetLogPrefix() << "GroupTopkOpsRunner::SetupKernelGraph";

    size_t inTensorId = 0;
    Mki::Tensor &tokenTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &idxArrTensor = kernelGraph_.inTensors.at(inTensorId++);

    size_t outTensorId = 0;
    Mki::Tensor &outTensor = kernelGraph_.outTensors.at(outTensorId++);

    size_t nodeId = 0;
    KernelGraphNode &groupTopkNode = kernelGraph_.nodes.at(nodeId++);

    AsdOps::OpParam::GroupTopk groupTopkParam;
    groupTopkParam.groupNum = param_.groupNum;
    groupTopkParam.k = param_.k;
    if (param_.groupMultiFlag == infer::GroupTopkParam::GroupMultiFlag::SUM_MULTI_MAX) {
        groupTopkParam.kInner = param_.n;
    } else {
        groupTopkParam.kInner = 1; // 1:get the max value in each group
    }
    groupTopkNode.opDesc = {0, "GroupTopkOperation", groupTopkParam};
    groupTopkNode.inTensors = {&tokenTensor, &idxArrTensor};
    groupTopkNode.outTensors = {&outTensor};

    return NO_ERROR;
}

void GroupTopkOpsRunner::SetParam(const Mki::Any &param)
{
    infer::GroupTopkParam newParam = Mki::AnyCast<infer::GroupTopkParam>(param);
    if (!(newParam == param_)) {
        ATB_LOG(DEBUG) << GetLogPrefix() << "GroupTopkOpsRunner Param Changed!";
        param_ = newParam;
        isParamUpdated_ = true;
    }
}
} // namespace atb
