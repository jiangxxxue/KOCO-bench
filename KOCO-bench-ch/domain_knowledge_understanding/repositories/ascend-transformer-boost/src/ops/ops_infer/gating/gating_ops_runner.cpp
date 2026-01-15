/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gating_ops_runner.h"
#include <atbops/params/params.h>
#include "atb/utils/log.h"

namespace atb {
GatingOpsRunner::GatingOpsRunner(const infer::GatingParam &param)
    : OpsRunner("GatingOpsRunner", RUNNER_TYPE_GATING), param_(param)
{
    ATB_LOG(INFO) << GetLogPrefix() << "GatingOpsRunner::GatingOpsRunner";

    gatingParam_.headNum = param_.topkExpertNum;
    gatingParam_.headSize = param_.cumSumNum;
    gatingParam_.cumSumInt64 = param_.cumSumInt64;
    gatingParam_.deviceExpert = param_.deviceExpert;
}

GatingOpsRunner::~GatingOpsRunner() {}

Status GatingOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    (void)opsTensorPack;

    const size_t inTensorNum = 2;
    const size_t nodeNum = 1;
    kernelGraph_.inTensors.resize(inTensorNum);
    kernelGraph_.nodes.resize(nodeNum);

    if (param_.deviceExpert.empty()) {
        size_t outTensorNum = 3;
        kernelGraph_.outTensors.resize(outTensorNum);
        return SetupKernelGraphGating();
    } else {
        size_t outTensorNum = 4;
        kernelGraph_.outTensors.resize(outTensorNum);
        return SetupKernelGraphGatingExpertParallelism();
    }
}

Status GatingOpsRunner::SetupKernelGraphGating()
{
    ATB_LOG(INFO) << GetLogPrefix() << "GatingOpsRunner::SetupKernelGraphGating";

    size_t inTensorId = 0;
    Mki::Tensor &topKTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &idxArrTensor = kernelGraph_.inTensors.at(inTensorId++);

    size_t outTensorId = 0;
    Mki::Tensor &tokenIndexTensor = kernelGraph_.outTensors.at(outTensorId++);
    Mki::Tensor &cumSumTensor = kernelGraph_.outTensors.at(outTensorId++);
    Mki::Tensor &originalIndexTensor = kernelGraph_.outTensors.at(outTensorId++);

    size_t nodeId = 0;
    auto &gatingNode = kernelGraph_.nodes.at(nodeId++);

    gatingNode.opDesc = {0, "GatingOperation", gatingParam_};
    gatingNode.inTensors = {&topKTensor, &idxArrTensor};
    gatingNode.outTensors = {&tokenIndexTensor, &cumSumTensor, &originalIndexTensor, &nullTensor_};

    return NO_ERROR;
}

Status GatingOpsRunner::SetupKernelGraphGatingExpertParallelism()
{
    ATB_LOG(INFO) << GetLogPrefix() << "GatingOpsRunner::SetupKernelGraphGatingExpertParallelism";

    size_t inTensorId = 0;
    Mki::Tensor &topKTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &idxArrTensor = kernelGraph_.inTensors.at(inTensorId++);

    size_t outTensorId = 0;
    Mki::Tensor &tokenIndexTensor = kernelGraph_.outTensors.at(outTensorId++);
    Mki::Tensor &cumSumTensor = kernelGraph_.outTensors.at(outTensorId++);
    Mki::Tensor &originalIndexTensor = kernelGraph_.outTensors.at(outTensorId++);
    Mki::Tensor &validIndexTensor = kernelGraph_.outTensors.at(outTensorId++);

    size_t nodeId = 0;
    auto &gatingNode = kernelGraph_.nodes.at(nodeId++);

    gatingNode.opDesc = {0, "GatingOperation", gatingParam_};
    gatingNode.inTensors = {&topKTensor, &idxArrTensor};
    gatingNode.outTensors = {&tokenIndexTensor, &cumSumTensor, &originalIndexTensor, &validIndexTensor};

    return NO_ERROR;
}
} // namespace atb