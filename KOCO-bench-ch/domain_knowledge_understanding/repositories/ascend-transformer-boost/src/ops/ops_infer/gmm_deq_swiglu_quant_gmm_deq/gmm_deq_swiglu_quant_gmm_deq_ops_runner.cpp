/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <asdops/params/params.h>
#include <atbops/params/params.h>

#include "atb/utils/log.h"

#include "gmm_deq_swiglu_quant_gmm_deq_ops_runner.h"

namespace atb {
GmmDeqSwigluQuantGmmDeqOpsRunner::GmmDeqSwigluQuantGmmDeqOpsRunner(
    const infer::GmmDeqSwigluQuantGmmDeqParam &param)
    : OpsRunner("GmmDeqSwigluQuantGmmDeqOpsRunner", RUNNER_TYPE_GMM_DEQ_SWIGLU_QUANT_GMM_DEQ), param_(param)
{
}

GmmDeqSwigluQuantGmmDeqOpsRunner::~GmmDeqSwigluQuantGmmDeqOpsRunner() {}

void GmmDeqSwigluQuantGmmDeqOpsRunner::SetParam(const Mki::Any &param)
{
    infer::GmmDeqSwigluQuantGmmDeqParam newParam = Mki::AnyCast<infer::GmmDeqSwigluQuantGmmDeqParam>(param);
    if (!(newParam == param_)) {
        ATB_LOG(DEBUG) << GetLogPrefix() << "GmmDeqSwigluQuantGmmDeqOpsRunner param changed!";
        param_ = newParam;
        isParamUpdated_ = true;
    }
}

AtbOps::OpParam::GmmDeqSwigluQuantGmmDeq::OutputType GetAtbOpsOutputType(
    infer::GmmDeqSwigluQuantGmmDeqParam::OutputType outputType)
{
    switch (outputType) {
        case infer::GmmDeqSwigluQuantGmmDeqParam::OutputType::OUTPUT_FLOAT16:
            return AtbOps::OpParam::GmmDeqSwigluQuantGmmDeq::OutputType::OUTPUT_FLOAT16;
        default:
            return AtbOps::OpParam::GmmDeqSwigluQuantGmmDeq::OutputType::OUTPUT_INVALID;
    }
}

AtbOps::OpParam::GmmDeqSwigluQuantGmmDeq::GroupListType GetAtbOpsGroupListType(
    infer::GmmDeqSwigluQuantGmmDeqParam::GroupListType groupListType)
{
    switch (groupListType) {
        case infer::GmmDeqSwigluQuantGmmDeqParam::GroupListType::GROUP_LIST_CUMSUM:
            return AtbOps::OpParam::GmmDeqSwigluQuantGmmDeq::GroupListType::GROUP_LIST_CUMSUM;
        default:
            return AtbOps::OpParam::GmmDeqSwigluQuantGmmDeq::GroupListType::GROUP_LIST_INVALID;
    }
}

AtbOps::OpParam::GmmDeqSwigluQuantGmmDeq::WeightUpPermuteType GetAtbOpsWeightUpPermuteType(
    infer::GmmDeqSwigluQuantGmmDeqParam::WeightUpPermuteType weightUpPermuteType)
{
    switch (weightUpPermuteType) {
        case infer::GmmDeqSwigluQuantGmmDeqParam::WeightUpPermuteType::PERMUTE_N256:
            return AtbOps::OpParam::GmmDeqSwigluQuantGmmDeq::WeightUpPermuteType::PERMUTE_N256;
        case infer::GmmDeqSwigluQuantGmmDeqParam::WeightUpPermuteType::PERMUTE_N128:
            return AtbOps::OpParam::GmmDeqSwigluQuantGmmDeq::WeightUpPermuteType::PERMUTE_N128;
        default:
            return AtbOps::OpParam::GmmDeqSwigluQuantGmmDeq::WeightUpPermuteType::PERMUTE_INVALID;
    }
}

Status GmmDeqSwigluQuantGmmDeqOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    (void)opsTensorPack;

    const size_t inTensorNum = 7;
    const size_t outTensorNum = 1;
    kernelGraph_.inTensors.resize(inTensorNum);
    kernelGraph_.outTensors.resize(outTensorNum);

    size_t inTensorId = 0;
    Mki::Tensor &x1 = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &weight1 = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &scale1 = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &perTokenScale1 = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &groupList = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &weight2 = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &scale2 = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &out = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    KernelGraphNode &gmmDeqSwigluQuantGmmDeqNode = kernelGraph_.nodes.at(0);

    AtbOps::OpParam::GmmDeqSwigluQuantGmmDeq gmmDeqSwigluQuantGmmDeqParam;
    gmmDeqSwigluQuantGmmDeqParam.outputType = GetAtbOpsOutputType(param_.outputType);
    gmmDeqSwigluQuantGmmDeqParam.groupListType = GetAtbOpsGroupListType(param_.groupListType);
    gmmDeqSwigluQuantGmmDeqParam.weightUpPermuteType = GetAtbOpsWeightUpPermuteType(param_.weightUpPermuteType);
    gmmDeqSwigluQuantGmmDeqParam.transposeWeightUp = param_.transposeWeightUp;
    gmmDeqSwigluQuantGmmDeqParam.transposeWeightDown = param_.transposeWeightDown;

    gmmDeqSwigluQuantGmmDeqNode.opDesc = {0, "GmmDeqSwigluQuantGmmDeqOperation", gmmDeqSwigluQuantGmmDeqParam};
    gmmDeqSwigluQuantGmmDeqNode.inTensors = {&x1, &weight1, &scale1, &perTokenScale1, &groupList,
        &weight2, &scale2};
    gmmDeqSwigluQuantGmmDeqNode.outTensors = {&out};

    return NO_ERROR;
}
} // namespace atb
