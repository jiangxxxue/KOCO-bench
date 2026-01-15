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

#include "mm_deq_swiglu_quant_mm_deq_ops_runner.h"

namespace atb {
MmDeqSwigluQuantMmDeqOpsRunner::MmDeqSwigluQuantMmDeqOpsRunner(
    const infer::MmDeqSwigluQuantMmDeqParam &param)
    : OpsRunner("MmDeqSwigluQuantMmDeqOpsRunner", RUNNER_TYPE_MM_DEQ_SWIGLU_QUANT_MM_DEQ), param_(param)
{
}

MmDeqSwigluQuantMmDeqOpsRunner::~MmDeqSwigluQuantMmDeqOpsRunner() {}

void MmDeqSwigluQuantMmDeqOpsRunner::SetParam(const Mki::Any &param)
{
    infer::MmDeqSwigluQuantMmDeqParam newParam = Mki::AnyCast<infer::MmDeqSwigluQuantMmDeqParam>(param);
    if (!(newParam == param_)) {
        ATB_LOG(DEBUG) << GetLogPrefix() << "MmDeqSwigluQuantMmDeqOpsRunner param changed!";
        param_ = newParam;
        isParamUpdated_ = true;
    }
}

AtbOps::OpParam::MmDeqSwigluQuantMmDeq::OutputType GetAtbOpsOutputType(
    infer::MmDeqSwigluQuantMmDeqParam::OutputType outputType)
{
    switch (outputType) {
        case infer::MmDeqSwigluQuantMmDeqParam::OutputType::OUTPUT_FLOAT16:
            return AtbOps::OpParam::MmDeqSwigluQuantMmDeq::OutputType::OUTPUT_FLOAT16;
        default:
            return AtbOps::OpParam::MmDeqSwigluQuantMmDeq::OutputType::OUTPUT_INVALID;
    }
}

AtbOps::OpParam::MmDeqSwigluQuantMmDeq::WeightUpPermuteType GetAtbOpsWeightUpPermuteType(
    infer::MmDeqSwigluQuantMmDeqParam::WeightUpPermuteType weightUpPermuteType)
{
    switch (weightUpPermuteType) {
        case infer::MmDeqSwigluQuantMmDeqParam::WeightUpPermuteType::PERMUTE_N256:
            return AtbOps::OpParam::MmDeqSwigluQuantMmDeq::WeightUpPermuteType::PERMUTE_N256;
        case infer::MmDeqSwigluQuantMmDeqParam::WeightUpPermuteType::PERMUTE_N128:
            return AtbOps::OpParam::MmDeqSwigluQuantMmDeq::WeightUpPermuteType::PERMUTE_N128;
        default:
            return AtbOps::OpParam::MmDeqSwigluQuantMmDeq::WeightUpPermuteType::PERMUTE_INVALID;
    }
}

Status MmDeqSwigluQuantMmDeqOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    (void)opsTensorPack;

    const size_t inTensorNum = 6;
    const size_t outTensorNum = 1;
    kernelGraph_.inTensors.resize(inTensorNum);
    kernelGraph_.outTensors.resize(outTensorNum);

    size_t inTensorId = 0;
    Mki::Tensor &x1 = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &weight1 = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &scale1 = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &perTokenScale1 = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &weight2 = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &scale2 = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &out = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    KernelGraphNode &mmDeqSwigluQuantMmDeqNode = kernelGraph_.nodes.at(0);

    AtbOps::OpParam::MmDeqSwigluQuantMmDeq mmDeqSwigluQuantMmDeqParam;
    mmDeqSwigluQuantMmDeqParam.outputType = GetAtbOpsOutputType(param_.outputType);
    mmDeqSwigluQuantMmDeqParam.weightUpPermuteType = GetAtbOpsWeightUpPermuteType(param_.weightUpPermuteType);
    mmDeqSwigluQuantMmDeqParam.transposeWeightUp = param_.transposeWeightUp;
    mmDeqSwigluQuantMmDeqParam.transposeWeightDown = param_.transposeWeightDown;

    mmDeqSwigluQuantMmDeqNode.opDesc = {0, "MmDeqSwigluQuantMmDeqOperation", mmDeqSwigluQuantMmDeqParam};
    mmDeqSwigluQuantMmDeqNode.inTensors = {&x1, &weight1, &scale1, &perTokenScale1, &weight2, &scale2};
    mmDeqSwigluQuantMmDeqNode.outTensors = {&out};

    return NO_ERROR;
}
} // namespace atb
