/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gather_pre_rms_norm_ops_runner.h"
#include "atb/utils/log.h"

namespace atb {
static const uint64_t IN_TENSOR_COUNT_FOUR = 4;

void GatherPreRmsNormOpsRunner::SetGatherPreRMSNormParam(const infer::GatherPreRmsNormParam &inferParam,
                                                         AsdOps::OpParam::Norm &asdopsParam) const
{
    asdopsParam.normType = AsdOps::OpParam::Norm::GATHER_PRE_RMS_NORM;
    asdopsParam.epsilon = inferParam.epsilon;
}

void GatherPreRmsNormOpsRunner::BuildGatherPreRMSNormGraph(const AsdOps::OpParam::Norm &rmsNormParam)
{
    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT_FOUR);
    size_t inId = 0;
    Mki::Tensor &inputXTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &resInTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &indicesTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &gammaTensor = kernelGraph_.inTensors.at(inId++);

    kernelGraph_.outTensors.resize(2); // 2 2 outtensors
    size_t outId = 0;
    Mki::Tensor &outYTensor = kernelGraph_.outTensors.at(outId++);
    Mki::Tensor &resOutTensor = kernelGraph_.outTensors.at(outId++);

    kernelGraph_.nodes.resize(1);
    auto &rmsNormNode = kernelGraph_.nodes.at(0);
    rmsNormNode.opDesc = {0, "NormOperation", rmsNormParam};
    rmsNormNode.inTensors = {&inputXTensor, &resInTensor, &indicesTensor, &gammaTensor};
    rmsNormNode.outTensors = {&outYTensor, &resOutTensor};
}

GatherPreRmsNormOpsRunner::GatherPreRmsNormOpsRunner(const infer::GatherPreRmsNormParam &param)
    : OpsRunner("GatherPreRmsNormOpsRunner", RUNNER_TYPE_GATHER_PRE_RMS_NORM), param_(param)
{
    AsdOps::OpParam::Norm rmsNormParam = {AsdOps::OpParam::Norm::RMS_NORM};
    SetGatherPreRMSNormParam(param_, rmsNormParam);
    BuildGatherPreRMSNormGraph(rmsNormParam);
    ATB_LOG(INFO) << GetName() << " GatherPreRmsNormOperation opDesc GatherPreRmsNormParam epsilon:"
                  << rmsNormParam.epsilon;
}

GatherPreRmsNormOpsRunner::~GatherPreRmsNormOpsRunner() {}
} // namespace atb