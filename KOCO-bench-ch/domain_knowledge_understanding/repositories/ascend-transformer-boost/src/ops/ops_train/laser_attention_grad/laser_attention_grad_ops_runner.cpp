/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "laser_attention_grad_ops_runner.h"
#include <asdops/params/params.h>
#include <atbops/params/params.h>
#include "atb/utils/log.h"

static constexpr size_t SIZE_4 = 4;
static constexpr size_t SIZE_15 = 15;

namespace atb {
LaserAttentionGradOpsRunner::LaserAttentionGradOpsRunner(const train::LaserAttentionGradParam &param)
    : OpsRunner("LaserAttentionGradOpsRunner", RUNNER_TYPE_LASER_ATTENTION_GRAD), param_(param)
{
}

LaserAttentionGradOpsRunner::~LaserAttentionGradOpsRunner() {}

Status LaserAttentionGradOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    kernelGraph_.inTensors.resize(SIZE_15);
    size_t inTensorId = 0;
    Mki::Tensor &queryTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &keyTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &valueTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &attentionOutGradTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &pseShiftTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &dropMaskTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &paddingMaskTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &attenMaskTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &softmaxMaxTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &softmaxSumTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &softmaxInTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &attentionInTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &prefixTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &actualSeqQLenTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &actualSeqKVLenTensor = kernelGraph_.inTensors.at(inTensorId++);

    kernelGraph_.outTensors.resize(SIZE_4);
    size_t outTensorId = 0;
    Mki::Tensor &queryGradTensor = kernelGraph_.outTensors.at(outTensorId++);
    Mki::Tensor &keyGradTensor = kernelGraph_.outTensors.at(outTensorId++);
    Mki::Tensor &valueGradTensor = kernelGraph_.outTensors.at(outTensorId++);
    Mki::Tensor &dpseTensor = kernelGraph_.outTensors.at(outTensorId++);

    kernelGraph_.internalTensors.resize(0);

    kernelGraph_.nodes.resize(1);
    KernelGraphNode &laserAttentionNode = kernelGraph_.nodes.at(0);

    AtbOps::OpParam::LaserAttentionGrad laserAttentionGradParam;
    laserAttentionGradParam.headNum = param_.headNum;
    laserAttentionGradParam.inputLayout = param_.inputLayout;
    laserAttentionGradParam.scaleValue = param_.scaleValue;
    laserAttentionGradParam.keepProb = param_.keepProb;
    laserAttentionGradParam.preTokens = param_.preTokens;
    laserAttentionGradParam.nextTokens = param_.nextTokens;
    laserAttentionGradParam.innerPrecise = param_.innerPrecise;
    laserAttentionNode.opDesc = {0, "LaserAttentionGradOperation", laserAttentionGradParam};
    laserAttentionNode.inTensors = {&queryTensor,      &keyTensor,        &valueTensor,     &attentionOutGradTensor,
                                    &pseShiftTensor,   &dropMaskTensor,   &attenMaskTensor, &softmaxMaxTensor,
                                    &softmaxSumTensor, &attentionInTensor};
    laserAttentionNode.outTensors = {&queryGradTensor, &keyGradTensor, &valueGradTensor};

    (void)opsTensorPack;
    (void)paddingMaskTensor;
    (void)softmaxInTensor;
    (void)prefixTensor;
    (void)actualSeqQLenTensor;
    (void)actualSeqKVLenTensor;
    (void)dpseTensor;
    return NO_ERROR;
}

void LaserAttentionGradOpsRunner::SetParam(const Mki::Any &param)
{
    train::LaserAttentionGradParam newParam = Mki::AnyCast<train::LaserAttentionGradParam>(param);
    if (!(newParam == param_)) {
        ATB_LOG(DEBUG) << GetLogPrefix() << "LaserAttentionGradOpsRunner Param Changed!";
        param_ = newParam;
        isParamUpdated_ = true;
    }
}
} // namespace atb