/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "laser_attention_ops_runner.h"
#include <asdops/params/params.h>
#include <atbops/params/params.h>
#include "atb/utils/log.h"

static constexpr size_t SIZE_4 = 4;
static constexpr size_t SIZE_10 = 10;

namespace atb {
LaserAttentionOpsRunner::LaserAttentionOpsRunner(const train::LaserAttentionParam &param)
    : OpsRunner("LaserAttentionOpsRunner", RUNNER_TYPE_LASER_ATTENTION), param_(param)
{
}

LaserAttentionOpsRunner::~LaserAttentionOpsRunner() {}

Status LaserAttentionOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    kernelGraph_.inTensors.resize(SIZE_10);
    size_t inTensorId = 0;
    Mki::Tensor &queryTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &keyTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &valueTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &pseShiftTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &dropMaskTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &paddingMaskTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &attenMaskTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &prefixTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &actualSeqQLenTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &actualSeqKVLenTensor = kernelGraph_.inTensors.at(inTensorId++);

    kernelGraph_.outTensors.resize(SIZE_4);
    size_t outTensorId = 0;
    Mki::Tensor &softmaxMaxTensor = kernelGraph_.outTensors.at(outTensorId++);
    Mki::Tensor &softmaxSumTensor = kernelGraph_.outTensors.at(outTensorId++);
    Mki::Tensor &softmaxOutTensor = kernelGraph_.outTensors.at(outTensorId++);
    Mki::Tensor &attentionOutTensor = kernelGraph_.outTensors.at(outTensorId++);

    kernelGraph_.internalTensors.resize(0);

    kernelGraph_.nodes.resize(1);
    KernelGraphNode &laserAttentionNode = kernelGraph_.nodes.at(0);

    AtbOps::OpParam::LaserAttention laserAttentionParam;
    laserAttentionParam.headNum = param_.headNum;
    laserAttentionParam.inputLayout = param_.inputLayout;
    laserAttentionParam.scaleValue = param_.scaleValue;
    laserAttentionParam.keepProb = param_.keepProb;
    laserAttentionParam.preTokens = param_.preTokens;
    laserAttentionParam.nextTokens = param_.nextTokens;
    laserAttentionParam.innerPrecise = param_.innerPrecise;
    laserAttentionNode.opDesc = {0, "LaserAttentionOperation", laserAttentionParam};
    laserAttentionNode.inTensors = {&queryTensor,    &keyTensor,      &valueTensor,
                                    &pseShiftTensor, &dropMaskTensor, &attenMaskTensor};
    laserAttentionNode.outTensors = {&softmaxMaxTensor, &softmaxSumTensor, &attentionOutTensor};

    (void)opsTensorPack;
    (void)paddingMaskTensor;
    (void)prefixTensor;
    (void)actualSeqQLenTensor;
    (void)actualSeqKVLenTensor;
    (void)softmaxOutTensor;
    return NO_ERROR;
}

void LaserAttentionOpsRunner::SetParam(const Mki::Any &param)
{
    train::LaserAttentionParam newParam = Mki::AnyCast<train::LaserAttentionParam>(param);
    if (!(newParam == param_)) {
        ATB_LOG(DEBUG) << GetLogPrefix() << "LaserAttentionOpsRunner Param Changed!";
        param_ = newParam;
        isParamUpdated_ = true;
    }
}
} // namespace atb