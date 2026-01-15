/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rope_ops_runner.h"
#include <atbops/params/params.h>
#include "atb/utils/log.h"
#include "atb/utils/tensor_util.h"

namespace atb {
static const uint64_t IN_TENSOR_COUNT = 5;
static const uint64_t OUT_TENSOR_COUNT = 2;

void RopeKqView(const Mki::SVector<int64_t> &oldDims, Mki::SVector<int64_t> &newDims)
{
    if (oldDims.size() == 4) { // 4: 判断原张量的维数
        newDims = {oldDims.at(0) * oldDims.at(1), oldDims.at(2) * oldDims.at(3)};
    } else if (oldDims.size() == 2) { // 2: 判断原张量的维数
        newDims = oldDims;
    } else {
        ATB_LOG(ERROR) << "rope intensor qLayer or kLayer dimNum need 2 or 4";
    }
}

RopeOpsRunner::RopeOpsRunner(const infer::RopeParam &param)
    : OpsRunner("RopeOpsRunner", RUNNER_TYPE_ROPE), param_(param)
{
}

RopeOpsRunner::~RopeOpsRunner() {}

Status RopeOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    (void)opsTensorPack;
    ATB_LOG(INFO) << "RopeOpsRunner::RopeOpsRunner called";
    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT);
    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT);

    int64_t inTensorNum = 0;
    Mki::Tensor &qLayer = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &kLayer = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &cos = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &sin = kernelGraph_.inTensors.at(inTensorNum++);
    Mki::Tensor &seqLen = kernelGraph_.inTensors.at(inTensorNum++);

    int64_t outTensorNum = 0;
    Mki::Tensor &qEmbedded = kernelGraph_.outTensors.at(outTensorNum++);
    Mki::Tensor &kEmbedded = kernelGraph_.outTensors.at(outTensorNum++);

    kernelGraph_.nodes.resize(1);
    auto &ropeNode = kernelGraph_.nodes.at(0);

    AtbOps::OpParam::Rope ropeParam;
    ropeParam.rotaryCoeff = param_.rotaryCoeff;
    ropeParam.cosFormat = param_.cosFormat;
    ropeNode.opDesc = {0, "RopeOperation", ropeParam};
    ropeNode.inTensors = {&qLayer, &kLayer, &cos, &sin, &seqLen};
    ropeNode.outTensors = {&qEmbedded, &kEmbedded};
    ropeNode.inTensorViewFuncs = {&RopeKqView, &RopeKqView};
    ropeNode.mkiInferShapePreFunc = [](Mki::LaunchParam &launchParam) {
        launchParam.GetInTensor(4).desc.dtype = Mki::TENSOR_DTYPE_UINT32; // 4 ： 设置第五个输入张量的dtype
    };

    return NO_ERROR;
}

void RopeOpsRunner::SetParam(const Mki::Any &param)
{
    infer::RopeParam newParam = Mki::AnyCast<infer::RopeParam>(param);
    if (!(newParam == param_)) {
        ATB_LOG(DEBUG) << GetLogPrefix() << "RopeOpsRunner Param Changed!";
        param_ = newParam;
        isParamUpdated_ = true;
    }
}
} // namespace atb