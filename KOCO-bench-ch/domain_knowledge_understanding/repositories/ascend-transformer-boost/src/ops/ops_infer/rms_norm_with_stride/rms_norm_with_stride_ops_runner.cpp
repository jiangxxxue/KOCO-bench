/*
 * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rms_norm_with_stride_ops_runner.h"
#include "atb/utils/log.h"

namespace atb {
static const uint64_t IN_TENSOR_COUNT_TWO = 2;
static const uint64_t IN_TENSOR_COUNT_THREE = 3;
static const uint64_t IN_TENSOR_COUNT_FOUR = 4;
static const uint64_t IN_TENSOR_COUNT_FIVE = 5;
static const uint64_t IN_TENSOR_COUNT_SIX = 6;
static const uint64_t OUT_TENSOR_COUNT_ONE = 1;
static const uint64_t OUT_TENSOR_COUNT_TWO = 2;
static const uint64_t OUT_TENSOR_COUNT_THREE = 3;
static const uint32_t GEMMA_MODE = 1;

void RmsNormWithStrideOpsRunner::SetRmsNormParam(const infer::RmsNormWithStrideParam &inferParam,
                                                 AsdOps::OpParam::Norm &asdopsParam) const
{
    asdopsParam.inGamma = true;
    asdopsParam.epsilon = inferParam.normParam.epsilon;
    asdopsParam.precisionMode = inferParam.normParam.precisionMode;
    if (inferParam.normParam.modelType == infer::RmsNormWithStrideParam::GEMMA_MODEL) {
        asdopsParam.gemmaMode = GEMMA_MODE;
    }
}

void RmsNormWithStrideOpsRunner::BuildRmsNormGraph(const AsdOps::OpParam::Norm &rmsNormParam)
{
    // 4 in -> 1 out, RmsNorm
    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT_FOUR);
    size_t inId = 0;
    Mki::Tensor &inputXTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &gammaTensor = kernelGraph_.inTensors.at(inId++);

    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT_ONE);
    size_t outId = 0;
    Mki::Tensor &resultTensor = kernelGraph_.outTensors.at(outId++);

    kernelGraph_.nodes.resize(1);
    auto &rmsNormNode = kernelGraph_.nodes.at(0);
    rmsNormNode.opDesc = {0, "NormOperation", rmsNormParam};
    rmsNormNode.inTensors = {&inputXTensor, &gammaTensor};
    rmsNormNode.outTensors = {&resultTensor};
}

RmsNormWithStrideOpsRunner::RmsNormWithStrideOpsRunner(const infer::RmsNormWithStrideParam &param)
    : OpsRunner("RmsNormWithStrideOpsRunner", RUNNER_TYPE_RMS_NORM_WITH_STRIDE), param_(param)
{
    AsdOps::OpParam::Norm rmsNormParam = {AsdOps::OpParam::Norm::RMS_NORM};
    rmsNormParam.normType = AsdOps::OpParam::Norm::RMS_NORM;
    SetRmsNormParam(param_, rmsNormParam);
    BuildRmsNormGraph(rmsNormParam);
    ATB_LOG(INFO) << GetName() << " NormOperation opDesc normParam beginNormAxis:" << rmsNormParam.beginNormAxis
                  << ", beginParamsAxis:" << rmsNormParam.beginParamsAxis << ", epsilon:" << rmsNormParam.epsilon
                  << ", zoomScaleValue:" << rmsNormParam.zoomScaleValue << ", inGamma:" << rmsNormParam.inGamma
                  << ", inBeta:" << rmsNormParam.inBeta << ", inRes:" << rmsNormParam.inRes
                  << ", inNormBias:" << rmsNormParam.inNormBias << ", outMean:" << rmsNormParam.outMean
                  << ", outVarience:" << rmsNormParam.outVarience << ", outResQuant:" << rmsNormParam.outResQuant
                  << ", outRes:" << rmsNormParam.outRes;
}

Status RmsNormWithStrideOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    Mki::Tensor &tensor = kernelGraph_.inTensors.at(0);
    size_t strideSize = static_cast<size_t>(opsTensorPack.inTensors.at(IN_TENSOR_COUNT_TWO).Numel());
    tensor.desc.strides.resize(strideSize);
    for (size_t i = 0; i < strideSize; ++i) {
        tensor.desc.strides.at(i) = static_cast<int64_t *>(opsTensorPack.inTensors.at(IN_TENSOR_COUNT_TWO).hostData)[i];
    }
    tensor.desc.offset = static_cast<int64_t *>(opsTensorPack.inTensors.at(IN_TENSOR_COUNT_THREE).hostData)[0];
    ATB_LOG(INFO) << "" << tensor.desc.offset;
    return NO_ERROR;
}

RmsNormWithStrideOpsRunner::~RmsNormWithStrideOpsRunner() {}
} // namespace atb