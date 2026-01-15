/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "layer_norm_with_stride_ops_runner.h"
#include "atb/utils/log.h"

namespace atb {
static const uint64_t IN_TENSOR_COUNT_SIX = 6;
static const uint64_t IN_TENSOR_COUNT_FIVE = 5;
static const uint64_t IN_TENSOR_COUNT_FOUR = 4;
static const uint64_t IN_TENSOR_COUNT_THREE = 3;
static const uint64_t OUT_TENSOR_COUNT_TWO = 2;
static const uint64_t OUT_TENSOR_COUNT_ONE = 1;

void LayerNormWithStrideOpsRunner::SetLayerNormParam(const infer::LayerNormWithStrideParam &inferParam,
                                                     AsdOps::OpParam::Norm &asdopsParam) const
{
    asdopsParam.inGamma = true;
    asdopsParam.inBeta = true;
    asdopsParam.outMean = false;
    asdopsParam.outVarience = false;
    asdopsParam.epsilon = inferParam.normParam.epsilon;
    asdopsParam.beginNormAxis = inferParam.normParam.beginNormAxis;
    asdopsParam.beginParamsAxis = inferParam.normParam.beginParamsAxis;
}

void LayerNormWithStrideOpsRunner::BuildLayerNormGraph(const AsdOps::OpParam::Norm &layerNormParam)
{
    // 3 in -> 3 out, layerNorm, any dim
    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT_THREE);
    int inId = 0;
    Mki::Tensor &inputXTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &gammaTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &betaTensor = kernelGraph_.inTensors.at(inId++);

    kernelGraph_.outTensors.resize(1);
    Mki::Tensor &resultTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.internalTensors.resize(2); // 承载2个无用输出
    kernelGraph_.nodes.resize(1);
    auto &layerNormNode = kernelGraph_.nodes[0];
    layerNormNode.opDesc = {0, "NormOperation", layerNormParam};
    layerNormNode.inTensors = {&inputXTensor, &gammaTensor, &betaTensor};
    layerNormNode.outTensors = {&resultTensor};
}

LayerNormWithStrideOpsRunner::LayerNormWithStrideOpsRunner(const infer::LayerNormWithStrideParam &param)
    : OpsRunner("LayerNormWithStrideOpsRunner", RUNNER_TYPE_LAYER_NORM_WITH_STRIDE), param_(param)
{
    AsdOps::OpParam::Norm layerNormParam = {AsdOps::OpParam::Norm::LAYER_NORM};
    SetLayerNormParam(param_, layerNormParam);
    BuildLayerNormGraph(layerNormParam);
    ATB_LOG(INFO) << GetName() << " NormOperation opDesc normParam beginNormAxis:" << layerNormParam.beginNormAxis
                  << ", beginParamsAxis:" << layerNormParam.beginParamsAxis << ", epsilon:" << layerNormParam.epsilon
                  << ", zoomScaleValue:" << layerNormParam.zoomScaleValue << ", inGamma:" << layerNormParam.inGamma
                  << ", inBeta:" << layerNormParam.inBeta << ", inRes:" << layerNormParam.inRes
                  << ", inNormBias:" << layerNormParam.inNormBias << ", outMean:" << layerNormParam.outMean
                  << ", outVarience:" << layerNormParam.outVarience << ", outResQuant:" << layerNormParam.outResQuant
                  << ", outRes:" << layerNormParam.outRes;
}

Status LayerNormWithStrideOpsRunner::SetupKernelGraph(const OpsTensorPack &opsTensorPack)
{
    Mki::Tensor &tensor = kernelGraph_.inTensors.at(0);
    size_t strideSize = static_cast<size_t>(opsTensorPack.inTensors.at(IN_TENSOR_COUNT_THREE).Numel());
    tensor.desc.strides.resize(strideSize);
    for (size_t i = 0; i < strideSize; ++i) {
        tensor.desc.strides.at(i) =
            static_cast<int64_t *>(opsTensorPack.inTensors.at(IN_TENSOR_COUNT_THREE).hostData)[i];
    }
    tensor.desc.offset = static_cast<int64_t *>(opsTensorPack.inTensors.at(IN_TENSOR_COUNT_FOUR).hostData)[0];
    return NO_ERROR;
}

LayerNormWithStrideOpsRunner::~LayerNormWithStrideOpsRunner() {}
} // namespace atb