/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "layer_norm_ops_runner.h"
#include "atb/utils/log.h"

namespace atb {
static const uint64_t IN_TENSOR_COUNT_SIX = 6;
static const uint64_t IN_TENSOR_COUNT_FIVE = 5;
static const uint64_t IN_TENSOR_COUNT_FOUR = 4;
static const uint64_t IN_TENSOR_COUNT_THREE = 3;
static const uint64_t OUT_TENSOR_COUNT_TWO = 2;
static const uint64_t OUT_TENSOR_COUNT_ONE = 1;

void LayerNormOpsRunner::SetLayerNormParam(const infer::LayerNormParam &inferParam,
                                           AsdOps::OpParam::Norm &asdopsParam) const
{
    asdopsParam.inGamma = true;
    asdopsParam.inBeta = true;
    asdopsParam.outMean = true;
    asdopsParam.outVarience = true;
    asdopsParam.epsilon = inferParam.normParam.epsilon;
    asdopsParam.beginNormAxis = inferParam.normParam.beginNormAxis;
    asdopsParam.beginParamsAxis = inferParam.normParam.beginParamsAxis;
}

void LayerNormOpsRunner::SetLayerNormQuantParam(const infer::LayerNormParam &inferParam,
                                                AsdOps::OpParam::Norm &asdopsParam) const
{
    asdopsParam.inGamma = true;
    asdopsParam.inBeta = true;
    asdopsParam.outResQuant = true;
    asdopsParam.epsilon = inferParam.normParam.epsilon;
    asdopsParam.isDynamicQuant = inferParam.normParam.dynamicQuantType != infer::DYNAMIC_QUANT_UNDEFINED;
    asdopsParam.isSymmetric = inferParam.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_SYMMETRIC;
}

void LayerNormOpsRunner::SetPreLayerNormParam(const infer::LayerNormParam &inferParam,
                                              AsdOps::OpParam::Norm &asdopsParam) const
{
    asdopsParam.inGamma = true;
    asdopsParam.inBeta = true;
    asdopsParam.inRes = true;
    asdopsParam.outRes = true;
    asdopsParam.epsilon = inferParam.preNormParam.epsilon;
    asdopsParam.opsMode = inferParam.preNormParam.opMode;
    asdopsParam.zoomScaleValue = inferParam.preNormParam.zoomScaleValue;
}

void LayerNormOpsRunner::SetPostLayerNormParam(const infer::LayerNormParam &inferParam,
                                               AsdOps::OpParam::Norm &asdopsParam) const
{
    asdopsParam.inGamma = true;
    asdopsParam.inBeta = true;
    asdopsParam.inRes = true;
    asdopsParam.epsilon = inferParam.postNormParam.epsilon;
    asdopsParam.opsMode = inferParam.postNormParam.opMode;
    asdopsParam.zoomScaleValue = inferParam.postNormParam.zoomScaleValue;
}

void LayerNormOpsRunner::SetPostLayerNormQuantParam(const infer::LayerNormParam &inferParam,
                                                    AsdOps::OpParam::Norm &asdopsParam) const
{
    asdopsParam.inGamma = true;
    asdopsParam.inBeta = true;
    asdopsParam.inRes = true;
    asdopsParam.outResQuant = true;
    asdopsParam.epsilon = inferParam.postNormParam.epsilon;
    asdopsParam.opsMode = inferParam.postNormParam.opMode;
}

void LayerNormOpsRunner::BuildLayerNormGraph(const AsdOps::OpParam::Norm &layerNormParam)
{
    // 3 in -> 3 out, layerNorm, any dim
    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT_THREE);
    int inId = 0;
    Mki::Tensor &inputXTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &gammaTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &betaTensor = kernelGraph_.inTensors.at(inId++);

    kernelGraph_.outTensors.resize(1);
    Mki::Tensor &resultTensor = kernelGraph_.outTensors.at(0);

    int interId = 0;
    kernelGraph_.internalTensors.resize(2); // 承载2个无用输出
    Mki::Tensor &meanTensor = kernelGraph_.internalTensors.at(interId++);
    Mki::Tensor &variencetTensor = kernelGraph_.internalTensors.at(interId++);

    kernelGraph_.nodes.resize(1);
    auto &layerNormNode = kernelGraph_.nodes[0];
    layerNormNode.opDesc = {0, "NormOperation", layerNormParam};
    layerNormNode.inTensors = {&inputXTensor, &gammaTensor, &betaTensor};
    layerNormNode.outTensors = {&resultTensor, &meanTensor, &variencetTensor};
}

void LayerNormOpsRunner::BuildLayerNormQuantGraph(const AsdOps::OpParam::Norm &layerNormParam)
{
    // 3 in -> 2 out, layerNormQuant, dim -1
    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT_FIVE);
    int inId = 0;
    Mki::Tensor &inputXTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &gammaTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &betaTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &scaleTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &offsetTensor = kernelGraph_.inTensors.at(inId++);

    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT_ONE);
    int outId = 0;
    Mki::Tensor &resQuant = kernelGraph_.outTensors.at(outId++);

    kernelGraph_.nodes.resize(1);
    auto &layerNormNode = kernelGraph_.nodes[0];
    layerNormNode.opDesc = {0, "NormOperation", layerNormParam};
    layerNormNode.inTensors = {&inputXTensor, &gammaTensor, &betaTensor, &scaleTensor, &offsetTensor};
    layerNormNode.outTensors = {&resQuant};
}

void LayerNormOpsRunner::BuildLayerNormDynamicQuantGraph(const AsdOps::OpParam::Norm &layerNormParam)
{
    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT_THREE);
    size_t inTensorId = 0;
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &gammaTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &betaTensor = kernelGraph_.inTensors.at(inTensorId++);

    kernelGraph_.outTensors.resize(IN_TENSOR_COUNT_THREE);
    size_t outTensorId = 0;
    Mki::Tensor &resQuant = kernelGraph_.outTensors.at(outTensorId++);
    Mki::Tensor &scaleTensor = kernelGraph_.outTensors.at(outTensorId++);
    Mki::Tensor &offsetTensor = param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_SYMMETRIC ?
                                    nullTensor_ :
                                    kernelGraph_.outTensors.at(outTensorId++);

    kernelGraph_.nodes.resize(1);
    KernelGraphNode &layerNormNode = kernelGraph_.nodes.at(0);
    layerNormNode.opDesc = {0, "NormOperation", layerNormParam};
    layerNormNode.inTensors = {&xTensor, &gammaTensor, &betaTensor};
    layerNormNode.outTensors = {&resQuant, &scaleTensor, &offsetTensor};
}

void LayerNormOpsRunner::BuildPreLayerNormGraph(const AsdOps::OpParam::Norm &layerNormParam)
{
    // 4 in -> 2 out, preLayerNorm, dim -1
    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT_FOUR);
    int inId = 0;
    Mki::Tensor &inputXTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &residualAddIn = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &gammaTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &betaTensor = kernelGraph_.inTensors.at(inId++);

    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT_TWO);
    int outId = 0;
    Mki::Tensor &resultZTensor = kernelGraph_.outTensors.at(outId++);
    Mki::Tensor &resultTensor = kernelGraph_.outTensors.at(outId++);

    kernelGraph_.nodes.resize(1);
    auto &layerNormNode = kernelGraph_.nodes[0];
    layerNormNode.opDesc = {0, "NormOperation", layerNormParam};
    layerNormNode.inTensors = {&inputXTensor, &residualAddIn, &gammaTensor, &betaTensor};
    layerNormNode.outTensors = {&resultZTensor, &resultTensor};
}

void LayerNormOpsRunner::BuildPostLayerNormGraph(const AsdOps::OpParam::Norm &layerNormParam)
{
    // 4 in -> 1 out, postLayerNorm, dim -1
    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT_FOUR);
    int inId = 0;
    Mki::Tensor &inputXTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &residualAddIn = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &gammaTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &betaTensor = kernelGraph_.inTensors.at(inId++);

    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT_ONE);
    int outId = 0;
    Mki::Tensor &resultTensor = kernelGraph_.outTensors.at(outId++);

    kernelGraph_.nodes.resize(1);
    auto &layerNormNode = kernelGraph_.nodes[0];
    layerNormNode.opDesc = {0, "NormOperation", layerNormParam};
    layerNormNode.inTensors = {&inputXTensor, &residualAddIn, &gammaTensor, &betaTensor};
    layerNormNode.outTensors = {&resultTensor};
}

void LayerNormOpsRunner::BuildPostLayerNormQuantGraph(const AsdOps::OpParam::Norm &layerNormParam)
{
    // 6 in -> 2 out, postLayerNormQuant, dim -1
    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT_SIX);
    int inId = 0;
    Mki::Tensor &inputXTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &residualAddIn = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &gammaTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &betaTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &scaleTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &offsetTensor = kernelGraph_.inTensors.at(inId++);

    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT_TWO);
    int outId = 0;
    Mki::Tensor &resultTensor = kernelGraph_.outTensors.at(outId++);
    Mki::Tensor &resQuant = kernelGraph_.outTensors.at(outId++);

    kernelGraph_.nodes.resize(1);
    auto &layerNormNode = kernelGraph_.nodes[0];
    layerNormNode.opDesc = {0, "NormOperation", layerNormParam};
    layerNormNode.inTensors = {&inputXTensor, &gammaTensor, &betaTensor, &residualAddIn, &scaleTensor, &offsetTensor};
    layerNormNode.outTensors = {&resQuant, &resultTensor};
}

LayerNormOpsRunner::LayerNormOpsRunner(const infer::LayerNormParam &param)
    : OpsRunner("LayerNormOpsRunner", RUNNER_TYPE_LAYER_NORM), param_(param)
{
    AsdOps::OpParam::Norm layerNormParam = {AsdOps::OpParam::Norm::LAYER_NORM};
    if (param_.layerType == infer::LayerNormParam::LAYER_NORM_NORM) {
        if (param_.normParam.quantType == infer::QUANT_UNQUANT) {
            SetLayerNormParam(param_, layerNormParam);
            BuildLayerNormGraph(layerNormParam);
        } else {
            SetLayerNormQuantParam(param_, layerNormParam);
            if (param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_UNDEFINED) {
                BuildLayerNormQuantGraph(layerNormParam);
            } else {
                BuildLayerNormDynamicQuantGraph(layerNormParam);
            }
        }
    } else if (param_.layerType == infer::LayerNormParam::LAYER_NORM_PRENORM) {
        if (param_.preNormParam.quantType == infer::QUANT_UNQUANT) {
            SetPreLayerNormParam(param_, layerNormParam);
            BuildPreLayerNormGraph(layerNormParam);
        } else {
            ATB_LOG(ERROR) << "PreLayerNorm does not support quant yet";
        }
    } else if (param_.layerType == infer::LayerNormParam::LAYER_NORM_POSTNORM) {
        if (param_.postNormParam.quantType == infer::QUANT_UNQUANT) {
            SetPostLayerNormParam(param_, layerNormParam);
            BuildPostLayerNormGraph(layerNormParam);
        } else {
            SetPostLayerNormQuantParam(param_, layerNormParam);
            BuildPostLayerNormQuantGraph(layerNormParam);
        }
    } else {
        ATB_LOG(ERROR) << "LayerNormType Undefined";
    }
    ATB_LOG(INFO) << GetName() << " NormOperation opDesc normParam beginNormAxis:" << layerNormParam.beginNormAxis
                  << ", beginParamsAxis:" << layerNormParam.beginParamsAxis << ", epsilon:" << layerNormParam.epsilon
                  << ", zoomScaleValue:" << layerNormParam.zoomScaleValue << ", inGamma:" << layerNormParam.inGamma
                  << ", inBeta:" << layerNormParam.inBeta << ", inRes:" << layerNormParam.inRes
                  << ", inNormBias:" << layerNormParam.inNormBias << ", outMean:" << layerNormParam.outMean
                  << ", outVarience:" << layerNormParam.outVarience << ", outResQuant:" << layerNormParam.outResQuant
                  << ", outRes:" << layerNormParam.outRes;
}

LayerNormOpsRunner::~LayerNormOpsRunner() {}
} // namespace atb