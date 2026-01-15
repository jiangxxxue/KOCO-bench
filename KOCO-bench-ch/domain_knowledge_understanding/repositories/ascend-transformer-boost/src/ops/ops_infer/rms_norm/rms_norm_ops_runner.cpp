/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rms_norm_ops_runner.h"
#include "atb/utils/log.h"

namespace atb {
static const uint64_t IN_TENSOR_COUNT_TWO = 2;
static const uint64_t IN_TENSOR_COUNT_THREE = 3;
static const uint64_t IN_TENSOR_COUNT_FOUR = 4;
static const uint64_t IN_TENSOR_COUNT_FIVE = 5;
static const uint64_t IN_TENSOR_COUNT_SIX = 6;
static const uint64_t OUT_TENSOR_COUNT_TWO = 2;
static const uint64_t OUT_TENSOR_COUNT_THREE = 3;
static const uint32_t GEMMA_MODE = 1;

void RmsNormOpsRunner::SetRmsNormParam(const infer::RmsNormParam &inferParam, AsdOps::OpParam::Norm &asdopsParam) const
{
    asdopsParam.inGamma = true;
    asdopsParam.epsilon = inferParam.normParam.epsilon;
    asdopsParam.precisionMode = inferParam.normParam.precisionMode;
    if (inferParam.normParam.modelType == infer::RmsNormParam::GEMMA_MODEL) {
        asdopsParam.gemmaMode = GEMMA_MODE;
    }
}

void RmsNormOpsRunner::SetRmsNormQuantParam(const infer::RmsNormParam &inferParam,
                                            AsdOps::OpParam::Norm &asdopsParam) const
{
    asdopsParam.inGamma = true;
    asdopsParam.inBeta = true;
    asdopsParam.epsilon = inferParam.normParam.epsilon;
    asdopsParam.isDynamicQuant = inferParam.normParam.dynamicQuantType != infer::DYNAMIC_QUANT_UNDEFINED;
    asdopsParam.isSymmetric = inferParam.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_SYMMETRIC;
}

void RmsNormOpsRunner::SetPreRmsNormQuantParam(const infer::RmsNormParam &inferParam,
                                               AsdOps::OpParam::Norm &asdopsParam) const
{
    asdopsParam.inGamma = true;
    asdopsParam.inRes = true;
    asdopsParam.inNormBias = true;
    asdopsParam.outRes = true;
    asdopsParam.epsilon = inferParam.preNormParam.epsilon;
}

void RmsNormOpsRunner::SetPreRmsNormParam(const infer::RmsNormParam &inferParam,
                                          AsdOps::OpParam::Norm &asdopsParam) const
{
    asdopsParam.inGamma = true;
    asdopsParam.inBeta = true;
    asdopsParam.inRes = true;
    asdopsParam.outRes = true;
    asdopsParam.epsilon = inferParam.preNormParam.epsilon;
}

void RmsNormOpsRunner::SetPostRmsNormQuantParam(const infer::RmsNormParam &inferParam,
                                                AsdOps::OpParam::Norm &asdopsParam) const
{
    asdopsParam.inGamma = true;
    asdopsParam.inRes = true;
    asdopsParam.outRes = true;
    asdopsParam.epsilon = inferParam.postNormParam.epsilon;
}

void RmsNormOpsRunner::SetPostRmsNormParam(const infer::RmsNormParam &inferParam,
                                           AsdOps::OpParam::Norm &asdopsParam) const
{
    asdopsParam.inGamma = true;
    asdopsParam.inBeta = true;
    asdopsParam.inRes = true;
    asdopsParam.epsilon = inferParam.postNormParam.epsilon;
}

void RmsNormOpsRunner::BuildRmsNormGraph(const AsdOps::OpParam::Norm &rmsNormParam)
{
    // 2 in -> 1 out, RmsNorm, -1
    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT_TWO);
    size_t inId = 0;
    Mki::Tensor &inputXTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &gammaTensor = kernelGraph_.inTensors.at(inId++);

    kernelGraph_.outTensors.resize(1);
    Mki::Tensor &resultTensor = kernelGraph_.outTensors.at(0);

    kernelGraph_.nodes.resize(1);
    auto &rmsNormNode = kernelGraph_.nodes.at(0);
    rmsNormNode.opDesc = {0, "NormOperation", rmsNormParam};
    rmsNormNode.inTensors = {&inputXTensor, &gammaTensor};
    rmsNormNode.outTensors = {&resultTensor};
}

void RmsNormOpsRunner::BuildRmsNormForwardGraph(const AsdOps::OpParam::Norm &rmsNormParam)
{
    // 2 in -> 2 out, RmsNormForward
    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT_TWO);
    size_t inId = 0;
    Mki::Tensor &inputXTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &gammaTensor = kernelGraph_.inTensors.at(inId++);

    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT_TWO);
    size_t outId = 0;
    Mki::Tensor &resultTensor = kernelGraph_.outTensors.at(outId++);
    Mki::Tensor &rstd = kernelGraph_.outTensors.at(outId++);

    kernelGraph_.nodes.resize(1);
    auto &rmsNormNode = kernelGraph_.nodes.at(0);
    rmsNormNode.opDesc = {0, "NormOperation", rmsNormParam};
    rmsNormNode.inTensors = {&inputXTensor, &gammaTensor};
    rmsNormNode.outTensors = {&resultTensor, &rstd};
}

void RmsNormOpsRunner::BuildRmsNormQuantGraph(const AsdOps::OpParam::Norm &rmsNormParam)
{
    // 3 in -> 1 out, RmsNormQuant, dim -1
    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT_FIVE);
    size_t inId = 0;
    Mki::Tensor &inputXTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &gammaTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &betaTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &scaleTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &offsetTensor = kernelGraph_.inTensors.at(inId++);

    kernelGraph_.outTensors.resize(1);
    size_t outId = 0;
    Mki::Tensor &resultTensor = kernelGraph_.outTensors.at(outId++);

    kernelGraph_.nodes.resize(1);
    auto &rmsNormNode = kernelGraph_.nodes.at(0);
    rmsNormNode.opDesc = {0, "NormOperation", rmsNormParam};
    rmsNormNode.outTensors = {&resultTensor};
    rmsNormNode.inTensors = {&inputXTensor, &gammaTensor, &betaTensor, &scaleTensor, &offsetTensor};
}

void RmsNormOpsRunner::BuildRmsNormDynamicQuantGraph(const AsdOps::OpParam::Norm &rmsNormParam)
{
    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT_THREE);
    size_t inTensorId = 0;
    Mki::Tensor &xTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &gammaTensor = kernelGraph_.inTensors.at(inTensorId++);
    Mki::Tensor &betaTensor = kernelGraph_.inTensors.at(inTensorId++);

    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT_THREE);
    size_t outTensorId = 0;
    Mki::Tensor &yTensor = kernelGraph_.outTensors.at(outTensorId++);
    Mki::Tensor &scaleTensor = kernelGraph_.outTensors.at(outTensorId++);
    Mki::Tensor &offsetTensor = param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_SYMMETRIC ?
                                    nullTensor_ :
                                    kernelGraph_.outTensors.at(outTensorId++);

    kernelGraph_.nodes.resize(1);
    auto &rmsNormNode = kernelGraph_.nodes.at(0);
    rmsNormNode.opDesc = {0, "NormOperation", rmsNormParam};
    rmsNormNode.inTensors = {&xTensor, &gammaTensor, &betaTensor};
    rmsNormNode.outTensors = {&yTensor, &scaleTensor, &offsetTensor};
}

void RmsNormOpsRunner::BuildPreRmsNormQuantGraph(const AsdOps::OpParam::Norm &rmsNormParam)
{
    // 6 in -> 2 out, preRmsNormQuant, dim -1
    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT_SIX);
    size_t inId = 0;
    Mki::Tensor &inputXTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &residualAddIn = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &gammaTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &betaTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &scaleTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &offsetTensor = kernelGraph_.inTensors.at(inId++);

    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT_TWO);
    size_t outId = 0;
    Mki::Tensor &resQuant = kernelGraph_.outTensors.at(outId++);
    Mki::Tensor &resultTensor = kernelGraph_.outTensors.at(outId++);

    kernelGraph_.nodes.resize(1);
    auto &rmsNormNode = kernelGraph_.nodes.at(0);
    rmsNormNode.opDesc = {0, "NormOperation", rmsNormParam};
    rmsNormNode.inTensors = {&inputXTensor, &residualAddIn, &gammaTensor, &betaTensor, &scaleTensor, &offsetTensor};
    // 两个outTensor顺序与算子库适配
    rmsNormNode.outTensors = {&resQuant, &resultTensor};
}

void RmsNormOpsRunner::BuildPreRmsNormGraph(const AsdOps::OpParam::Norm &rmsNormParam)
{
    // 4 in -> 2 out, preRmsNormQuant, dim -1
    uint64_t inTensorNum = param_.preNormParam.hasBias ? IN_TENSOR_COUNT_FOUR : IN_TENSOR_COUNT_THREE;
    kernelGraph_.inTensors.resize(inTensorNum);
    size_t inId = 0;
    Mki::Tensor &inputXTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &biasTensor = param_.preNormParam.hasBias ? kernelGraph_.inTensors.at(inId++) : nullTensor_;
    Mki::Tensor &resInTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &gammaTensor = kernelGraph_.inTensors.at(inId++);

    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT_TWO);
    size_t outId = 0;
    Mki::Tensor &yTensor = kernelGraph_.outTensors.at(outId++);
    Mki::Tensor &resOut = kernelGraph_.outTensors.at(outId++);

    kernelGraph_.nodes.resize(1);
    auto &rmsNormNode = kernelGraph_.nodes.at(0);
    rmsNormNode.opDesc = {0, "NormOperation", rmsNormParam};
    rmsNormNode.inTensors = {&inputXTensor, &biasTensor, &resInTensor, &gammaTensor};
    rmsNormNode.outTensors = {&yTensor, &resOut};
}

void RmsNormOpsRunner::BuildPostRmsNormQuantGraph(const AsdOps::OpParam::Norm &rmsNormParam)
{
    // 5 in -> 2 out
    kernelGraph_.inTensors.resize(IN_TENSOR_COUNT_FIVE);
    size_t inId = 0;
    Mki::Tensor &inputXTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &residualAddIn = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &gammaTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &scaleTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &offsetTensor = kernelGraph_.inTensors.at(inId++);

    kernelGraph_.outTensors.resize(OUT_TENSOR_COUNT_TWO);
    size_t outId = 0;
    Mki::Tensor &resQuant = kernelGraph_.outTensors.at(outId++);
    Mki::Tensor &resultTensor = kernelGraph_.outTensors.at(outId++);

    kernelGraph_.nodes.resize(1);
    auto &rmsNormNode = kernelGraph_.nodes.at(0);
    rmsNormNode.opDesc = {0, "NormOperation", rmsNormParam};
    rmsNormNode.inTensors = {&inputXTensor, &residualAddIn, &gammaTensor, &scaleTensor, &offsetTensor};
    rmsNormNode.outTensors = {&resQuant, &resultTensor};
}

void RmsNormOpsRunner::BuildPostRmsNormGraph(const AsdOps::OpParam::Norm &rmsNormParam)
{
    uint64_t inTensorNum = param_.postNormParam.hasBias ? IN_TENSOR_COUNT_FOUR : IN_TENSOR_COUNT_THREE;
    kernelGraph_.inTensors.resize(inTensorNum);
    size_t inId = 0;
    Mki::Tensor &inputXTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &biasTensor = param_.postNormParam.hasBias ? kernelGraph_.inTensors.at(inId++) : nullTensor_;
    Mki::Tensor &resInTensor = kernelGraph_.inTensors.at(inId++);
    Mki::Tensor &gammaTensor = kernelGraph_.inTensors.at(inId++);

    kernelGraph_.outTensors.resize(1);
    size_t outId = 0;
    Mki::Tensor &outYTensor = kernelGraph_.outTensors.at(outId++);

    kernelGraph_.nodes.resize(1);
    auto &rmsNormNode = kernelGraph_.nodes.at(0);
    rmsNormNode.opDesc = {0, "NormOperation", rmsNormParam};
    rmsNormNode.inTensors = {&inputXTensor, &biasTensor, &resInTensor, &gammaTensor};
    rmsNormNode.outTensors = {&outYTensor};
}

RmsNormOpsRunner::RmsNormOpsRunner(const infer::RmsNormParam &param)
    : OpsRunner("RmsNormOpsRunner", RUNNER_TYPE_RMS_NORM), param_(param)
{
    AsdOps::OpParam::Norm rmsNormParam = {AsdOps::OpParam::Norm::RMS_NORM};
    if (param_.layerType == infer::RmsNormParam::RMS_NORM_NORM) {
        if (param_.normParam.quantType == infer::QUANT_UNQUANT) {
            rmsNormParam = {param_.normParam.rstd ? AsdOps::OpParam::Norm::RMS_NORM_FORWARD :
                                                    AsdOps::OpParam::Norm::RMS_NORM};
            SetRmsNormParam(param_, rmsNormParam);
            param_.normParam.rstd ? BuildRmsNormForwardGraph(rmsNormParam) : BuildRmsNormGraph(rmsNormParam);
        } else {
            SetRmsNormQuantParam(param_, rmsNormParam);
            if (param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_UNDEFINED) {
                BuildRmsNormQuantGraph(rmsNormParam);
            } else {
                BuildRmsNormDynamicQuantGraph(rmsNormParam);
            }
        }
    } else if (param_.layerType == infer::RmsNormParam::RMS_NORM_PRENORM) {
        if (param_.preNormParam.quantType == infer::QUANT_UNQUANT) {
            SetPreRmsNormParam(param_, rmsNormParam);
            BuildPreRmsNormGraph(rmsNormParam);
        } else {
            SetPreRmsNormQuantParam(param_, rmsNormParam);
            BuildPreRmsNormQuantGraph(rmsNormParam);
        }
    } else if (param_.layerType == infer::RmsNormParam::RMS_NORM_POSTNORM) {
        if (param_.postNormParam.quantType == infer::QUANT_UNQUANT) {
            SetPostRmsNormParam(param_, rmsNormParam);
            BuildPostRmsNormGraph(rmsNormParam);
        } else {
            SetPostRmsNormQuantParam(param_, rmsNormParam);
            BuildPostRmsNormQuantGraph(rmsNormParam);
        }
    } else {
        ATB_LOG(ERROR) << "RmsNormType Undefined";
    }
    ATB_LOG(INFO) << GetName() << " NormOperation opDesc normParam beginNormAxis:" << rmsNormParam.beginNormAxis
                  << ", beginParamsAxis:" << rmsNormParam.beginParamsAxis << ", epsilon:" << rmsNormParam.epsilon
                  << ", zoomScaleValue:" << rmsNormParam.zoomScaleValue << ", inGamma:" << rmsNormParam.inGamma
                  << ", inBeta:" << rmsNormParam.inBeta << ", inRes:" << rmsNormParam.inRes
                  << ", inNormBias:" << rmsNormParam.inNormBias << ", outMean:" << rmsNormParam.outMean
                  << ", outVarience:" << rmsNormParam.outVarience << ", outResQuant:" << rmsNormParam.outResQuant
                  << ", outRes:" << rmsNormParam.outRes;
}

RmsNormOpsRunner::~RmsNormOpsRunner() {}
} // namespace atb