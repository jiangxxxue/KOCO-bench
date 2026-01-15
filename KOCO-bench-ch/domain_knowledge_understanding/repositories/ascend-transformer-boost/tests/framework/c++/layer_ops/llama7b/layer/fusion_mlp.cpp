/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "fusion_mlp.h"

namespace atb_speed {
namespace llama_7b {
enum FusionMlpTensorId {
    IN_HIDDENSTATES_ID = 0,             // [batch, seqLen, hiddenSize], half
    IN_WEIGHT_GATE_UP_ID,                  // [11008*2, hiddenSize], half
    IN_WEIGHT_DOWN_ID,                  // [hiddenSize, 11008], half
    OUT_TRANSPOSED_RESULT_ID,           // [batch, seqLen, hiddenSize], half
    INTERMEDIATE_MATMUL_GATEUP_OUT_ID, // [batch, seqLen, 11008], half
    INTERMEDIATE_MATMUL_GATE_OUT_ID, // [batch, seqLen, 11008], half
    INTERMEDIATE_MATMUL_UP_OUT_ID, // [batch, seqLen, 11008], half
    INTERMEDIATE_SWISH_OUT_ID,          // [batch, seqLen, 11008], half
    INTERMEDIATE_MUL_OUT_ID,            // [batch, seqLen, 11008], half
};

static const uint64_t IN_TENSOR_COUNT = 3;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 5;
static const uint64_t NODE_COUNT = 5;

atb::Status FusionMlp(const FusionMlpParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = "fusion_mlp";

    size_t nodeId = 0;
    auto &matmulGateUpNode = opGraph.nodes.at(nodeId++);
    auto &splitNode = opGraph.nodes.at(nodeId++);
    auto &swishNode = opGraph.nodes.at(nodeId++);
    auto &mulNode = opGraph.nodes.at(nodeId++);
    auto &matmulDownNode = opGraph.nodes.at(nodeId++);

    atb::infer::LinearParam matmulGateParam;
    matmulGateParam.transposeB = param.transpose;
    matmulGateParam.hasBias = false;
    CreateOperation(matmulGateParam, &matmulGateUpNode.operation);
    matmulGateUpNode.inTensorIds = {IN_HIDDENSTATES_ID, IN_WEIGHT_GATE_UP_ID};
    matmulGateUpNode.outTensorIds = {INTERMEDIATE_MATMUL_GATEUP_OUT_ID};

    atb::infer::SplitParam splitParam = {2, 2};
    CreateOperation(splitParam, &splitNode.operation);
    splitNode.inTensorIds = {INTERMEDIATE_MATMUL_GATEUP_OUT_ID};
    splitNode.outTensorIds = {INTERMEDIATE_MATMUL_GATE_OUT_ID, INTERMEDIATE_MATMUL_UP_OUT_ID};

    atb::infer::ActivationParam swishParam;
    swishParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    CreateOperation(swishParam, &swishNode.operation);
    swishNode.inTensorIds = {INTERMEDIATE_MATMUL_GATE_OUT_ID};
    swishNode.outTensorIds = {INTERMEDIATE_SWISH_OUT_ID};

    atb::infer::ElewiseParam mulParam;
    mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CreateOperation(mulParam, &mulNode.operation);
    mulNode.inTensorIds = {INTERMEDIATE_SWISH_OUT_ID, INTERMEDIATE_MATMUL_UP_OUT_ID};
    mulNode.outTensorIds = {INTERMEDIATE_MUL_OUT_ID};

    atb::infer::LinearParam matmulDownParam;
    matmulDownParam.transposeB = param.transpose;
    matmulDownParam.hasBias = false;
    CreateOperation(matmulDownParam, &matmulDownNode.operation);
    matmulDownNode.inTensorIds = {INTERMEDIATE_MUL_OUT_ID, IN_WEIGHT_DOWN_ID};
    matmulDownNode.outTensorIds = {OUT_TRANSPOSED_RESULT_ID};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace llama_7b
} // namespace atb_speed