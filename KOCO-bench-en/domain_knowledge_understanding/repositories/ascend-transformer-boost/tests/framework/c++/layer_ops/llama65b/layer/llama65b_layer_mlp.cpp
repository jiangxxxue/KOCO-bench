/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "llama65b_layer_mlp.h"
#include <atb/atb_infer.h>

namespace atb_speed {

enum LlamaMlpTensorId {
    IN_HIDDENSTATUS = 0,
    IN_WEIGHTTENSOR,
    OUT_MLPRESULTSTENSOR,
    INTERMIDATE_MATMUL_ALL_OUT,
    INTERMIDATE_MATMUL_GATE_OUT,
    INTERMIDATE_MATMUL_UP_OUT,
    INTERMIDATE_SWISH_OUT,
};

static const uint64_t IN_TENSOR_COUNT = 2;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 4;
static const uint64_t NODE_COUNT = 4;
static uint64_t DIM3 = 3;

atb::Status CreateLlamaMlpOperation(const LlamaMlpParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    atb::Node &splitNode = opGraph.nodes.at(nodeId++);
    atb::Node &swishNode = opGraph.nodes.at(nodeId++);
    atb::Node &mulNode = opGraph.nodes.at(nodeId++);

    atb::infer::LinearParam linearParam;
    linearParam.hasBias = false;
    linearParam.transposeB = param.transpose;
    CreateOperation(linearParam, &linearNode.operation);
    linearNode.inTensorIds = {IN_HIDDENSTATUS, IN_WEIGHTTENSOR};
    linearNode.outTensorIds = {INTERMIDATE_MATMUL_ALL_OUT};
    linearNode.inTensorReshapeFuncs.resize(linearNode.inTensorIds.size());
    linearNode.inTensorReshapeFuncs[0] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // dimNum: 2
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };

    atb::infer::SplitParam splitParam = {2, 2};
    CreateOperation(splitParam, &splitNode.operation);
    splitNode.inTensorIds = {INTERMIDATE_MATMUL_ALL_OUT};
    splitNode.outTensorIds = {INTERMIDATE_MATMUL_GATE_OUT, INTERMIDATE_MATMUL_UP_OUT};
    splitNode.inTensorReshapeFuncs.resize(splitNode.inTensorIds.size());
    splitNode.inTensorReshapeFuncs[0] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // dimNum: 3
        newShape.dims[0] = 1;
        newShape.dims[1] = oldShape.dims[0];
        newShape.dims[2] = oldShape.dims[1];
    };

    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    CreateOperation(activationParam, &swishNode.operation);
    swishNode.inTensorIds = {INTERMIDATE_MATMUL_GATE_OUT};
    swishNode.outTensorIds = {INTERMIDATE_SWISH_OUT};

    atb::infer::ElewiseParam elewiseParam;
    elewiseParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    CreateOperation(elewiseParam, &mulNode.operation);
    mulNode.inTensorIds = {INTERMIDATE_SWISH_OUT, INTERMIDATE_MATMUL_UP_OUT};
    mulNode.outTensorIds = {OUT_MLPRESULTSTENSOR};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        if (param.transpose == true) {
            outTensorDescs.at(0).shape.dimNum = DIM3;
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
            outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
            outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(1).shape.dims[0] / 2;
        } else {
            outTensorDescs.at(0).shape.dimNum = DIM3;
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
            outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
            outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(1).shape.dims[1] / 2;
        }
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace atb_speed