/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "layer_ops/view_graph/view_graph_operation.h"
#include <atb/atb_infer.h>
#include "atb/operation.h"
#include "atb/utils/log.h"
 
namespace atb_speed {
 
enum ViewGraphTensorId {
    IN_INPUTTENSOR = 0,
    IN_WEIGHTTENSOR,
    OUT_QTENSOR_ONE,
    INTERMIDATE_MATMUL_OUT,
    INTERMIDATE_ACTIVITION_ONE_OUT,
};
 
static const uint64_t IN_TENSOR_COUNT = 2;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 2;
static const uint64_t NODE_COUNT = 3;
 
atb::Status CreateViewGraphOperation(const ViewGraphParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
 
    size_t nodeId = 0;
    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    atb::Node &activationNodeOne = opGraph.nodes.at(nodeId++);
    atb::Node &activationOutNode = opGraph.nodes.at(nodeId++);
 
    atb::infer::LinearParam linearParam;
    linearParam.hasBias = false;
    CreateOperation(linearParam, &linearNode.operation);
    linearNode.inTensorIds = {IN_INPUTTENSOR, IN_WEIGHTTENSOR};
    linearNode.outTensorIds = {INTERMIDATE_MATMUL_OUT};
 
    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ACTIVATION_SWISH;
    CreateOperation(activationParam, &activationNodeOne.operation);
    activationNodeOne.inTensorIds = {INTERMIDATE_MATMUL_OUT};
    activationNodeOne.outTensorIds = {INTERMIDATE_ACTIVITION_ONE_OUT};
    activationNodeOne.inTensorChunks.resize(activationNodeOne.inTensorIds.size());
    activationNodeOne.inTensorChunks.at(0) = {2, 0};
 
    CreateOperation(activationParam, &activationOutNode.operation);
    activationOutNode.inTensorIds = {INTERMIDATE_ACTIVITION_ONE_OUT};
    activationOutNode.outTensorIds = {OUT_QTENSOR_ONE};
    activationOutNode.inTensorChunks.resize(activationOutNode.inTensorIds.size());
    activationOutNode.inTensorChunks.at(0) = {2, 1};
 
    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = 2;
        outTensorDescs.at(0).shape.dims[0] = 2;
        outTensorDescs.at(0).shape.dims[1] = 32;
        return atb::NO_ERROR;
    };
 
    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace atb_speed