/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "chatglm6b_lmhead_operation.h"
#include <atb/atb_infer.h>
#include <atb/utils/log.h>

namespace atb_speed {

enum Chatglm6BLayerLmHeadTensorId { IN_HIDDENSTATES = 0, IN_LINEARWEIGHT, OUT_LMHEAD_OUT, INTERMIDATE_LINEAR_OUT };

static const uint64_t IN_TENSOR_COUNT = 2;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 1;
static const uint64_t NODE_COUNT = 2;
static uint64_t DIM3 = 3;

atb::Status CreateChatGlm6BLmHeadOperation(const ChatGlm6BLmHeadParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = "LmHeadOperation";
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    atb::Node &transposeNode = opGraph.nodes.at(nodeId++);

    atb::infer::LinearParam linearParam;
    linearParam.hasBias = false;
    CreateOperation(linearParam, &linearNode.operation);
    linearNode.inTensorIds = {IN_HIDDENSTATES, IN_LINEARWEIGHT};
    linearNode.outTensorIds = {INTERMIDATE_LINEAR_OUT};

    atb::infer::TransposeParam transposeParam;
    transposeParam.perm = {1, 0, 2};
    CreateOperation(transposeParam, &transposeNode.operation);
    transposeNode.inTensorIds = {INTERMIDATE_LINEAR_OUT};
    transposeNode.outTensorIds = {OUT_LMHEAD_OUT};

    opGraph.inferShapeFunc = [&](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
        outTensorDescs.at(0).format = inTensorDescs.at(0).format;
        outTensorDescs.at(0).shape.dimNum = DIM3;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(1).shape.dims[0]; // index: 2
        ATB_LOG(INFO) << "LmHead infershape success";
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace atb_speed