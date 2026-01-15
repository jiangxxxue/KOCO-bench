/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "layer_ops/post_process/post_process_opration.h"
#include <atb/atb_infer.h>
#include "atb/utils/log.h"
#include "nlohmann/json.hpp"

namespace atb_speed {

const size_t sampleInTensorNum = 3;
const size_t sampleOutTensorNum = 2;
const size_t sampleOutputDimNum = 2;

enum PostProcessTensorId : int {
    IN_SCORES = 0,
    IN_TEMPERATURE,
    IN_TOPP,
    OUT_INDICES,
    OUT_SCORES,
    INTERMEDIATE_SCORES_AFTER_SOFTMAX,
    INTERMEDIATE_SCORES_AFTER_TEMPERATURE,
};

void SetSampleOpGraph(atb::GraphParam &opGraph, const PostProcessParam &param)
{
    size_t sampleInterTensorNum = 1;
    size_t nodeCount = 2;
    if (param.temperature != 1.0) {
        sampleInterTensorNum += 1;
        nodeCount += 1;
    }
    if (param.topK <= 0) {
        ATB_LOG(ERROR) << "topK must be greater than zero, "
                        << "if you want a full vocabulary list as input, set topK = (vocabulary list's size)";
    }
    opGraph.inTensorNum = sampleInTensorNum;
    opGraph.outTensorNum = sampleOutTensorNum;
    opGraph.internalTensorNum = sampleInterTensorNum;
    opGraph.nodes.resize(nodeCount);
    opGraph.name = "SampleOpGraph";
}

atb::Status CreatePostProcessOperation(const PostProcessParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    SetSampleOpGraph(opGraph, param);

    size_t nodeId = 0;
    uint32_t interId = IN_SCORES;
    if (param.temperature != 1.0) {
        atb::Node &divNode = opGraph.nodes.at(nodeId++);
        atb::infer::ElewiseParam divParam;
        divParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_REALDIV;
        CreateOperation(divParam, &divNode.operation);
        divNode.inTensorIds = {interId, IN_TEMPERATURE};
        divNode.outTensorIds = {INTERMEDIATE_SCORES_AFTER_TEMPERATURE};
        interId = INTERMEDIATE_SCORES_AFTER_TEMPERATURE;
    }
    atb::Node &softmaxNode = opGraph.nodes.at(nodeId++);
    atb::infer::SoftmaxParam softmaxParam;
    softmaxParam.axes = {-1};
    CreateOperation(softmaxParam, &softmaxNode.operation);
    softmaxNode.inTensorIds = {interId};
    softmaxNode.outTensorIds = {INTERMEDIATE_SCORES_AFTER_SOFTMAX};

    atb::Node &topkToppSamplingNode = opGraph.nodes.at(nodeId++);
    atb::infer::TopkToppSamplingParam topkToppSamplingParam;
    topkToppSamplingParam.topk = param.topK;
    topkToppSamplingParam.randSeed = param.randSeed;
    CreateOperation(topkToppSamplingParam, &topkToppSamplingNode.operation);
    topkToppSamplingNode.inTensorIds = {INTERMEDIATE_SCORES_AFTER_SOFTMAX, IN_TOPP};
    topkToppSamplingNode.outTensorIds = {OUT_INDICES, OUT_SCORES};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                    atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0).dtype = ACL_INT32;
        outTensorDescs.at(0).format = inTensorDescs.at(0).format;
        outTensorDescs.at(0).shape.dimNum = sampleOutputDimNum;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = 1;

        outTensorDescs.at(1).dtype = inTensorDescs.at(0).dtype;
        outTensorDescs.at(1).format = inTensorDescs.at(0).format;
        outTensorDescs.at(1).shape.dimNum = sampleOutputDimNum;
        outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(1).shape.dims[1] = 1;

        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}

} // namespace atb_speed