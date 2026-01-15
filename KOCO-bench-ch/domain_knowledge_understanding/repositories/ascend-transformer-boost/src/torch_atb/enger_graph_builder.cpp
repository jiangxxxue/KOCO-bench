/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "enger_graph_builder.h"
#include <stdexcept>
#include "graph_node.h"
#include "operation_wrapper.h"
#include "atb/utils/log.h"
#include "atb/operation/operation_base.h"
#include "atb/context.h"
#include "resource/utils.h"

namespace TorchAtb {
using namespace atb;

GraphBuilder::GraphBuilder(const std::string &graphName)
{
    graphParam_.name = graphName;
}

std::string GraphBuilder::AddInput(const std::string &name)
{
    inTensorIds_[name] = graphParam_.inTensorNum++;
    return name;
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, atb::Operation *operation)
{
    if (operation == nullptr) {
        throw std::runtime_error("add operation is null.");
    }
    GraphNode node;
    node.SetOperation(operation);
    node.inTensorIds = inputs;
    int64_t outputNum = operation->GetOutputNum();
    node.outTensorIds.resize(outputNum);
    std::string name = operation->GetName();
    for (size_t i = 0; i < static_cast<size_t>(outputNum); ++i) {
        node.outTensorIds[i] =
            "node" + std::to_string(graphNodes_.size()) + "_outTensor" + std::to_string(i) + "_" + name;
        graphParam_.internalTensorNum++;
    }
    graphNodes_.push_back(node);
    return graphNodes_.back();
}

template <typename OpParam>
GraphNode &GraphBuilder::AddNodeByParamType(const std::vector<std::string> &inputs, const OpParam &param)
{
    atb::Operation *operation = nullptr;
    atb::Status st = CreateOperation(param, &operation);
    if (st != atb::NO_ERROR) {
        throw std::runtime_error("CreateOperation failed!");
    }
    return AddNode(inputs, operation);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::LinearParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::LayerNormParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::ElewiseParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::SoftmaxParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::SelfAttentionParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::RopeParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::SplitParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::GatherParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::ActivationParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::RmsNormParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::AllGatherParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::AsStridedParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::CumsumParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::DynamicNTKParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::MultinomialParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::ConcatParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::SliceParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::TransposeParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::GatingParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::ReshapeAndCacheParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::FillParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs,
                                 const atb::infer::RazorFusionAttentionParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::AllReduceParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::BroadcastParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::ReduceScatterParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::ReduceScatterVParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::LinearParallelParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::LinearSparseParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::RelayAttentionParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::TopkToppSamplingParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, const atb::infer::AllToAllParam &param)
{
    return AddNodeByParamType(inputs, param);
}

GraphNode &GraphBuilder::AddNode(const std::vector<std::string> &inputs, OperationWrapper &opWrapper)
{
    atb::Operation *operation = opWrapper.ReleaseOperation();
    return AddNode(inputs, operation);
}

GraphBuilder &GraphBuilder::Reshape(const std::string &srcTensorName, const ReshapeHandler &reshapeHandler,
                                    const std::string &reshapedTensorName)
{
    atb::ReshapeFunc reshapeFunc = [reshapeHandler](const Dims &oldShape, Dims &newShape) {
        std::vector<int64_t> oldShapeVec(oldShape.dimNum);
        for (size_t i = 0; i < oldShape.dimNum; ++i) {
            oldShapeVec[i] = oldShape.dims[i];
        }
        std::vector<int64_t> newShapeVec = reshapeHandler(oldShapeVec);
        ATB_LOG(INFO) << "oldShapeVec: " << oldShapeVec << ", newShapeVec: " << newShapeVec;
        newShape.dimNum = newShapeVec.size();
        for (uint64_t i = 0; i < newShape.dimNum; ++i) {
            newShape.dims[i] = newShapeVec[i];
        }
    };
    reshapedTensorIds_[reshapedTensorName] = {srcTensorName, reshapeFunc};
    return *this;
}

void GraphBuilder::MarkOutput(const std::string &outTensor)
{
    bool findOutput = false;
    for (auto &graphNode : graphNodes_) {
        if (graphNode.FindOutput(outTensor)) {
            findOutput = true;
            break;
        }
    }
    if (!findOutput) {
        throw std::runtime_error(
            "Invalid outTensor passed to mark_output. It must be a valid output tensor from add_node.");
    }
    outTensorIds_[outTensor] = graphParam_.inTensorNum + graphParam_.outTensorNum++;
    graphParam_.internalTensorNum--;
}

OperationWrapper GraphBuilder::Build()
{
    graphParam_.nodes.clear();

    uint32_t internalId = 0;
    for (auto &graphNode : graphNodes_) {
        Node node;
        node.operation = graphNode.operation;
        uint32_t streamId = graphNode.GetStreamId();
        ATB_LOG(INFO) << "Operation: " << node.operation << ", at streamId: " << streamId;
        streamIds_.insert(streamId);
        node.inTensorIds.resize(0);
        node.outTensorIds.resize(0);
        node.inTensorReshapeFuncs.resize(0);
        for (const std::string &inTensorName : graphNode.inTensorIds) {
            node.inTensorIds.push_back(GetTensorId(inTensorName));
            if (reshapedTensorIds_.find(inTensorName) != reshapedTensorIds_.end()) {
                reshapedTensorIds_[inTensorName].first = GetTensorId(inTensorName);
                node.inTensorReshapeFuncs.push_back(reshapedTensorIds_[inTensorName].second);
            } else {
                node.inTensorReshapeFuncs.push_back(nullptr);
            }
        }
        for (const std::string &outTensorName : graphNode.outTensorIds) {
            if (outTensorIds_.count(outTensorName)) {
                node.outTensorIds.push_back(GetTensorId(outTensorName));
            } else {
                uint32_t id = graphParam_.inTensorNum + graphParam_.outTensorNum + internalId++;
                internalTensorIds_[outTensorName] = id;
                node.outTensorIds.push_back(id);
            }
        }
        graphParam_.nodes.push_back(node);
    }
    ExecuteStreamsAssign();
    return OperationWrapper(graphParam_);
}

void GraphBuilder::SetExecuteStreams(const std::vector<std::uintptr_t> &executeStreams)
{
    for (auto &stream : executeStreams) {
        aclrtStream actualStream = nullptr;
        actualStream = reinterpret_cast<aclrtStream>(stream);
        if (!actualStream) {
            throw std::runtime_error("Cast int type aclrtStream to aclrtStream failed");
        }
        executeStreams_.push_back(actualStream);
    }
}

void GraphBuilder::ExecuteStreamsAssign()
{
    if (executeStreams_.empty()) {
        aclrtStream defaultStream = Utils::GetCurrentStream();
        executeStreams_.push_back(defaultStream);
        for (size_t i = 1; i < streamIds_.size(); ++i) {
            aclrtStream stream;
            aclError ret = aclrtCreateStream(&stream);
            if (ret != 0) {
                throw std::runtime_error("Create aclrtStream during set execute streams failed");
            }
            executeStreams_.push_back(stream);
        }
    }
    atb::Context *context = Utils::GetAtbContext();
    context->SetExecuteStreams(executeStreams_);
    ATB_LOG(INFO) << "context: " << context << ", streams: [" << executeStreams_ << "]";
}

uint32_t GraphBuilder::GetTensorId(const std::string &tensorName)
{
    if (inTensorIds_.find(tensorName) != inTensorIds_.end()) {
        return inTensorIds_[tensorName];
    } else if (outTensorIds_.find(tensorName) != outTensorIds_.end()) {
        return outTensorIds_[tensorName];
    } else if (reshapedTensorIds_.find(tensorName) != reshapedTensorIds_.end()) {
        return GetTensorId(reshapedTensorIds_[tensorName].first);
    } else if (internalTensorIds_.find(tensorName) != internalTensorIds_.end()) {
        return internalTensorIds_[tensorName];
    } else {
        uint32_t internalTensorId = inTensorIds_.size() + outTensorIds_.size() + internalTensorNum_++;
        internalTensorIds_[tensorName] = internalTensorId;
        return internalTensorId;
    }
}
} // namespace TorchAtb