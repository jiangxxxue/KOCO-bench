/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TORCH_ATB_ENGER_GRAPH_OPERATION_BUILDER_H
#define TORCH_ATB_ENGER_GRAPH_OPERATION_BUILDER_H
#include "atb/atb_infer.h"
#include "graph_node.h"
#include "operation_wrapper.h"
 
namespace TorchAtb {
using ReshapeHandler = std::function<std::vector<int64_t>(const std::vector<int64_t> &oldShape)>;
 
class GraphBuilder {
public:
    explicit GraphBuilder(const std::string &graphName);
    std::string AddInput(const std::string &name);
    GraphNode &AddNode(const std::vector<std::string> &inputs, atb::Operation *operation);
    template <typename OpParam>
    GraphNode &AddNodeByParamType(const std::vector<std::string> &inputs, const OpParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::LinearParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::LayerNormParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::ElewiseParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::SoftmaxParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::SelfAttentionParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::RopeParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::SplitParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::GatherParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::ActivationParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::RmsNormParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::AllGatherParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::AsStridedParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::CumsumParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::DynamicNTKParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::MultinomialParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::ConcatParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::SliceParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::TransposeParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::GatingParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::ReshapeAndCacheParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::FillParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::RazorFusionAttentionParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::AllReduceParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::BroadcastParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::ReduceScatterParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::ReduceScatterVParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::LinearParallelParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::LinearSparseParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::RelayAttentionParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::TopkToppSamplingParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, const atb::infer::AllToAllParam &param);
    GraphNode &AddNode(const std::vector<std::string> &inputs, OperationWrapper &opWrapper);
    GraphBuilder &Reshape(const std::string &srcTensorName, const ReshapeHandler &reshapeHandler,
                                const std::string &reshapedTensorName);
    void MarkOutput(const std::string &outTensor);
    void SetExecuteStreams(const std::vector<std::uintptr_t> &executeStreams);
    OperationWrapper Build();

private:
    atb::GraphParam graphParam_;
    std::vector<GraphNode> graphNodes_;
    uint32_t internalTensorNum_ = 0;
    uint32_t GetTensorId(const std::string &tensorName);
    void ExecuteStreamsAssign();
    std::set<uint32_t> streamIds_;
    std::vector<aclrtStream> executeStreams_;
    std::map<std::string, uint32_t> inTensorIds_;
    std::map<std::string, uint32_t> outTensorIds_;
    std::map<std::string, uint32_t> internalTensorIds_;
    std::map<std::string, std::pair<std::string, atb::ReshapeFunc>> reshapedTensorIds_;
};

} // namespace TorchAtb
#endif // TORCH_ATB_ENGER_GRAPH_OPERATION_BUILDER_H