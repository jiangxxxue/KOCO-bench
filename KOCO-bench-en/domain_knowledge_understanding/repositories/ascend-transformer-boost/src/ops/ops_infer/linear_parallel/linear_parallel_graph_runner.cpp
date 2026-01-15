/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "linear_parallel_graph_runner.h"
#include "atb/utils/log.h"
#include "atb/utils/runner_util.h"

namespace atb {
static constexpr size_t DIM_2 = 2;
static constexpr size_t DIM_3 = 3;
LinearParallelGraphRunner::LinearParallelGraphRunner(const infer::LinearParallelParam &param, ContextBase &context)
    : GraphRunner("LinearParallelGraphRunner"), param_(param)
{
    ATB_LOG(INFO) << "LinearParallelGraphRunner::LinearParallelGraphRunner called";
    InitGraph(context);
}

void LinearParallelGraphRunner::InitLinearNodeInTensorReshapeFuncs(GraphRunner::Node &linearNode) const
{
    linearNode.inTensorReshapeFuncs.resize(1);
    linearNode.inTensorReshapeFuncs.at(0) = [&](const Dims &oldShape, Dims &newShape) {
        if (oldShape.dimNum == DIM_3) {
            newShape.dimNum = DIM_2;
            newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
            newShape.dims[1] = oldShape.dims[DIM_2];
        } else {
            newShape = oldShape;
        }
    };
}

bool LinearParallelGraphRunner::InitLinearNode(infer::LinearParam &linearParam, GraphRunner::Node &linearNode,
                                               std::vector<int64_t> &nodeOperationIds, ContextBase &context) const
{
    Operation *linearOperation = nullptr;
    if (CreateOperation(linearParam, &linearOperation) != NO_ERROR) {
        ATB_LOG(ERROR) << "Create Linear Operation failed!";
        return false;
    }
    if (!RunnerUtil::InitGraphRunnerNode(linearNode, linearOperation, nodeOperationIds, context)) {
        return false;
    }
    nodeOperationIds.back()++;
    return true;
}

bool LinearParallelGraphRunner::InitAllReduceNode(infer::AllReduceParam &allReduceParam,
                                                  GraphRunner::Node &allReduceNode,
                                                  std::vector<int64_t> &nodeOperationIds, ContextBase &context) const
{
    Operation *allReduceOperation = nullptr;
    if (CreateOperation(allReduceParam, &allReduceOperation) != NO_ERROR) {
        ATB_LOG(ERROR) << "Create AllReduce Operation failed!";
        return false;
    }
    if (!RunnerUtil::InitGraphRunnerNode(allReduceNode, allReduceOperation, nodeOperationIds, context)) {
        return false;
    }
    nodeOperationIds.back()++;
    return true;
}

bool LinearParallelGraphRunner::InitAddNode(GraphRunner::Node &addNode, std::vector<int64_t> &nodeOperationIds,
                                            ContextBase &context) const
{
    infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    Operation *addOperation = nullptr;
    if (CreateOperation(addParam, &addOperation) != NO_ERROR) {
        ATB_LOG(ERROR) << "Create Elewise-add Operation failed!";
        return false;
    }
    if (!RunnerUtil::InitGraphRunnerNode(addNode, addOperation, nodeOperationIds, context)) {
        return false;
    }
    nodeOperationIds.back()++;
    return true;
}

void LinearParallelGraphRunner::InitGraph(ContextBase &context)
{
    GraphRunner::Graph &runnerGraph = GetGraph();
    size_t inTensorSize = param_.hasResidual ? 3 : 2;
    runnerGraph.inTensors.resize(inTensorSize);
    uint32_t inTensorId = 0;
    Tensor &inputTensor = runnerGraph.inTensors.at(inTensorId++);
    Tensor &weightTensor = runnerGraph.inTensors.at(inTensorId++);
    size_t outTensorSize = 1;
    runnerGraph.outTensors.resize(outTensorSize);
    Tensor &outTensor = runnerGraph.outTensors.at(0);
    std::vector<int64_t> nodeOperationIds = {0, 0};
    size_t nodeSize = param_.hasResidual ? 3 : 2;
    runnerGraph.nodes.resize(nodeSize);
    size_t nodeId = 0;
    GraphRunner::Node &linearNode = runnerGraph.nodes.at(nodeId++);
    infer::LinearParam linearParam;
    linearParam.transposeA = false;
    linearParam.transposeB = param_.transWeight;
    linearParam.hasBias = false;
    if (!InitLinearNode(linearParam, linearNode, nodeOperationIds, context)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "InitLinearNode failed.";
        return;
    }
    linearNode.inTensors = {&inputTensor, &weightTensor};
    linearNode.outTensors = {&outTensor};
    InitLinearNodeInTensorReshapeFuncs(linearNode);
    GraphRunner::Node &allReduceNode = runnerGraph.nodes.at(nodeId++);
    infer::AllReduceParam allReduceParam = {param_.rank,     param_.rankSize,      param_.rankRoot,
                                            "sum",           param_.backend,       param_.hcclComm,
                                            param_.commMode, param_.rankTableFile, param_.commDomain};
    if (!InitAllReduceNode(allReduceParam, allReduceNode, nodeOperationIds, context)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "InitAllReduceNode failed.";
        return;
    }
    allReduceNode.inTensors = {&outTensor};
    allReduceNode.outTensors = {&outTensor};
    if (param_.hasResidual) {
        GraphRunner::Node &addNode = runnerGraph.nodes.at(nodeId++);
        if (!InitAddNode(addNode, nodeOperationIds, context)) {
            ATB_LOG(ERROR) << GetLogPrefix() << "InitAddNode failed.";
            return;
        }
        Tensor &residualTensor = runnerGraph.inTensors.at(inTensorId++);
        addNode.inTensors = {&outTensor, &residualTensor};
        addNode.outTensors = {&outTensor};
    }
}

LinearParallelGraphRunner::~LinearParallelGraphRunner() {}
} // namespace atb