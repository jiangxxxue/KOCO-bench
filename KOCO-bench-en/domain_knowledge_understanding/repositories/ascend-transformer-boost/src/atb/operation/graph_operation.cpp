/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/operation/graph_operation.h"
#include "atb/types.h"
#include "atb/utils/log.h"
#include "atb/operation/plugin_operation.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/common_utils.h"

namespace atb {
const size_t MAX_NODE_NUM = 1024;
const size_t MAX_GRAPH_NAME_LEN = 128;

static bool CheckNodeTensorNum(const Node &node, size_t nodeId)
{
    if (!node.operation) {
        ATB_LOG(ERROR) << " node[" << nodeId << "].operation is null";
        return false;
    }
    if (node.operation->GetInputNum() != node.inTensorIds.size()) {
        ATB_LOG(ERROR) << " node[" << nodeId << "].inTensorIds.size: " << node.inTensorIds.size()
                       << " != operation.inputNum: " << node.operation->GetInputNum();
        return false;
    }
    if (node.operation->GetOutputNum() != node.outTensorIds.size()) {
        ATB_LOG(ERROR) << " node[" << nodeId << "].outTensorIds.size: " << node.outTensorIds.size()
                       << " != operation.outputNum: " << node.operation->GetOutputNum();
        return false;
    }
    return true;
}

static bool CheckNode(const Node &node, const size_t &nodeId, const uint64_t &totalTensorNum,
                      std::vector<bool> &tensorIsValued, std::vector<bool> &tensorIsUsed)
{
    if (!CheckNodeTensorNum(node, nodeId)) {
        return false;
    }

    for (size_t i = 0; i < node.inTensorIds.size(); ++i) {
        uint32_t tensorId = node.inTensorIds.at(i);
        if (tensorId >= totalTensorNum) {
            ATB_LOG(ERROR) << " node[" << nodeId << "].inTensorIds[" << i << "]: " << tensorId
                           << " is invalid, need less than:" << totalTensorNum;
            return false;
        }
        if (!tensorIsValued.at(tensorId)) { // tensor尚未被赋值
            ATB_LOG(ERROR) << " node[" << nodeId << "].inTensorIds[" << i << "]: " << tensorId
                           << " is not assigned value yet, please check your graph.";
            return false;
        }
        tensorIsUsed.at(tensorId) = true;
    }

    for (size_t i = 0; i < node.outTensorIds.size(); ++i) {
        uint32_t tensorId = node.outTensorIds.at(i);
        if (tensorId >= totalTensorNum) {
            ATB_LOG(ERROR) << " node[" << nodeId << "].outTensorIds[" << i << "]: " << tensorId
                           << " is invalid, need less than: " << totalTensorNum;
            return false;
        }
        if (tensorIsValued.at(tensorId)) { // tensor已被赋值
            bool writeInPlaceFlag = false;
            for (size_t j = 0; j < node.inTensorIds.size(); ++j) {
                if (node.inTensorIds.at(j) == tensorId) {
                    writeInPlaceFlag = true;
                    break;
                }
            }
            if (writeInPlaceFlag) {
                ATB_LOG(WARN) << " node[" << nodeId << "].outTensorIds[" << i << "]: " << tensorId
                              << " is write in place.";
                tensorIsUsed.at(tensorId) = false; // 原地写之后，tensor尚未被使用
            } else {
                ATB_LOG(WARN) << " node[" << nodeId << "].outTensorIds[" << i << "]: " << tensorId
                               << " has already been assigned value, please check your graph.";
            }
        } else {
            tensorIsValued.at(tensorId) = true;
        }
    }
    return true;
}

static bool IsValidGraphName(const std::string &name)
{
    if (name.length() > MAX_GRAPH_NAME_LEN) {
        return false;
    }
    for (char c : name) {
        if (!isalnum(c) && c != '_') {
            return false;
        }
    }
    return true;
}

static bool CheckAllTensor(const GraphParam &param, const std::vector<bool> &tensorIsValued,
                           const std::vector<bool> &tensorIsUsed)
{
    uint64_t totalTensorNum = param.inTensorNum + param.outTensorNum + param.internalTensorNum;
    for (size_t tensorId = param.inTensorNum; tensorId < totalTensorNum; tensorId++) {
        if (!tensorIsValued.at(tensorId)) {
            ATB_LOG(ERROR) << "graph tensorId: " << tensorId << " is not assigned value, please check your graph.";
            return false;
        }
    }

    for (size_t tensorId = 0; tensorId < param.inTensorNum; tensorId++) {
        if (!tensorIsUsed.at(tensorId)) {
            ATB_LOG(WARN) << "graph intensorId: " << tensorId << " is not used.";
        }
    }
    uint64_t internalTensorStartIdx = param.inTensorNum + param.outTensorNum;
    for (size_t tensorId = internalTensorStartIdx; tensorId < totalTensorNum; tensorId++) {
        if (!tensorIsUsed.at(tensorId)) {
            ATB_LOG(WARN) << "graph internal tensorId: " << tensorId << " is not used.";
        }
    }

    return true;
}

static bool CheckGraphParam(const GraphParam &param)
{
    ATB_LOG(INFO) << "start check " << param.name << " graph.";
    if (!IsValidGraphName(param.name)) {
        ATB_LOG(ERROR) << "GraphParam.name: " << param.name << " is invalid.";
        return false;
    }
    const uint64_t totalTensorNum = param.inTensorNum + param.outTensorNum + param.internalTensorNum;
    if (param.inTensorNum > MAX_SVECTOR_SIZE || param.outTensorNum > MAX_SVECTOR_SIZE ||
        param.internalTensorNum > MAX_SVECTOR_SIZE) {
        ATB_LOG(ERROR) << "graph intensor: " << param.inTensorNum << ", outtensor: " << param.outTensorNum
                       << ", internaltensor: " << param.internalTensorNum << " need less than " << MAX_SVECTOR_SIZE;
        return false;
    }
    if (param.nodes.size() > MAX_NODE_NUM) {
        ATB_LOG(ERROR) << "graph nodes num is too large, nodes num: " << param.nodes.size()
                       << " > max nodes num: " << MAX_NODE_NUM;
        return false;
    }
    std::vector<bool> tensorIsValued(totalTensorNum, false); // true: 已被赋值, false: 尚未赋值
    std::vector<bool> tensorIsUsed(totalTensorNum, false);   // true: 已被使用, false: 尚未使用
    for (size_t tensorId = 0; tensorId < param.inTensorNum; tensorId++) {
        tensorIsValued.at(tensorId) = true;
    }

    for (size_t nodeId = 0; nodeId < param.nodes.size(); ++nodeId) {
        const auto &node = param.nodes.at(nodeId);
        if (!CheckNode(node, nodeId, totalTensorNum, tensorIsValued, tensorIsUsed)) {
            ATB_LOG(ERROR) << "graph: " << param.name << " check failed.";
            return false;
        }
    }

    if (!CheckAllTensor(param, tensorIsValued, tensorIsUsed)) {
        return false;
    }

    ATB_LOG(INFO) << "check " << param.name << " graph success.";
    return true;
}

static std::string JoinInts(const SVector<uint32_t> &ids)
{
    std::string ret;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i == 0) {
            ret.append(std::to_string(ids.at(i)));
        } else {
            ret.append(", " + std::to_string(ids.at(i)));
        }
    }
    return ret;
}

static std::string GraphToString(const GraphParam &param)
{
    std::stringstream ss;
    ss << "inTensorNum:" << param.inTensorNum << ", outTensorNum:" << param.outTensorNum
       << ", internalTensorNum:" << param.internalTensorNum;
    for (size_t i = 0; i < param.nodes.size(); ++i) {
        ss << "\nnode[" << i << "]: operation:" << param.nodes.at(i).operation << ", inTensorIds:["
           << JoinInts(param.nodes.at(i).inTensorIds) << "], outTensorIds:[" << JoinInts(param.nodes.at(i).outTensorIds)
           << "]";
    }
    return ss.str();
}

template <> Status CreateOperation(const GraphParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        ATB_LOG(ERROR) << "invalid param, operation is null";
        return ERROR_INVALID_PARAM;
    }

    if (!CheckGraphParam(opParam)) {
        std::string graphString = GraphToString(opParam);
        ATB_LOG(ERROR) << "invalid param, graph param is invalid, graph:" << graphString;
        return ERROR_INVALID_GRAPH;
    }
    ATB_LOG(INFO) << GraphToString(opParam);

    std::string name = opParam.name.empty() ? "GraphOperation" : opParam.name;
    *operation = new (std::nothrow) GraphOperation(name, opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation:" << name;
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

GraphOperation::GraphOperation(const std::string &name) : OperationBase(name)
{
    UsePluginOperations();
    InitEmptyInTensorPerms();
    InitEmptyOutTensorPerms();
}

GraphOperation::GraphOperation(const std::string &name, const GraphParam &opGraph)
    : OperationBase(name), opGraph_(opGraph)
{
    UsePluginOperations();
    InitEmptyInTensorPerms();
    InitEmptyOutTensorPerms();
}

GraphOperation::~GraphOperation()
{
    for (size_t i = 0; i < opGraph_.nodes.size(); i++) {
        if (opGraph_.nodes.at(i).operation != nullptr) {
            DestroyOperation(opGraph_.nodes.at(i).operation);
            opGraph_.nodes.at(i).operation = nullptr;
        }
    }
}

uint32_t GraphOperation::GetInputNum() const
{
    return opGraph_.inTensorNum;
}

uint32_t GraphOperation::GetOutputNum() const
{
    return opGraph_.outTensorNum;
}

Status GraphOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                      SVector<TensorDesc> &outTensorDescs) const
{
    if (opGraph_.inferShapeFunc) {
        ATB_LOG(INFO) << GetLogPrefix() << "call user infer shape func";
        try {
            Status st = opGraph_.inferShapeFunc(inTensorDescs, outTensorDescs);
            if (st != NO_ERROR) {
                ATB_LOG(ERROR) << GetLogPrefix() << "user infer shape func fail, error: " << st;
                return st;
            }
        } catch (const std::exception &e) {
            ATB_LOG(ERROR) << GetLogPrefix() << "user infer shape func throw an exception:" << e.what();
            return ERROR_GRAPH_INFERSHAPE_FUNC_FAIL;
        }
        for (size_t i = 0; i < outTensorDescs.size(); i++) {
            TensorDesc desc = outTensorDescs.at(i);
            if (desc.shape.dimNum == 0 || desc.dtype == ACL_DT_UNDEFINED || desc.format == ACL_FORMAT_UNDEFINED) {
                ATB_LOG(ERROR) << GetLogPrefix() << "user infer shape result error, outTensorDesc[" << i
                               << "]: " << TensorUtil::TensorDescToString(desc);
                return ERROR_GRAPH_INFERSHAPE_FUNC_FAIL;
            }
        }
        return NO_ERROR;
    }

    return InferShapeImplDefault(inTensorDescs, outTensorDescs);
}

std::shared_ptr<Runner> GraphOperation::CreateRunner(Context &context) const
{
    std::shared_ptr<GraphRunner> runner = std::make_shared<GraphRunner>(GetName() + "Runner");
    if (!runner) {
        ATB_LOG(ERROR) << GetLogPrefix() << "make_shared GraphRunner fail";
        return std::shared_ptr<Runner>();
    }

    GraphRunner::Graph &runnerGraph = runner->GetGraph();
    runnerGraph.inTensors.reserve(opGraph_.inTensorNum);
    runnerGraph.outTensors.reserve(opGraph_.outTensorNum);
    runnerGraph.internalTensors.reserve(opGraph_.internalTensorNum);
    runnerGraph.inTensors.resize(opGraph_.inTensorNum);
    runnerGraph.outTensors.resize(opGraph_.outTensorNum);
    runnerGraph.internalTensors.resize(opGraph_.internalTensorNum);
    runnerGraph.nodes.resize(opGraph_.nodes.size());

    std::vector<Tensor *> fullTensorPtrs(opGraph_.inTensorNum + opGraph_.outTensorNum + opGraph_.internalTensorNum);
    BuildFullTensorPtrs(fullTensorPtrs, runnerGraph);

    size_t nodeIdIdx = operationBaseIds_.size();
    std::vector<int64_t> nodeOperationIds = operationBaseIds_;
    nodeOperationIds.push_back(0);

    for (size_t i = 0; i < runnerGraph.nodes.size(); ++i) {
        nodeOperationIds.at(nodeIdIdx) = static_cast<int64_t>(i);
        Status ret = CreateRunnerNode(i, runnerGraph, nodeOperationIds, fullTensorPtrs, context);
        if (ret != NO_ERROR) {
            return std::shared_ptr<Runner>();
        }
    }

    ATB_LOG(INFO) << GetLogPrefix() << "create runner success";
    return runner;
}

Status GraphOperation::CreateRunnerNode(const size_t nodeId, GraphRunner::Graph &runnerGraph,
                                        std::vector<int64_t> &nodeOperationIds,
                                        const std::vector<Tensor *> &fullTensorPtrs, Context &context) const
{
    auto &opNode = opGraph_.nodes.at(nodeId);
    GraphRunner::Node &runnerNode = runnerGraph.nodes.at(nodeId);
    if (!opNode.operation) {
        ATB_LOG(ERROR) << GetLogPrefix() << "node[" << nodeId << "] operation is invalid.";
        return ERROR_INVALID_PARAM;
    }

    runnerNode.op.reset(opNode.operation, [](Operation *operation) { (void)operation; });
    OperationBase *opBase = dynamic_cast<OperationBase *>(opNode.operation);
    if (!opBase) {
        ATB_LOG(ERROR) << GetLogPrefix() << "node[" << nodeId << "] operation is not inherit from OperationBase";
        return ERROR_INVALID_PARAM;
    }
    opBase->runner_ = opBase->CreateRunner(context);
    runnerNode.runner = opBase->runner_;
    if (!runnerNode.runner) {
        ATB_LOG(ERROR) << GetLogPrefix() << "node[" << nodeId << "] runner is null.";
        return ERROR_INVALID_PARAM;
    }
    runnerNode.runner->SetRunnerInfo(runnerNode.op->GetName(), nodeOperationIds);

    runnerNode.inTensorReshapeFuncs = opNode.inTensorReshapeFuncs;
    runnerNode.inTensors.reserve(opNode.inTensorIds.size());
    runnerNode.outTensors.reserve(opNode.outTensorIds.size());
    runnerNode.inTensorChunks = opNode.inTensorChunks;
    runnerNode.inTensors.resize(opNode.inTensorIds.size());
    runnerNode.outTensors.resize(opNode.outTensorIds.size());

    for (size_t j = 0; j < opNode.inTensorIds.size(); ++j) {
        runnerNode.inTensors.at(j) = fullTensorPtrs.at(opNode.inTensorIds.at(j));
    }
    for (size_t k = 0; k < opNode.outTensorIds.size(); ++k) {
        runnerNode.outTensors.at(k) = fullTensorPtrs.at(opNode.outTensorIds.at(k));
    }
    return NO_ERROR;
}

Status GraphOperation::SetNodeOperationIds()
{
    for (size_t i = 0; i < opGraph_.nodes.size(); i++) {
        auto &opNode = opGraph_.nodes.at(i);
        if (!opNode.operation) {
            ATB_LOG(ERROR) << GetLogPrefix() << "node[" << i << "] operation is invalid.";
            return ERROR_INVALID_PARAM;
        }
        OperationBase *opBase = dynamic_cast<OperationBase *>(opNode.operation);
        if (!opBase) {
            ATB_LOG(ERROR) << GetLogPrefix() << "node[" << i << "] operation is not inherit from OperationBase";
            return ERROR_INVALID_PARAM;
        }
        Status st = opBase->SetOperationBaseIds(operationBaseIds_, i);
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "set graphoperation node[" << i << "] operationBaseId fail";
            return st;
        }
    }
    return NO_ERROR;
}

void GraphOperation::UsePluginOperations()
{
    for (size_t i = 0; i < opGraph_.nodes.size(); ++i) {
        auto &opNode = opGraph_.nodes.at(i);
        OperationBase *opBase = dynamic_cast<OperationBase *>(opNode.operation);
        if (!opBase) {
            Operation *oldOperation = opNode.operation;
            PluginOperation *pluginOp = new PluginOperation(oldOperation);
            if (!pluginOp) {
                ATB_LOG(ERROR) << GetLogPrefix() << "new PluginOperation Failed!";
                return;
            }
            ATB_LOG(INFO) << GetLogPrefix() << "node[" << i << "] operation is plugin operation, new PluginOperation";
            opNode.operation = pluginOp;
        }
    }
}

Status GraphOperation::InferShapeImplDefault(const SVector<TensorDesc> &inTensorDescs,
                                             SVector<TensorDesc> &outTensorDescs) const
{
    ATB_LOG(INFO) << GetLogPrefix() << "InferShapeImplDefault start, inTensorDescsSize:" << inTensorDescs.size()
                  << ", outTensorDescs:" << outTensorDescs.size();
    std::vector<TensorDesc> totalTensorDescs(opGraph_.inTensorNum + opGraph_.outTensorNum + opGraph_.internalTensorNum);
    for (size_t i = 0; i < inTensorDescs.size(); ++i) {
        totalTensorDescs.at(i) = inTensorDescs.at(i);
    }

    for (size_t nodeId = 0; nodeId < opGraph_.nodes.size(); ++nodeId) {
        auto &opNode = opGraph_.nodes.at(nodeId);
        SVector<TensorDesc> opInTensorDescs;
        opInTensorDescs.reserve(opNode.operation->GetInputNum());
        opInTensorDescs.resize(opNode.operation->GetInputNum());
        for (size_t i = 0; i < opNode.inTensorIds.size(); ++i) {
            uint32_t tensorId = opNode.inTensorIds.at(i);
            opInTensorDescs.at(i) = totalTensorDescs.at(tensorId);
            if (i < opNode.inTensorReshapeFuncs.size() && opNode.inTensorReshapeFuncs.at(i)) {
                Dims newShape;
                opNode.inTensorReshapeFuncs.at(i)(opInTensorDescs.at(i).shape, newShape);
                opInTensorDescs.at(i).shape = newShape;
            }
        }

        SVector<TensorDesc> opOutTensorDescs;
        Status st = opNode.operation->InferShape(opInTensorDescs, opOutTensorDescs);
        if (st != 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << "node[" << nodeId << "] infer shape fail, error code: " << st;
            return st;
        }

        for (size_t i = 0; i < opNode.outTensorIds.size(); ++i) {
            uint32_t tensorId = opNode.outTensorIds.at(i);
            totalTensorDescs.at(tensorId) = opOutTensorDescs.at(i);
        }
    }
    outTensorDescs.reserve(opGraph_.outTensorNum);
    outTensorDescs.resize(opGraph_.outTensorNum);
    for (size_t i = 0; i < opGraph_.outTensorNum; ++i) {
        outTensorDescs.at(i) = totalTensorDescs.at(inTensorDescs.size() + i);
    }

    return NO_ERROR;
}

void GraphOperation::BuildFullTensorPtrs(std::vector<Tensor *> &fullTensorPtrs, GraphRunner::Graph &runnerGraph) const
{
    size_t offset = 0;
    for (size_t i = 0; i < runnerGraph.inTensors.size(); ++i) {
        fullTensorPtrs.at(offset++) = &runnerGraph.inTensors.at(i);
    }
    for (size_t i = 0; i < runnerGraph.outTensors.size(); ++i) {
        fullTensorPtrs.at(offset++) = &runnerGraph.outTensors.at(i);
    }
    for (size_t i = 0; i < runnerGraph.internalTensors.size(); ++i) {
        fullTensorPtrs.at(offset++) = &runnerGraph.internalTensors.at(i);
    }
}

SVector<bool> GraphOperation::GetEmptyInTensorPermissions() const
{
    return emptyInTensorPerms_;
}

void GraphOperation::InitEmptyInTensorPerms()
{
    emptyInTensorPerms_.reserve(opGraph_.inTensorNum);
    emptyInTensorPerms_.resize(opGraph_.inTensorNum);
    for (size_t i = 0; i < emptyInTensorPerms_.size(); ++i) {
        emptyInTensorPerms_.at(i) = false;
    }

    for (size_t nodeId = 0; nodeId < opGraph_.nodes.size(); ++nodeId) {
        auto &node = opGraph_.nodes.at(nodeId);
        if (!node.operation) {
            ATB_LOG(WARN) << GetLogPrefix() << "node[" << nodeId << "] operation is null";
            continue;
        }
        OperationBase *opBase = dynamic_cast<OperationBase *>(node.operation);
        if (!opBase) {
            ATB_LOG(INFO) << GetLogPrefix() << "node[" << nodeId << "] operation is not OperationBase";
            continue;
        }

        SVector<bool> childOpEmptyInTensorPerms = opBase->GetEmptyInTensorPermissions();
        if (childOpEmptyInTensorPerms.size() != node.inTensorIds.size()) {
            ATB_LOG(WARN) << GetLogPrefix() << "node[" << nodeId
                          << "] childOpEmptyInTensorPerms.size:" << childOpEmptyInTensorPerms.size()
                          << " != inTensorIds.size:" << node.inTensorIds.size();
            continue;
        }

        for (size_t i = 0; i < childOpEmptyInTensorPerms.size(); ++i) {
            uint32_t inTensorId = node.inTensorIds.at(i); // graph in tensor id in range [0, inTensorNum)
            if (childOpEmptyInTensorPerms.at(i) && inTensorId < emptyInTensorPerms_.size()) {
                emptyInTensorPerms_.at(inTensorId) = true;
                ATB_LOG(INFO) << GetLogPrefix() << "node[" << nodeId << "] " << node.operation->GetName()
                              << " inTensor[" << i << "] is allow empty";
            }
        }
    }
    ATB_LOG(INFO) << GetLogPrefix() << "emptyInTensorPerms:" << emptyInTensorPerms_;
}

SVector<bool> GraphOperation::GetEmptyOutTensorPermissions() const
{
    return emptyOutTensorPerms_;
}
 
void GraphOperation::InitEmptyOutTensorPerms()
{
    emptyOutTensorPerms_.reserve(opGraph_.outTensorNum);
    emptyOutTensorPerms_.resize(opGraph_.outTensorNum);
    for (size_t i = 0; i < emptyOutTensorPerms_.size(); ++i) {
        emptyOutTensorPerms_.at(i) = false;
    }
 
    for (size_t nodeId = 0; nodeId < opGraph_.nodes.size(); ++nodeId) {
        auto &node = opGraph_.nodes.at(nodeId);
        if (!node.operation) {
            ATB_LOG(WARN) << GetLogPrefix() << "node[" << nodeId << "] operation is null";
            continue;
        }
        OperationBase *opBase = dynamic_cast<OperationBase *>(node.operation);
        if (!opBase) {
            ATB_LOG(INFO) << GetLogPrefix() << "node[" << nodeId << "] operation is not OperationBase";
            continue;
        }
 
        SVector<bool> childOpEmptyOutTensorPerms = opBase->GetEmptyOutTensorPermissions();
        if (childOpEmptyOutTensorPerms.size() != node.outTensorIds.size()) {
            ATB_LOG(WARN) << GetLogPrefix() << "node[" << nodeId
                          << "] childOpEmptyOutTensorPerms.size:" << childOpEmptyOutTensorPerms.size()
                          << " != outTensorIds.size:" << node.outTensorIds.size();
            continue;
        }
 
        for (size_t i = 0; i < childOpEmptyOutTensorPerms.size(); ++i) {
            uint32_t outTensorId = node.outTensorIds.at(i);
            // graph out tensor id in range [inTensorNum, inTensorNum + outTensorNum)
            if (childOpEmptyOutTensorPerms.at(i) && outTensorId >= opGraph_.inTensorNum &&
                outTensorId < opGraph_.inTensorNum + opGraph_.outTensorNum) {
                emptyOutTensorPerms_.at(outTensorId - opGraph_.inTensorNum) = true;
                ATB_LOG(INFO) << GetLogPrefix() << "node[" << nodeId << "] " << node.operation->GetName()
                              << " outTensor[" << i << "] is allow empty";
            }
        }
    }
    ATB_LOG(INFO) << GetLogPrefix() << "emptyOutTensorPerms:" << emptyOutTensorPerms_;
}

void GraphOperation::GetGraphInfoImpl(nlohmann::json &graphJson) const
{
    graphJson["internalTensorNum"] = opGraph_.internalTensorNum;
    graphJson["param"] = "";
    for (size_t i = 0; i < opGraph_.nodes.size(); i++) {
        nlohmann::json graphNodeJson;
        auto &node = opGraph_.nodes.at(i);
        nlohmann::json tmpNodeJson = "";
        OperationBase *opBase = dynamic_cast<OperationBase *>(node.operation);

        if (opBase) {
            tmpNodeJson = opBase->GetGraphInfo();
        } else {
            nlohmann::json tmpJson;
            tmpJson["name"] = node.operation ? node.operation->GetName() : "Unknown Operation";
            tmpNodeJson = tmpJson;
        }

        graphNodeJson = tmpNodeJson;
        ATB_LOG(INFO) << GetLogPrefix() << "node[" << i << "] json string:" << graphNodeJson.dump();

        graphNodeJson["inTensorIds"] = SVectorToVector(node.inTensorIds);
        graphNodeJson["outTensorIds"] = SVectorToVector(node.outTensorIds);
        graphJson["nodes"].emplace_back(graphNodeJson);
    }
}

void GraphOperation::SetExecuteStreamId(uint32_t streamId)
{
    streamId_ = streamId;
    for (size_t i = 0; i < opGraph_.nodes.size(); i++) {
        OperationBase *opBase = dynamic_cast<OperationBase*>(opGraph_.nodes.at(i).operation);
        if (!opBase) {
            ATB_LOG(ERROR) << GetLogPrefix() << "Graph node internal error! set stream id ERROR!";
            return;
        }
        if (opBase->GetExecuteStreamId() == 0) {
            ATB_LOG(INFO) << GetLogPrefix() << "Change node[" << i <<"] stream id to " << streamId;
            opBase->SetExecuteStreamId(streamId);
        }
    }
}
} // namespace atb