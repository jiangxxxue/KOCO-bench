/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/runner/graph_runner.h"
#include <sstream>
#include <string>
#include <acl/acl_rt.h>
#include "atb/utils/log.h"
#include "atb/utils/mem_allocation_solver/mem_allocation_solver_creator.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/config.h"
#include "atb/utils/statistic.h"
#include "atb/operation.h"
#include "atb/utils.h"
#include "atb/utils/singleton.h"
#include "atb/runner/ops_runner.h"

namespace atb {
const int ALIGN_INT = 512;
constexpr uint64_t FREE_TENSOR_KEY = 65536;
const int MAX_NODE_ID = 2048;

std::string GraphRunner::Graph::ToString() const
{
    std::stringstream ss;

    for (size_t i = 0; i < inTensors.size(); ++i) {
        ss << "inTensors[" << i << "]:" << TensorUtil::TensorToString(inTensors.at(i)) << std::endl;
    }
    for (size_t i = 0; i < outTensors.size(); ++i) {
        ss << "outTensors[" << i << "]:" << TensorUtil::TensorToString(outTensors.at(i)) << std::endl;
    }
    for (size_t i = 0; i < internalTensors.size(); ++i) {
        ss << "internalTensors[" << i << "]:" << TensorUtil::TensorToString(internalTensors.at(i)) << std::endl;
    }
    ss << "nodes:" << nodes.size() << std::endl;
    for (size_t i = 0; i < nodes.size(); ++i) {
        auto &node = nodes.at(i);
        ss << "node[" << i << "] opeation:" << node.op.get()
           << ", operationName:" << (node.op ? node.op->GetName() : "null") << ", runner:" << node.runner.get()
           << ", runnerName:" << (node.runner ? node.runner->GetName() : "null") << std::endl;
        for (auto tensorIt : node.inTensors) {
            ss << "node[" << i << "]:" << TensorUtil::TensorToString(*tensorIt) << std::endl;
        }
        for (auto tensorIt : node.outTensors) {
            ss << "node[" << i << "]:" << TensorUtil::TensorToString(*tensorIt) << std::endl;
        }
    }
    return ss.str();
}

void GraphRunner::Graph::Init()
{
    InitTensorMaxNodeMap();
    InitTensorType();
    // 临时方案，如果不是主流（streamId != 0）的node的inTensors的最大节点数全部设为2048不参与内存复用
    SetNonReuseTensors();
}

void GraphRunner::Graph::SetNonReuseTensors()
{
    for (size_t nodeId = 0; nodeId < nodes.size(); ++nodeId) {
        auto &node = nodes.at(nodeId);
        uint32_t streamId = GetExecuteStreamId(node.op.get());
        if (streamId == 0)
            continue;
        for (auto inTensorIt : node.inTensors) {
            auto it = isInTensorCanFree.find(inTensorIt);
            if (it != isInTensorCanFree.end()) {
                // 该inTensor是大图的inTensor，不参与内存分配
                continue; // 若intensor的isInTensorCanFree为false，不参与内存释放
            }
            maxNodeIdTensorMap[tensorMaxNodeIdMap[inTensorIt]].erase(inTensorIt);
            tensorMaxNodeIdMap[inTensorIt] = MAX_NODE_ID;
            maxNodeIdTensorMap[MAX_NODE_ID].insert(inTensorIt);
        }
    }
}

void GraphRunner::Graph::SearchTensorInNodeInTensor(const Tensor *tensor, uint64_t &maxNodeId,
                                                    uint64_t &dependNodeCount)
{
    for (size_t nodeId = 0; nodeId < nodes.size(); ++nodeId) {
        auto &node = nodes.at(nodeId);
        for (auto inTensorIt : node.inTensors) {
            if (tensor == inTensorIt) {
                maxNodeId = nodeId;
                dependNodeCount++;
            }
        }
    }
}

bool GraphRunner::Graph::SearchTensorInNodeOutTensor(const Tensor *tensor, uint64_t &maxNodeId)
{
    for (size_t nodeId = 0; nodeId < nodes.size(); ++nodeId) {
        auto &node = nodes.at(nodeId);
        for (auto outTensorIt : node.outTensors) {
            if (tensor == outTensorIt) {
                maxNodeId = nodeId;
                return true;
            }
        }
    }
    return false;
}

void GraphRunner::ReserveSvector(GraphRunner::Node &node)
{
    tilingBufferSizes_.reserve(runnerGraph_.nodes.size());
    intermediateBufferSizes_.reserve(runnerGraph_.nodes.size());
    node.runnerVariantPack.inTensors.reserve(node.inTensors.size());
    node.runnerVariantPack.outTensors.reserve(node.outTensors.size());
    node.runnerVariantPack.isInTensorCanFree.reserve(node.inTensors.size());
    node.runnerVariantPack.isOutTensorNeedMalloc.reserve(node.outTensors.size());
}

void GraphRunner::Graph::InitTensorMaxNodeMap()
{
    tensorMaxNodeIdMap.clear();
    maxNodeIdTensorMap.clear();

    if (atb::GetSingleton<Config>().IsworkspaceMemAllocGlobal()) {
        for (size_t i = 0; i < inTensors.size(); ++i) {
            Tensor *tensor = &inTensors.at(i);
            auto it = isInTensorCanFree.find(tensor);
            if (it == isInTensorCanFree.end() || !it->second) {
                continue; // 若intensor的isInTensorCanFree为false，不参与内存释放
            }
            uint64_t maxNodeId = 0;
            uint64_t dependNodeCount = 0;

            SearchTensorInNodeInTensor(tensor, maxNodeId, dependNodeCount);
            if (dependNodeCount == 0) {
                ATB_LOG(WARN) << "runner graph intensor[" << i << "] dependNodeCount is 0, graph wrong";
                tensorMaxNodeIdMap[tensor] = FREE_TENSOR_KEY; // 当intensor在graph内未被使用时，setup开始时立即释放
                maxNodeIdTensorMap[FREE_TENSOR_KEY].insert(tensor);
            } else {
                ATB_LOG(INFO) << "runner graph intensor[" << i << "] maxNodeId: " << maxNodeId
                              << ", dependNodeCount: " << dependNodeCount;
                tensorMaxNodeIdMap[tensor] = maxNodeId;
                maxNodeIdTensorMap[maxNodeId].insert(tensor);
            }
        }
    }

    for (size_t i = 0; i < internalTensors.size(); ++i) {
        Tensor *tensor = &internalTensors.at(i);
        uint64_t maxNodeId = 0;
        uint64_t dependNodeCount = 0;
        SearchTensorInNodeInTensor(tensor, maxNodeId, dependNodeCount);
        if (dependNodeCount == 0) {
            ATB_LOG(WARN) << "runner graph internal tensor[" << i << "] dependNodeCount is 0, graph wrong";

            if (!SearchTensorInNodeOutTensor(tensor, maxNodeId)) {
                ATB_LOG(WARN) << "runner graph internal tensor[" << i << "] is not valued, graph wrong";
                continue; // 当中间tensor未在graph内使用时，若有被赋值，则赋值结束立刻释放；若未被赋值，则无需释放
            }
        } else {
            ATB_LOG(INFO) << "runner graph internal tensor[" << i << "] maxNodeId: " << maxNodeId
                          << ", dependNodeCount: " << dependNodeCount;
        }
        tensorMaxNodeIdMap[tensor] = maxNodeId;
        maxNodeIdTensorMap[maxNodeId].insert(tensor);
    }
}

void GraphRunner::Graph::InitTensorType()
{
    for (auto &node : nodes) {
        node.inTensorTypes.reserve(node.inTensors.size());
        node.inTensorTypes.resize(node.inTensors.size());
        node.outTensorTypes.reserve(node.outTensors.size());
        node.outTensorTypes.resize(node.outTensors.size());
        for (size_t i = 0; i < node.inTensors.size(); ++i) {
            node.inTensorTypes.at(i) = IsInternalTensor(node.inTensors.at(i)) ? GraphRunner::INTERMEDIATE_TENSOR :
                                                                                GraphRunner::NOT_INTERMEDIATE_TENSOR;
        }
        for (size_t i = 0; i < node.outTensors.size(); ++i) {
            node.outTensorTypes.at(i) = IsInternalTensor(node.outTensors.at(i)) ? GraphRunner::INTERMEDIATE_TENSOR :
                                                                                  GraphRunner::NOT_INTERMEDIATE_TENSOR;
        }
    }
}

bool GraphRunner::Graph::IsInternalTensor(const Tensor *tensor)
{
    for (auto &internalTensor : internalTensors) {
        if (&internalTensor == tensor) {
            return true;
        }
    }
    return false;
}

GraphRunner::GraphRunner(const std::string &name) : Runner(name)
{
    if (GetSingleton<Config>().IsworkspaceMemAllocGlobal()) {
        memAllocationSolver_ = GetGlobalMemAllocationSolver();
    } else {
        memAllocationSolver_ = CreateMemAllocationSolver();
    }
}

GraphRunner::~GraphRunner() {}

GraphRunner::Graph &GraphRunner::GetGraph()
{
    return runnerGraph_;
}

Status GraphRunner::InitTensorFromRunnerVariantPack(RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "runnerVariantPack:" << runnerVariantPack.ToString();
    TensorUtil::FastCopyTensors(runnerVariantPack.inTensors, runnerGraph_.inTensors);
    TensorUtil::FastCopyTensors(runnerVariantPack.outTensors, runnerGraph_.outTensors);

    if (!runnerGraphInitFlag_) {
        runnerGraphInitFlag_ = true;

        if (runnerVariantPack.inTensors.size() != runnerVariantPack.isInTensorCanFree.size() ||
            runnerVariantPack.outTensors.size() != runnerVariantPack.isOutTensorNeedMalloc.size()) {
            ATB_LOG(ERROR) << "RunnerVariantPack error! inTensor size: " << runnerVariantPack.inTensors.size()
                           << ", isInTensorCanFree size: " << runnerVariantPack.isInTensorCanFree.size()
                           << ", outTensor size: " << runnerVariantPack.outTensors.size()
                           << ", isOutTensorNeedMalloc size: " << runnerVariantPack.isOutTensorNeedMalloc.size();
            return ERROR_INVALID_PARAM;
        }

        for (size_t i = 0; i < runnerGraph_.inTensors.size(); i++) {
            runnerGraph_.isInTensorCanFree[&runnerGraph_.inTensors.at(i)] = runnerVariantPack.isInTensorCanFree.at(i);
        }
        for (size_t i = 0; i < runnerGraph_.outTensors.size(); i++) {
            runnerGraph_.isOutTensorNeedMalloc[&runnerGraph_.outTensors.at(i)] =
                runnerVariantPack.isOutTensorNeedMalloc.at(i);
        }
        runnerGraph_.Init();
    }
    return NO_ERROR;
}

void GraphRunner::UpdateOutTensorDeviceData(RunnerVariantPack &runnerVariantPack)
{
    for (size_t i = 0; i < runnerGraph_.outTensors.size(); i++) {
        Tensor *outTensor = &runnerGraph_.outTensors.at(i);
        auto it = runnerGraph_.isOutTensorNeedMalloc.find(outTensor);
        if (it == runnerGraph_.isOutTensorNeedMalloc.end()) {
            ATB_LOG(ERROR) << GetLogPrefix() << "outTensor[" << i << "] isOutTensorNeedMalloc not found";
            if (!runnerVariantPack.outTensors.at(i).deviceData) {
                runnerVariantPack.outTensors.at(i).deviceData = outTensor->deviceData;
            }
        } else if (it->second) {
            runnerVariantPack.outTensors.at(i).deviceData = outTensor->deviceData;
        }
    }
}

void GraphRunner::FreeUselessInTensor()
{
    auto it = runnerGraph_.maxNodeIdTensorMap.find(FREE_TENSOR_KEY);
    if (it != runnerGraph_.maxNodeIdTensorMap.end()) {
        ATB_LOG(INFO) << "free useless intensor at the beginning of setup";
        for (auto tensorIt : it->second) {
            ATB_LOG(INFO) << GetLogPrefix() << "free tensor:" << tensorIt;
            memAllocationSolver_->Free((uint8_t *)tensorIt->deviceData);
        }
    }
}

Status GraphRunner::SetupNodes(const RunnerVariantPack &runnerVariantPack)
{
    uint8_t *nodeHostTilingBuffer = runnerVariantPack.hostTilingBuffer;
    uint64_t maxTilingSize = runnerVariantPack.tilingBufferSize;
    Status st = NO_ERROR;
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);

        st = PreparseNodeVariantPack(nodeId, node, runnerVariantPack, nodeHostTilingBuffer, maxTilingSize);
        if (st != 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << "node[" << nodeId
                           << "] setup fail, PreparseNodeVariantPack fail, error code:" << st;
            return st;
        }

        st = SetupNodeRunners(nodeId, node);
        if (st != 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << "node[" << nodeId << "] SetupNodeRunners fail, error code:" << st;
            return st;
        }
        uint64_t nodeTilingSize = node.runner->GetTilingBufferSize();
        nodeHostTilingBuffer += nodeTilingSize;
        maxTilingSize = maxTilingSize > nodeTilingSize ? (maxTilingSize - nodeTilingSize) : 0;
    }

    ATB_LOG(INFO) << GetLogPrefix() << " setup all node success";
    return NO_ERROR;
}

Status GraphRunner::SetupImpl(RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "setup start";
    if (GetSingleton<Config>().IsworkspaceMemAllocGlobal()) {
        memAllocationSolver_ = GetGlobalMemAllocationSolver();
    }
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        ReserveSvector(node);
    }
    if (runnerGraph_.inTensors.size() != runnerVariantPack.inTensors.size() ||
        runnerGraph_.outTensors.size() != runnerVariantPack.outTensors.size()) {
        ATB_LOG(FATAL) << GetLogPrefix() << "runnerVariantPack tensor size not graph tensor size";
        return ERROR_INVALID_GRAPH;
    }

    Status st = InitTensorFromRunnerVariantPack(runnerVariantPack);
    if (st != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "setup fail, InitTensorFromRunnerVariantPack fail, error code:" << st;
        return st;
    }

    Reset();
    FreeUselessInTensor();

    st = SetupNodes(runnerVariantPack);
    if (st != NO_ERROR) {
        ATB_LOG(INFO) << GetLogPrefix() << "runner graph setup nodes fail, error code:" << st;
        return st;
    }

    if (!GetSingleton<Config>().IsworkspaceMemAllocGlobal()) {
        selfIntermediateBufferSize_ =
            memAllocationSolver_->GetSize(); // 全局mem alloc时，selfIntermediateBufferSize_为0
    } else {
        UpdateOutTensorDeviceData(runnerVariantPack); // 全局mem alloc时，更新在内部malloc的outTensor的地址偏移
    }
    ATB_LOG(INFO) << GetLogPrefix() << " malloc size:" << memAllocationSolver_->GetMallocSize()
                  << ", real size:" << memAllocationSolver_->GetSize();

    CalcTilingBufferSize();
    CalcIntermediateBufferSize();
    ATB_LOG(INFO) << GetLogPrefix() << "runner graph:\n" << runnerGraph_.ToString();
    return NO_ERROR;
}

uint64_t GraphRunner::GetTilingBufferSizeImpl()
{
    return totalTilingBufferSize_;
}

Status GraphRunner::FillHostTilingBufferImpl(uint8_t *hostTilingBuffer, uint64_t tilingBufferSize,
                                             ContextBase *context)
{
    uint64_t tilingOffset = 0;
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        Status ret = node.runner->FillHostTilingBuffer(hostTilingBuffer + tilingOffset,
                                                       tilingBufferSizes_.at(nodeId), context);
        if (ret != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "GraphRunner::FillHostTilingBufferImpl failed! error code:" << ret;
            return ret;
        }
        tilingOffset += tilingBufferSizes_.at(nodeId);
    }
    ATB_LOG(INFO) << GetLogPrefix() << "fill all node host tiling buffer, tilingBufferSize:" << tilingBufferSize;
    return NO_ERROR;
}

std::vector<uint64_t> &GraphRunner::GetWorkspaceBufferSize()
{
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        const std::vector<uint64_t> &runnerWorkspaceBufferSize = node.runner->GetWorkspaceBufferSize();
        for (size_t i = 0; i < runnerWorkspaceBufferSize.size(); ++i) {
            multiStreamWorkspaceSizes_.at(i) =
                std::max(multiStreamWorkspaceSizes_.at(i), runnerWorkspaceBufferSize.at(i));
        }
    }
    return multiStreamWorkspaceSizes_;
}

uint64_t GraphRunner::GetIntermediateBufferSizeImpl()
{
    return selfIntermediateBufferSize_ + maxIntermediateBufferSize_;
}

Status GraphRunner::ExecuteImpl(RunnerVariantPack &runnerVariantPack)
{
    Status st = ExecuteAllRunner(runnerVariantPack);
    if (st != 0) {
        ATB_LOG(INFO) << GetLogPrefix() << "execute fail";
        return st;
    }

    ATB_LOG(INFO) << GetLogPrefix() << "execute success";
    return NO_ERROR;
}

Status GraphRunner::PreExecuteImpl(RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << " Execute start, runnerGraph_.nodes:" << runnerGraph_.nodes.size();

    if (runnerGraph_.inTensors.size() != runnerVariantPack.inTensors.size() ||
        runnerGraph_.outTensors.size() != runnerVariantPack.outTensors.size()) {
        ATB_LOG(FATAL) << GetLogPrefix()
                       << "runnerVariantPack tensor size not graph tensor size, runnerGraph_.inTensors:"
                       << runnerGraph_.inTensors.size()
                       << ", runnerVariantPack.inTensors:" << runnerVariantPack.inTensors.size()
                       << ", runnerGraph_.outTensors:" << runnerGraph_.outTensors.size()
                       << ", runnerVariantPack.outTensors:" << runnerVariantPack.outTensors.size();
        return ERROR_INVALID_GRAPH;
    }

    TensorUtil::FastCopyTensors(runnerVariantPack.inTensors, runnerGraph_.inTensors);
    TensorUtil::FastCopyTensors(runnerVariantPack.outTensors, runnerGraph_.outTensors);

    UpdateVariantPackBuffer(runnerVariantPack);
    UpdateVariantPackTensorData(runnerVariantPack);

    Status st = PreExecuteAllRunner(runnerVariantPack);
    if (st != 0) {
        ATB_LOG(INFO) << GetLogPrefix() << "execute fail";
        return st;
    }
    return NO_ERROR;
}

void GraphRunner::SetSaveTensorDir(const std::string &tensorDir)
{
    tensorDir_ = tensorDir;
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); nodeId++) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        node.runner->SetSaveTensorDir(tensorDir + "/" + std::to_string(nodeId) + "_" + node.runner->operationName_);
    }
}

void GraphRunner::Reset()
{
    selfIntermediateBufferSize_ = 0;
    totalTilingBufferSize_ = 0;
    tilingBufferSizes_.clear();
    maxIntermediateBufferSize_ = 0;
    intermediateBufferSizes_.clear();
    if (!GetSingleton<Config>().IsworkspaceMemAllocGlobal()) {
        memAllocationSolver_->Reset();
    }
    for (auto &tensor : runnerGraph_.internalTensors) {
        Tensor emptyTensor = {};
        tensor = emptyTensor;
    }
    runnerGraph_.tensorMalloced.clear();
}

Status GraphRunner::PreparseNodeVariantPack(size_t nodeId, GraphRunner::Node &node,
                                            const RunnerVariantPack &runnerVariantPack, uint8_t *nodeHostTilingBuffer,
                                            uint64_t maxTilingSize)
{
    node.runnerVariantPack.inTensors.resize(node.inTensors.size());
    node.runnerVariantPack.outTensors.resize(node.outTensors.size());
    node.runnerVariantPack.isInTensorCanFree.resize(node.inTensors.size());
    node.runnerVariantPack.isOutTensorNeedMalloc.resize(node.outTensors.size());
    node.runnerVariantPack.context = runnerVariantPack.context;
    node.runnerVariantPack.hostTilingBuffer = nodeHostTilingBuffer;
    node.runnerVariantPack.tilingBufferSize = maxTilingSize;
    Status st = PreparseNodeInTensor(nodeId, node);
    if (st != NO_ERROR) {
        return st;
    }

    st = PreparseViewNodeInTensorInferShape(node);
    if (st != NO_ERROR) {
        return st;
    }

    st = PreparseNodeOutTensor(nodeId, node);
    if (st != NO_ERROR) {
        return st;
    }

    st = PreparseViewNodeInTensor(node);
    if (st != NO_ERROR) {
        return st;
    }
    return NO_ERROR;
}

Tensor GraphRunner::RunInTensorReshapeFuncs(size_t nodeId, GraphRunner::Node &node, size_t inTensorId) const
{
    if (inTensorId < node.inTensorReshapeFuncs.size() && node.inTensorReshapeFuncs.at(inTensorId)) {
        Tensor reshapeTensor = *node.inTensors.at(inTensorId);
        reshapeTensor.desc.shape.dimNum = 0;
        node.inTensorReshapeFuncs.at(inTensorId)(node.inTensors.at(inTensorId)->desc.shape, reshapeTensor.desc.shape);
        if (Utils::GetTensorNumel(reshapeTensor) != Utils::GetTensorNumel(*node.inTensors.at(inTensorId))) {
            ATB_LOG(ERROR) << GetLogPrefix() << "node[" << nodeId
                           << "] invalid reshape func, reshapeTensor.Numel:" << Utils::GetTensorNumel(reshapeTensor)
                           << ", tensor.Numel:" << Utils::GetTensorNumel(*node.inTensors.at(inTensorId));
            return reshapeTensor;
        }
        ATB_LOG(INFO) << GetLogPrefix() << " node[" << nodeId << "] reshape inTensor[" << inTensorId
                      << "], old:" << TensorUtil::ShapeToString(node.inTensors.at(inTensorId)->desc.shape)
                      << ", new:" << TensorUtil::ShapeToString(reshapeTensor.desc.shape);
        return reshapeTensor;
    }
    return *node.inTensors.at(inTensorId);
}

Status GraphRunner::PreparseViewNodeInTensorInferShape(Node &node) const
{
    if (node.inTensorChunks.size() == 0) {
        return NO_ERROR;
    }
    if (node.inTensorChunks.size() != node.inTensors.size()) {
        ATB_LOG(ERROR) << GetLogPrefix() << "node inTensorsChunks shoule be equal to node.inTensors.size()! "
                       << ",node.inTensorChunks.size(): " << node.inTensorChunks.size()
                       << ",node.inTensors.size(): " << node.inTensorChunks.size();
        return ERROR_INVALID_PARAM;
    }
    for (size_t i = 0; i < node.inTensors.size(); i++) {
        if (node.inTensorChunks.at(i).chunkNum == 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << "chunkNum cannot be equal to 0!"
                           << "chunkNum is: " << node.inTensorChunks.at(i).chunkNum << " , please check [" << i
                           << "], intensor!";
            return ERROR_INVALID_PARAM;
        }

        if (node.inTensors.at(i)->desc.shape.dims[0] % static_cast<int>(node.inTensorChunks.at(i).chunkNum) != 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << "dim[0] can not be chunked by chunkNum! "
                           << "intensors dim[0] shape is: " << node.inTensors.at(i)->desc.shape.dims[0]
                           << ", chunkIndex is: " << node.inTensorChunks.at(i).chunkNum;
            return ERROR_INVALID_PARAM;
        }
        node.runnerVariantPack.inTensors.at(i).desc.shape.dims[0] /=
            static_cast<int>(node.inTensorChunks.at(i).chunkNum);
    }
    return NO_ERROR;
}

Status GraphRunner::PreparseNodeInTensor(size_t nodeId, GraphRunner::Node &node)
{
    for (size_t i = 0; i < node.inTensors.size(); ++i) {
        node.runnerVariantPack.inTensors.at(i) = RunInTensorReshapeFuncs(nodeId, node, i);

        auto it = runnerGraph_.tensorMaxNodeIdMap.find(node.inTensors.at(i));
        if (GetSingleton<Config>().IsworkspaceMemAllocGlobal() && it != runnerGraph_.tensorMaxNodeIdMap.end() &&
            it->second == nodeId) {
            node.runnerVariantPack.isInTensorCanFree.at(i) = true;
        } else {
            node.runnerVariantPack.isInTensorCanFree.at(i) = false;
        }
    }
    return NO_ERROR;
}

Status GraphRunner::PreparseViewNodeInTensor(GraphRunner::Node &node) const
{
    if (node.inTensorChunks.size() == 0) {
        return NO_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "start preparse view node intensors!";
    for (size_t i = 0; i < node.runnerVariantPack.inTensors.size(); i++) {
        uint8_t *charDeviceData = reinterpret_cast<uint8_t *>(node.runnerVariantPack.inTensors.at(i).deviceData);
        TensorDesc &intensorDesc = node.runnerVariantPack.inTensors.at(i).desc;
        charDeviceData += atb::Utils::GetTensorSize(intensorDesc) * node.inTensorChunks.at(i).chunkIndex;
        node.runnerVariantPack.inTensors.at(i).deviceData = reinterpret_cast<void *>(charDeviceData);
        node.runnerVariantPack.inTensors.at(i).dataSize = TensorUtil::CalcTensorDataSize(intensorDesc);
    }
    return NO_ERROR;
}

void GraphRunner::WriteInPlaceCheck(TensorDesc &oriDesc, TensorDesc &newDesc, size_t nodeId, size_t tensorId,
                                    const std::string &tensorType) const
{
    if (oriDesc.dtype == newDesc.dtype && oriDesc.format == newDesc.format &&
        Utils::GetTensorNumel(oriDesc) == Utils::GetTensorNumel(newDesc)) {
        ATB_LOG(INFO) << GetLogPrefix() << "node[" << nodeId << "] outTensors[" << tensorId << "] is " << tensorType
                      << ", write in place";
    } else {
        ATB_LOG(ERROR) << GetLogPrefix() << "node[" << nodeId << "] outTensors[" << tensorId << "] is " << tensorType
                       << ", write in place, but tensordesc has changed, ori tensordesc: "
                       << TensorUtil::TensorDescToString(oriDesc)
                       << ", cur tensordesc: " << TensorUtil::TensorDescToString(newDesc);
    }
}

void GraphRunner::NodeOutTensorGlobalMemAlloc(size_t nodeId, GraphRunner::Node &node,
                                              SVector<TensorDesc> &outTensorDescs)
{
    // 全局mem alloc solver时，tensor内存的申请释放全部下发至叶子节点完成
    for (size_t i = 0; i < node.outTensors.size(); ++i) {
        Tensor *outTensor = node.outTensors.at(i);
        outTensor->desc = outTensorDescs.at(i);
        if (node.outTensorTypes.at(i) == GraphRunner::INTERMEDIATE_TENSOR) {
            // 中间tensor已被malloc，原地写
            if (runnerGraph_.tensorMalloced.find(outTensor) != runnerGraph_.tensorMalloced.end()) {
                WriteInPlaceCheck(outTensor->desc, outTensorDescs.at(i), nodeId, i,
                                  std::string("graph internal tensor"));
                node.runnerVariantPack.isOutTensorNeedMalloc.at(i) = false;
            } else {
                runnerGraph_.tensorMalloced.insert(outTensor);
                outTensor->dataSize = TensorUtil::CalcTensorDataSize(*outTensor);
                node.runnerVariantPack.isOutTensorNeedMalloc.at(i) = true;
                ATB_LOG(INFO) << GetLogPrefix() << "node[" << nodeId << "] outTensors[" << i << "] is internal tensor";
            }
        } else {
            auto it = runnerGraph_.isOutTensorNeedMalloc.find(outTensor);
            if (it == runnerGraph_.isOutTensorNeedMalloc.end()) { // outtensor为graph intensor，原地写
                WriteInPlaceCheck(outTensor->desc, outTensorDescs.at(i), nodeId, i, std::string("graph intensor"));
                node.runnerVariantPack.isOutTensorNeedMalloc.at(i) = false;
            } else {
                if (runnerGraph_.tensorMalloced.find(outTensor) == runnerGraph_.tensorMalloced.end()) {
                    if (it->second) {
                        runnerGraph_.tensorMalloced.insert(outTensor);
                    }
                    node.runnerVariantPack.isOutTensorNeedMalloc.at(i) = it->second;
                    ATB_LOG(INFO) << GetLogPrefix() << "node[" << nodeId << "] outTensors[" << i
                                  << "] is graph outtensor, isOutTensorNeedMalloc: "
                                  << node.runnerVariantPack.isOutTensorNeedMalloc.at(i);
                } else { // outtensor为graph outtensor，且已被malloc，原地写
                    WriteInPlaceCheck(outTensor->desc, outTensorDescs.at(i), nodeId, i, std::string("graph outtensor"));
                    node.runnerVariantPack.isOutTensorNeedMalloc.at(i) = false;
                }
            }
        }
        node.runnerVariantPack.outTensors.at(i) = *outTensor;
        ATB_LOG(INFO) << GetLogPrefix() << "node[" << nodeId << "] " << node.runner->GetName() << " outTensors[" << i
                      << "] " << TensorUtil::TensorToString(*outTensor);
    }
}

void GraphRunner::NodeOutTensorLocalMemAlloc(size_t nodeId, GraphRunner::Node &node,
                                             SVector<TensorDesc> &outTensorDescs)
{
    for (size_t i = 0; i < node.outTensors.size(); ++i) {
        Tensor *outTensor = node.outTensors.at(i);
        outTensor->desc = outTensorDescs.at(i);
        if (node.outTensorTypes.at(i) == GraphRunner::INTERMEDIATE_TENSOR) {
            // 中间tensor已被malloc，原地写
            if (runnerGraph_.tensorMalloced.find(outTensor) != runnerGraph_.tensorMalloced.end()) {
                WriteInPlaceCheck(outTensor->desc, outTensorDescs.at(i), nodeId, i,
                                  std::string("graph internal tensor"));
            } else {
                runnerGraph_.tensorMalloced.insert(outTensor);
                outTensor->dataSize = TensorUtil::CalcTensorDataSize(*outTensor);
                outTensor->deviceData =
                    memAllocationSolver_->GetOffset(TensorUtil::AlignInt(outTensor->dataSize, ALIGN_INT));
                ATB_LOG(INFO) << GetLogPrefix() << "node[" << nodeId << "] " << node.runner->GetName()
                              << " malloc outTensors[" << i << "] dataSize:" << outTensor->dataSize;
#ifdef _DEBUG
                ATB_LOG(INFO) << " data :" << reinterpret_cast<uint64_t>(outTensor->deviceData);
#endif
            }
        } else {
            ATB_LOG(INFO) << GetLogPrefix() << "node[" << nodeId << "] outTensors[" << i << "] is not internal tensor";
        }
        node.runnerVariantPack.outTensors.at(i) = *outTensor;
        node.runnerVariantPack.isOutTensorNeedMalloc.at(i) = false; // 本地mem alloc时，所有outtensor由外部申请
        ATB_LOG(INFO) << GetLogPrefix() << "node[" << nodeId << "] " << node.runner->GetName() << " outTensors[" << i
                      << "] " << TensorUtil::TensorToString(*outTensor);
    }

    auto it = runnerGraph_.maxNodeIdTensorMap.find(nodeId);
    if (it != runnerGraph_.maxNodeIdTensorMap.end()) {
        for (auto tensorIt : it->second) {
            ATB_LOG(INFO) << GetLogPrefix() << " node[" << nodeId << "] " << node.runner->GetName()
                          << " free tensor:" << tensorIt;
            memAllocationSolver_->Free((uint8_t *)tensorIt->deviceData);
        }
    }
}

Status GraphRunner::InferShapeNode(size_t nodeId, Node &node) const
{
    node.lastOutTensorDescs.reserve(node.op->GetOutputNum());
    node.lastOutTensorDescs.resize(node.op->GetOutputNum());
    node.lastInTensorDescs.reserve(node.runnerVariantPack.inTensors.size());
    node.lastInTensorDescs.resize(node.runnerVariantPack.inTensors.size());
    for (size_t i = 0; i < node.runnerVariantPack.inTensors.size(); i++) {
        node.lastInTensorDescs.at(i) = node.runnerVariantPack.inTensors.at(i).desc;
    }
    Status st = node.op->InferShape(node.lastInTensorDescs, node.lastOutTensorDescs);
    if (st != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "node[" << nodeId << "] infer shape fail, error code: " << st;
        return st;
    }

    for (size_t i = 0; i < node.lastOutTensorDescs.size(); ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << "node[" << nodeId << "] " << node.runner->GetName() << " outTensorDescs["
                      << i << "] " << TensorUtil::TensorDescToString(node.lastOutTensorDescs.at(i));
    }
    ATB_LOG(INFO) << GetLogPrefix() << "node[" << nodeId << "] infer shape end";
    if (node.lastOutTensorDescs.size() != node.outTensors.size()) {
        ATB_LOG(ERROR) << GetLogPrefix() << "node[" << nodeId << "] infer shape outtensor not euqal node outtensor";
        return ERROR_INVALID_PARAM;
    }
    return NO_ERROR;
}

Status GraphRunner::PreparseNodeOutTensor(size_t nodeId, GraphRunner::Node &node)
{
    ATB_LOG(INFO) << GetLogPrefix() << " node[" << nodeId << "] infer shape start";
    for (size_t i = 0; i < node.runnerVariantPack.inTensors.size(); ++i) {
        ATB_LOG(INFO) << GetLogPrefix() << " node[" << nodeId << "] " << node.runner->GetName() << " intensor[" << i
                      << "] " << TensorUtil::TensorToString(node.runnerVariantPack.inTensors.at(i));
    }

    Status st = NO_ERROR;
    if (!TensorUtil::TensorDescsEqual(node.runnerVariantPack.inTensors, node.lastInTensorDescs) ||
        node.runnerVariantPack.inTensors.empty()) {
        st = InferShapeNode(nodeId, node);
        if (st != NO_ERROR) {
            return st;
        }
    }

    if (GetSingleton<Config>().IsworkspaceMemAllocGlobal()) {
        NodeOutTensorGlobalMemAlloc(nodeId, node, node.lastOutTensorDescs);
    } else {
        NodeOutTensorLocalMemAlloc(nodeId, node, node.lastOutTensorDescs);
    }

    return NO_ERROR;
}

bool GraphRunner::CheckRunnerBase(size_t nodeId, GraphRunner::Node &node)
{
    if (node.runner->IsSupportGlbWorkspace()) {
        return true;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "node[" << nodeId << "] " << node.runner->GetName()
                  << " is not GraphRunner or OpsRunner, malloc outtensor before setup";
    for (size_t i = 0; i < node.outTensors.size(); i++) {
        if (node.runnerVariantPack.isOutTensorNeedMalloc.at(i)) {
            Tensor *outTensor = node.outTensors.at(i);
            outTensor->deviceData =
                memAllocationSolver_->GetOffset(TensorUtil::AlignInt(outTensor->dataSize, ALIGN_INT));
#ifdef _DEBUG
            ATB_LOG(INFO) << GetLogPrefix() << "node[" << nodeId << "] " << node.runner->GetName()
                          << " malloc outTensors[" << i << "] dataSize:" << outTensor->dataSize
                          << ", data:" << reinterpret_cast<uint64_t>(outTensor->deviceData);
#else
            ATB_LOG(INFO) << GetLogPrefix() << "node[" << nodeId << "] " << node.runner->GetName()
                          << " malloc outTensors[" << i << "] dataSize:" << outTensor->dataSize;
#endif
            node.runnerVariantPack.outTensors.at(i).deviceData = outTensor->deviceData;
        }
    }
    return false;
}

void GraphRunner::FreeInTensorAfterSetup(size_t nodeId, GraphRunner::Node &node)
{
    ATB_LOG(INFO) << GetLogPrefix() << "node[" << nodeId << "] " << node.runner->GetName()
                  << " is not GraphRunner or OpsRunner, free intensor after setup";
    for (size_t i = 0; i < node.inTensors.size(); i++) {
        if (node.runnerVariantPack.isInTensorCanFree.at(i)) {
            Tensor *inTensor = node.inTensors.at(i);
#ifdef _DEBUG
            ATB_LOG(INFO) << GetLogPrefix() << " node[" << nodeId << "] " << node.runner->GetName()
                          << " free intensor:" << inTensor;
#else
            ATB_LOG(INFO) << GetLogPrefix() << " node[" << nodeId << "] " << node.runner->GetName()
                          << " free intensor.";
#endif
            memAllocationSolver_->Free((uint8_t *)inTensor->deviceData);
        }
    }
}

Status GraphRunner::SetupNodeRunners(size_t nodeId, GraphRunner::Node &node)
{
    ATB_LOG(INFO) << GetLogPrefix() << " node[" << nodeId << "] setup start";

    bool isRunnerSupportGlobalMemAlloc = true; // global mem alloc时，需判断runner是否支持内部申请释放非中间tensor
    if (GetSingleton<Config>().IsworkspaceMemAllocGlobal()) {
        isRunnerSupportGlobalMemAlloc = CheckRunnerBase(nodeId, node);
    }

    node.runner->SetRunnerOperation(node.op.get());

    Status st = node.runner->Setup(node.runnerVariantPack);
    if (st != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << " node[" << nodeId << "] setup fail, error code:" << st;
        return st;
    }
    if (!isRunnerSupportGlobalMemAlloc) {
        FreeInTensorAfterSetup(nodeId, node);
    }

    if (GetSingleton<Config>().IsworkspaceMemAllocGlobal()) { // 更新内部malloc的tensor地址偏移
        for (size_t i = 0; i < node.runnerVariantPack.outTensors.size(); i++) {
            if (node.runnerVariantPack.isOutTensorNeedMalloc.at(i)) {
                node.outTensors.at(i)->deviceData = node.runnerVariantPack.outTensors.at(i).deviceData;
            }
        }
    }
    ATB_LOG(INFO) << GetLogPrefix() << " node[" << nodeId << "] setup success";

    return NO_ERROR;
}

void GraphRunner::CalcTilingBufferSize()
{
    totalTilingBufferSize_ = 0;
    tilingBufferSizes_.resize(runnerGraph_.nodes.size());
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        uint64_t runnerTilingBufferSize = node.runner->GetTilingBufferSize();
        ATB_LOG(INFO) << GetLogPrefix() << " node[" << nodeId << "] tiling buffer size:" << runnerTilingBufferSize;
        totalTilingBufferSize_ += runnerTilingBufferSize;
        tilingBufferSizes_.at(nodeId) = runnerTilingBufferSize;
    }
    ATB_LOG(INFO) << GetLogPrefix() << " total node tiling buffer size:" << totalTilingBufferSize_;
}

void GraphRunner::CalcIntermediateBufferSize()
{
    maxIntermediateBufferSize_ = 0;
    intermediateBufferSizes_.resize(runnerGraph_.nodes.size());
    if (GetSingleton<Config>().IsworkspaceMemAllocGlobal()) { // 全局mem alloc时，所有runner共用一份内存
        maxIntermediateBufferSize_ = memAllocationSolver_->GetSize();
        for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
            intermediateBufferSizes_.at(nodeId) = maxIntermediateBufferSize_;
        }
        return;
    }
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        uint64_t runnerIntermediateBufferSize = node.runner->GetIntermediateBufferSize();
        intermediateBufferSizes_.at(nodeId) = runnerIntermediateBufferSize;
        maxIntermediateBufferSize_ = std::max(maxIntermediateBufferSize_, runnerIntermediateBufferSize);
        ATB_LOG(INFO) << GetLogPrefix() << " node[" << nodeId
                      << "] intermediate buffer size:" << runnerIntermediateBufferSize
                      << ", max:" << maxIntermediateBufferSize_;
    }
}

void GraphRunner::UpdateVariantPackBuffer(RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << " update runner variant pack's buffer start";
    if (totalTilingBufferSize_ != 0) {
        uint64_t offset = 0;
        for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
            auto &node = runnerGraph_.nodes.at(nodeId);
            node.runnerVariantPack.tilingBuffer = runnerVariantPack.tilingBuffer + offset;
            node.runnerVariantPack.tilingBufferSize = tilingBufferSizes_.at(nodeId);
            offset += tilingBufferSizes_.at(nodeId);
        }
    } else {
        ATB_LOG(WARN) << GetLogPrefix() << " totalTilingBufferSize is 0, not update variantPack's tilingBuffer";
    }

    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        node.runnerVariantPack.workspaceBuffer = runnerVariantPack.workspaceBuffer;
    }

    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        node.runnerVariantPack.intermediateBuffer = runnerVariantPack.intermediateBuffer + selfIntermediateBufferSize_;
        node.runnerVariantPack.intermediateBufferSize = intermediateBufferSizes_.at(nodeId);
    }

    if (runnerVariantPack.argsDeviceBuffer != nullptr) {
        uint64_t offset = 0;
        for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
            auto &node = runnerGraph_.nodes.at(nodeId);
            node.runnerVariantPack.argsDeviceBuffer = runnerVariantPack.argsDeviceBuffer + offset;
            offset += node.runner->GetArgsSize();
            ATB_LOG(DEBUG) << GetLogPrefix() << "Graph node " << nodeId << " argsDeviceAddr is "
                << reinterpret_cast<void *>(node.runnerVariantPack.argsDeviceBuffer);
        }
    }

    if (runnerVariantPack.argsHostBuffer != nullptr) {
        uint64_t offset = 0;
        for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
            auto &node = runnerGraph_.nodes.at(nodeId);
            node.runnerVariantPack.argsHostBuffer = runnerVariantPack.argsHostBuffer + offset;
            offset += node.runner->GetArgsSize();
            ATB_LOG(DEBUG) << GetLogPrefix() << "Graph node " << nodeId << " argsHostAddr is "
                << reinterpret_cast<void *>(node.runnerVariantPack.argsHostBuffer);
        }
    }
    ATB_LOG(INFO) << GetLogPrefix() << " update runner variant pack's buffer end";
}

static std::string GetUpdateTensorDataString(size_t nodeId, std::string tensorFlag, size_t tensorId,
                                             std::string tensorType, Tensor &tensor)
{
    std::stringstream ss;
    ss << " update node[" << nodeId << "] " << tensorFlag << "[" << tensorId << "] is " << tensorType << ".";
    ss << " dataSize:" << tensor.dataSize;
#ifdef _DEBUG
    ss << " deviceData:" << tensor.deviceData << ", hostData:" << tensor.hostData;
#endif
    return ss.str();
}

void GraphRunner::UpdateVariantPackTensorData(RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << " update runner variant pack's tensor data start";
    uint8_t *selfIntermediateBuffer = runnerVariantPack.intermediateBuffer;

    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        ATB_LOG(INFO) << GetLogPrefix() << " update tensor.data node[" << nodeId << "]";
        for (size_t i = 0; i < node.runnerVariantPack.inTensors.size(); ++i) {
            auto &tensor = node.runnerVariantPack.inTensors.at(i);
            if (node.inTensorTypes.at(i) == GraphRunner::INTERMEDIATE_TENSOR) {
                tensor.deviceData = selfIntermediateBuffer + reinterpret_cast<uint64_t>(tensor.deviceData);
                ATB_LOG(INFO) << GetLogPrefix() << GetUpdateTensorDataString(nodeId, "intensor", i, "internal", tensor);
            } else {
                tensor.deviceData = node.inTensors.at(i)->deviceData;
                tensor.hostData = node.inTensors.at(i)->hostData;
                ATB_LOG(INFO) << GetLogPrefix()
                              << GetUpdateTensorDataString(nodeId, "intensor", i, "not internal", tensor);
            }
        }
        for (size_t i = 0; i < node.runnerVariantPack.outTensors.size(); ++i) {
            auto &tensor = node.runnerVariantPack.outTensors.at(i);
            if (node.outTensorTypes.at(i) == GraphRunner::INTERMEDIATE_TENSOR) {
                tensor.deviceData = selfIntermediateBuffer + reinterpret_cast<uint64_t>(tensor.deviceData);
                ATB_LOG(INFO) << GetLogPrefix()
                              << GetUpdateTensorDataString(nodeId, "outtensor", i, "internal", tensor);
            } else {
                tensor.deviceData = node.outTensors.at(i)->deviceData;
                tensor.hostData = node.outTensors.at(i)->hostData;
                ATB_LOG(INFO) << GetLogPrefix()
                              << GetUpdateTensorDataString(nodeId, "outtensor", i, "not internal", tensor);
            }
        }
    }
    ATB_LOG(INFO) << GetLogPrefix() << " update runner variant pack's tensor data end";
}


Status GraphRunner::ExecuteAllRunner(RunnerVariantPack &runnerVariantPack)
{
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        ATB_LOG(INFO) << GetLogPrefix() << " mstx registe tensor.data node[" << nodeId << "]" << "graphrunner start";
        if (runnerVariantPack.mstxMemRegister != nullptr && !(dynamic_cast<GraphRunner*>(node.runner.get()))) {
            runnerVariantPack.mstxMemRegister->ClearMstxMemRegions();
            for (size_t i = 0; i < node.runnerVariantPack.inTensors.size(); ++i) {
                auto &tensor = node.runnerVariantPack.inTensors.at(i);
                if (node.inTensorTypes.at(i) == GraphRunner::INTERMEDIATE_TENSOR) {
                    runnerVariantPack.mstxMemRegister->AddTensorMemRegions(tensor.deviceData, static_cast<uint64_t>(TensorUtil::AlignInt(tensor.dataSize, ALIGN_INT)));
                }
            }
            for (size_t i = 0; i < node.runnerVariantPack.outTensors.size(); ++i) {
                auto &tensor = node.runnerVariantPack.outTensors.at(i);
                if (node.outTensorTypes.at(i) == GraphRunner::INTERMEDIATE_TENSOR) {
                    runnerVariantPack.mstxMemRegister->AddTensorMemRegions(tensor.deviceData, static_cast<uint64_t>(TensorUtil::AlignInt(tensor.dataSize, ALIGN_INT)));
                }
            }
            if (runnerVariantPack.mstxMemRegister->CheckTensorRange()) {
                runnerVariantPack.mstxMemRegister->MstxMemRegionsRegister();
            }
        }
        ATB_LOG(INFO) << GetLogPrefix() << " node[" << nodeId << "] execute start, runner:" << node.runner->GetName()
                                        << ", variantPack:\n"
                                        << node.runnerVariantPack.ToString();
        node.runnerVariantPack.context = runnerVariantPack.context;
        node.runnerVariantPack.mstxMemRegister = runnerVariantPack.mstxMemRegister;
        Status st = node.runner->Execute(node.runnerVariantPack);
        if (st != 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << " node[" << nodeId
            << "] execute fail, runner name:" << node.runner->GetName();
            return st;
        }
        if (runnerVariantPack.mstxMemRegister != nullptr && runnerVariantPack.mstxMemRegister->CheckTensorRange()) {
            runnerVariantPack.mstxMemRegister->MstxMemRegionsUnregister();
        }
    }

    return NO_ERROR;
}

Status GraphRunner::PreExecuteAllRunner(RunnerVariantPack &runnerVariantPack)
{
    for (size_t nodeId = 0; nodeId < runnerGraph_.nodes.size(); ++nodeId) {
        auto &node = runnerGraph_.nodes.at(nodeId);
        ATB_LOG(INFO) << GetLogPrefix() << " node[" << nodeId
                      << "] PreExecute start, runner:" << node.runner->GetName();
        node.runnerVariantPack.context = runnerVariantPack.context;
        // 此处将GraphRunner Setup阶段得到的multiStreamWorkspaceSizes_赋值给其图内的各个node
        // 让图内各个node可以拥有GraphRunner中计算好的各个streamId的workspaceBufferSize
        for (size_t i = 0; i < multiStreamWorkspaceSizes_.size(); ++i) {
            node.runner->multiStreamWorkspaceSizes_.at(i) = multiStreamWorkspaceSizes_.at(i);
        }
        Status st = node.runner->PreExecute(node.runnerVariantPack);
        if (st != 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << " node[" << nodeId
                           << "] execute fail, runner name:" << node.runner->GetName();
            return st;
        }
    }

    return NO_ERROR;
}

bool GraphRunner::IsSupportGlbWorkspace()
{
    return true;
}

void GraphRunner::ChangeWorkspaceBufferByExecuteStream(RunnerVariantPack &runnerVariantPack)
{
    (void)runnerVariantPack;
}

uint64_t GraphRunner::GetArgsSize()
{
    uint64_t argsSize = 0;
    for (size_t i = 0; i < runnerGraph_.nodes.size(); i++) {
        argsSize += runnerGraph_.nodes.at(i).runner->GetArgsSize();
    }
    return argsSize;
}

Status GraphRunner::BuildArgs()
{
    Status st = NO_ERROR;
    for (size_t i = 0; i < runnerGraph_.nodes.size(); i++) {
        st = runnerGraph_.nodes.at(i).runner->BuildArgs();
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "BuildArgs failed!";
            return st;
        }
    }
    return st;
}

Status GraphRunner::UpdateTensorAddr(RunnerVariantPack &runnerVariantPack)
{
    TensorUtil::FastCopyTensors(runnerVariantPack.inTensors, runnerGraph_.inTensors);
    TensorUtil::FastCopyTensors(runnerVariantPack.outTensors, runnerGraph_.outTensors);

    UpdateVariantPackTensorData(runnerVariantPack);

    for (size_t i = 0; i < runnerGraph_.nodes.size(); i++) {
        auto &node = runnerGraph_.nodes.at(i);
        node.runnerVariantPack.intermediateBuffer = runnerVariantPack.intermediateBuffer;
        node.runner->UpdateTensorAddr(node.runnerVariantPack);
    }
    return NO_ERROR;
}

Status GraphRunner::UpdateWorkspaceBuffer(RunnerVariantPack &runnerVariantPack)
{
    for (size_t i = 0; i < runnerGraph_.nodes.size(); i++) {
        auto &node = runnerGraph_.nodes.at(i);
        node.runnerVariantPack.workspaceBuffer = runnerVariantPack.workspaceBuffer;
        node.runner->UpdateWorkspaceBuffer(node.runnerVariantPack);
    }
    return NO_ERROR;
}
} // namespace atb