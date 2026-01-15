/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_GRAPH_RUNNER_H
#define ATB_GRAPH_RUNNER_H
#include <map>
#include <set>
#include <functional>
#include <memory>
#include "atb/svector.h"
#include "atb/operation.h"
#include "atb/utils/mem_allocation_solver/mem_allocation_solver.h"
#include "runner.h"

namespace atb {

class GraphRunner : public Runner {
public:
    enum TensorType : int {
        INTERMEDIATE_TENSOR = 0,
        NOT_INTERMEDIATE_TENSOR,
    };
    struct Node {
        std::shared_ptr<Operation> op;
        std::shared_ptr<Runner> runner;
        SVector<Tensor *> inTensors;
        SVector<Tensor *> outTensors;
        SVector<ReshapeFunc> inTensorReshapeFuncs;
        RunnerVariantPack runnerVariantPack;
        SVector<TensorType> inTensorTypes;
        SVector<TensorType> outTensorTypes;
        SVector<Chunk> inTensorChunks;
        SVector<TensorDesc> lastInTensorDescs;
        SVector<TensorDesc> lastOutTensorDescs;
    };

    struct Graph {
        SVector<Tensor> inTensors;
        SVector<Tensor> outTensors;
        SVector<Tensor> internalTensors;
        std::vector<Node> nodes;
        std::map<Tensor *, uint64_t> tensorMaxNodeIdMap;
        std::map<uint64_t, std::set<Tensor *>> maxNodeIdTensorMap;
        std::map<Tensor *, bool> isInTensorCanFree;
        std::map<Tensor *, bool> isOutTensorNeedMalloc;
        std::set<Tensor *> tensorMalloced;
        std::string ToString() const;
        void Init();

    private:
        void InitTensorMaxNodeMap();
        void InitTensorType();
        void SetNonReuseTensors();
        bool IsInternalTensor(const Tensor *tensor);
        void SearchTensorInNodeInTensor(const Tensor *tensor, uint64_t &maxNodeId, uint64_t &dependNodeCount);
        bool SearchTensorInNodeOutTensor(const Tensor *tensor, uint64_t &maxNodeId);
    };

    explicit GraphRunner(const std::string &name);
    ~GraphRunner() override;
    Graph &GetGraph();
    void ReserveSvector(GraphRunner::Node &node);
    bool IsSupportGlbWorkspace() override;
    uint64_t GetArgsSize() override;
    Status BuildArgs() override;
    Status UpdateTensorAddr(RunnerVariantPack &runnerVariantPack) override;
    Status UpdateWorkspaceBuffer(RunnerVariantPack &runnerVariantPack) override;

protected:
    Status SetupImpl(RunnerVariantPack &runnerVariantPack) override;
    uint64_t GetTilingBufferSizeImpl() override;
    Status FillHostTilingBufferImpl(uint8_t *hostTilingBuffer, uint64_t tilingBufferSize,
                                    ContextBase *context) override;
    std::vector<uint64_t> &GetWorkspaceBufferSize() override;
    uint64_t GetIntermediateBufferSizeImpl() override;
    Status ExecuteImpl(RunnerVariantPack &runnerVariantPack) override;
    Status PreExecuteImpl(RunnerVariantPack &runnerVariantPack) override;
    void SetSaveTensorDir(const std::string &tensorDir) override;
    void ChangeWorkspaceBufferByExecuteStream(RunnerVariantPack &runnerVariantPack) override;

private:
    void Reset();
    void FreeUselessInTensor();
    Status InitTensorFromRunnerVariantPack(RunnerVariantPack &runnerVariantPack);
    Status PreparseNodeVariantPack(size_t nodeId, GraphRunner::Node &node, const RunnerVariantPack &runnerVariantPack,
                                   uint8_t *nodeHostTilingBuffer, uint64_t maxTilingSize);
    Status PreparseNodeInTensor(size_t nodeId, Node &node);
    Status PreparseViewNodeInTensor(Node &node) const;
    Status PreparseViewNodeInTensorInferShape(Node &node) const;
    Status PreparseNodeOutTensor(size_t nodeId, Node &node);
    Tensor RunInTensorReshapeFuncs(size_t nodeId, Node &node, size_t inTensorId) const;
    Status InferShapeNode(size_t nodeId, Node &node) const;
    void WriteInPlaceCheck(TensorDesc &oriDesc, TensorDesc &newDesc, size_t nodeId, size_t tensorId,
                           const std::string &tensorType) const;
    void NodeOutTensorGlobalMemAlloc(size_t nodeId, Node &node, SVector<TensorDesc> &outTensorDescs);
    void NodeOutTensorLocalMemAlloc(size_t nodeId, Node &node, SVector<TensorDesc> &outTensorDescs);

    Status SetupNodeRunners(size_t nodeId, Node &node);
    Status SetupNodes(const RunnerVariantPack &runnerVariantPack);
    bool CheckRunnerBase(size_t nodeId, Node &node);
    void FreeInTensorAfterSetup(size_t nodeId, Node &node);
    void UpdateOutTensorDeviceData(RunnerVariantPack &runnerVariantPack);
    void CalcTilingBufferSize();
    void CalcIntermediateBufferSize();
    void UpdateVariantPackBuffer(RunnerVariantPack &runnerVariantPack);
    void UpdateVariantPackTensorData(RunnerVariantPack &runnerVariantPack);
    Status ExecuteAllRunner(RunnerVariantPack &runnerVariantPack);
    Status PreExecuteAllRunner(RunnerVariantPack &runnerVariantPack);

private:
    Graph runnerGraph_;
    bool runnerGraphInitFlag_ = false;
    uint64_t selfIntermediateBufferSize_ = 0;
    uint64_t totalTilingBufferSize_ = 0;
    SVector<uint64_t> tilingBufferSizes_;
    uint64_t maxIntermediateBufferSize_ = 0;
    SVector<uint64_t> intermediateBufferSizes_;
    std::shared_ptr<MemAllocationSolver> memAllocationSolver_;
};
} // namespace atb
#endif