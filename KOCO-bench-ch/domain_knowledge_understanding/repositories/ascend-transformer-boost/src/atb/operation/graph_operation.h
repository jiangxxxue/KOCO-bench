/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_GRAPH_OPERATION_H
#define ATB_GRAPH_OPERATION_H
#include <string>
#include <functional>
#include <memory>
#include "atb/svector.h"
#include "operation_base.h"
#include "atb/runner/graph_runner.h"

namespace atb {
class GraphOperation : public OperationBase {
public:
    using ViewFunc = std::function<void(const SVector<int64_t> &oldDims, SVector<int64_t> &newDims)>;
    using InferShapePreFunc = std::function<void(SVector<Tensor> &inTensors, SVector<Tensor> &outTensors)>;

    explicit GraphOperation(const std::string &name);
    GraphOperation(const std::string &name, const GraphParam &opGraph);
    ~GraphOperation() override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;
    void SetExecuteStreamId(uint32_t streamId) override;

protected:
    Status InferShapeImpl(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const override;
    std::shared_ptr<Runner> CreateRunner(Context &context) const override;
    Status SetNodeOperationIds() override;
    void InitEmptyInTensorPerms();
    SVector<bool> GetEmptyInTensorPermissions() const override;
    void InitEmptyOutTensorPerms();
    SVector<bool> GetEmptyOutTensorPermissions() const override;
    void GetGraphInfoImpl(nlohmann::json &graphJson) const override;

protected:
    GraphParam opGraph_;

private:
    void UsePluginOperations();
    Status InferShapeImplDefault(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const;
    void BuildFullTensorPtrs(std::vector<Tensor *> &fullTensorPtrs, GraphRunner::Graph &runnerGraph) const;
    Status CreateRunnerNode(const size_t nodeId, GraphRunner::Graph &runnerGraph,
                            std::vector<int64_t> &nodeOperationIds, const std::vector<Tensor *> &fullTensorPtrs,
                            Context &context) const;
};
} // namespace atb
#endif
