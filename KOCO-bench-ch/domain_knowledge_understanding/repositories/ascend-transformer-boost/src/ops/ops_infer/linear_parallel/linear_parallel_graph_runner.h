/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_LINEAR_PARALLEL_GRAPH_RUNNER_H
#define ATB_LINEAR_PARALLEL_GRAPH_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/operation/graph_operation.h"

namespace atb {
class LinearParallelGraphRunner : public GraphRunner {
public:
    explicit LinearParallelGraphRunner(const infer::LinearParallelParam &param, ContextBase &context);
    ~LinearParallelGraphRunner() override;

private:
    void InitGraph(ContextBase &context);
    void InitLinearNodeInTensorReshapeFuncs(GraphRunner::Node &linearNode) const;
    bool InitLinearNode(infer::LinearParam &linearParam, GraphRunner::Node &linearNode,
                        std::vector<int64_t> &nodeOperationIds, ContextBase &context) const;
    bool InitAllReduceNode(infer::AllReduceParam &allReduceParam, GraphRunner::Node &allReduceNode,
                           std::vector<int64_t> &nodeOperationIds, ContextBase &context) const;
    bool InitAddNode(GraphRunner::Node &addNode, std::vector<int64_t> &nodeOperationIds, ContextBase &context) const;

private:
    infer::LinearParallelParam param_;
};
} // namespace atb
#endif