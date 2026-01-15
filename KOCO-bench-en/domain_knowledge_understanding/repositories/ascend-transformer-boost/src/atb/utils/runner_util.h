/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RUNNER_UTIL_H
#define RUNNER_UTIL_H
#include <mki/op_desc.h>
#include <asdops/params/params.h>
#include <mki/utils/SVector/SVector.h>
#include "atb/infer_op_params.h"
#include "atb/runner/ops_runner.h"
#include "atb/operation/graph_operation.h"

namespace atb {
constexpr int64_t DEFAULT_ALIGN = 16;

class RunnerUtil {
public:
    static Mki::OpDesc GetActivationNodeOpDesc(infer::ActivationParam activationParam);
    static void TransdataSqueeze(KernelGraphNode &transdataNode, bool needMergeAxis = false);
    static int64_t AlignUp(int64_t dim, int64_t align);
    static void ConfigViewFuncs(KernelGraphNode &matmulNode, bool needMergeAxis, bool needReshape = false,
                                int64_t align = DEFAULT_ALIGN);
    static bool InitGraphRunnerNode(GraphRunner::Node &node, Operation *operation,
                                    const std::vector<int64_t> &operationIds, ContextBase &context);
};
} // namespace atb
#endif