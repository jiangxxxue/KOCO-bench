/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_LINEAR_SPARSE_OPS_RUNNER_H
#define ATB_LINEAR_SPARSE_OPS_RUNNER_H
#include <asdops/params/params.h>
#include "atb/runner/ops_runner.h"
#include "atb/infer_op_params.h"

namespace atb {
class LinearSparseOpsRunner : public OpsRunner {
public:
    explicit LinearSparseOpsRunner(const infer::LinearSparseParam &param);
    ~LinearSparseOpsRunner() override;

protected:
    Status SetupKernelGraph(const OpsTensorPack &opsTensorPack) override;

private:
    void InitMemberVar();
    Status SetupKernelGraph310p();

private:
    infer::LinearSparseParam param_;                 // op参数
    bool needMergeAxis_ = false;                     // 判断x矩阵是否需要合轴
    AsdOps::OpParam::MatMul matmulParam_;            // 传给算子库的matmul参数
    AsdOps::OpParam::Transdata transdataNd2NzParam_; // 传给算子库的ND转NZ的transdata参数
    AsdOps::OpParam::Transdata transdataNz2NdParam_; // 传给算子库的NZ转ND的transdata参数
};
} // namespace atb
#endif // ATB_LINEAR_SPARSE_OPS_RUNNER_H
