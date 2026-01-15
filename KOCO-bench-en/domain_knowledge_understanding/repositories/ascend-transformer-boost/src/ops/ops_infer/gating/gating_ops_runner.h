/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_OPS_GATING_RUNNER_H
#define ATB_OPS_GATING_RUNNER_H
#include <atbops/params/params.h>
#include "atb/runner/ops_runner.h"
#include "atb/infer_op_params.h"

namespace atb {
class GatingOpsRunner : public OpsRunner {
public:
    explicit GatingOpsRunner(const infer::GatingParam &param);
    ~GatingOpsRunner() override;

protected:
    Status SetupKernelGraph(const OpsTensorPack &opsTensorPack) override;

private:
    Status SetupKernelGraphGating();
    Status SetupKernelGraphGatingExpertParallelism();

private:
    infer::GatingParam param_;
    AtbOps::OpParam::Gating gatingParam_;
    Mki::Tensor nullTensor_ = {};
};
} // namespace atb
#endif // ATB_GATING_OPS_RUNNER_H