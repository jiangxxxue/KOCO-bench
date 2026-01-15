/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_LINEAR_PARALLEL_LCOC_RUNNER_H
#define ATB_LINEAR_PARALLEL_LCOC_RUNNER_H
#include "atb/infer_op_params.h"
#include "atb/runner/lcoc_runner.h"

namespace atb {
class LinearParallelLcocRunner : public LcocRunner {
public:
    explicit LinearParallelLcocRunner(const infer::LinearParallelParam &param, Context &context);
    ~LinearParallelLcocRunner() override;

protected:
    Status SetupImpl(RunnerVariantPack &runnerVariantPack) override;
    Status ExecuteImpl(RunnerVariantPack &runnerVariantPack) override;
    uint64_t GetWorkspaceBufferSizeImpl() override;
    Status LaunchKernel(Lcal::CoCInputPkg inputPkg, Lcal::CoCOutputPkg outputPkg,
                           RunnerVariantPack &runnerVariantPack, infer::LinearParallelParam::ParallelType type);

private:
    infer::LinearParallelParam param_;
    Lcal::LcalType lcalType_ = Lcal::LcalType::MATMUL_ALL_REDUCE;
    bool isQuant_ = false;
};
} // namespace atb
#endif