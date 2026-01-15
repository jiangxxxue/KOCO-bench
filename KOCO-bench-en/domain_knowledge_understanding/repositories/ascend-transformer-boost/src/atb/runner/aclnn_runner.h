/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_ACLNN_RUNNER_H
#define ATB_ACLNN_RUNNER_H
#include "runner_type.h"
#include "runner.h"

namespace atb {
class AclnnRunner : public Runner {
public:
    explicit AclnnRunner(const std::string &name, RunnerType runnerType = RUNNER_TYPE_UNDEFINED);
    ~AclnnRunner() override;
protected:
    Status SetupImpl(RunnerVariantPack &runnerVariantPack) override;
    virtual Status BuildAclnnVariantPack(const RunnerVariantPack &runnerVariantPack) = 0;
    virtual aclnnStatus SetAclNNWorkspaceExecutor() = 0;

    Status PreExecuteImpl(RunnerVariantPack &runnerVariantPack) override;
    Status ExecuteImpl(RunnerVariantPack &runnerVariantPack) override;
    virtual Status LaunchAclnnKernel() = 0;
    uint64_t GetWorkspaceBufferSizeImpl() override;
    void UpdateWorkspace(const RunnerVariantPack &runnerVariantPack);
    RunnerType runnerType_ = RUNNER_TYPE_UNDEFINED;
    bool executorRepeatable_ = false;
    std::shared_ptr<aclOpExecutor> aclnnExecutor_ = nullptr;
    AclNNVariantPack aclnnVariantPack_;
    RunnerVariantPack atbVariantPack_;
};
} // namespace atb
#endif
