/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PLUGIN_PLUGINRUNNER_H
#define PLUGIN_PLUGINRUNNER_H
#include "atb/runner/runner.h"
#include "atb/operation.h"

namespace atb {
class PluginRunner : public Runner {
public:
    explicit PluginRunner(Operation *operation);
    ~PluginRunner() override;

private:
    Status SetupImpl(RunnerVariantPack &runnerVariantPack) override;
    uint64_t GetWorkspaceBufferSizeImpl() override;
    Status ExecuteImpl(RunnerVariantPack &runnerVariantPack) override;

private:
    Operation *operation_ = nullptr;
    uint64_t workspaceSize_ = 0;
    VariantPack variantPack_;
};
} // namespace atb
#endif