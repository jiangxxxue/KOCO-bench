/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_EVENT_RUNNER_H
#define ATB_EVENT_RUNNER_H

#include "atb/runner/runner.h"
#include "atb/common_op_params.h"

namespace atb {
class EventRunner : public Runner {
public:
    explicit EventRunner(const common::EventParam &param);
    ~EventRunner() override;
    void SetParam(const Mki::Any &param) override;

protected:
    Status SetupImpl(RunnerVariantPack &runnerVariantPack) override;
    Status ExecuteImpl(RunnerVariantPack &runnerVariantPack) override;

protected:
    RunnerType runnerType_ = RUNNER_TYPE_UNDEFINED;
    
private:
    common::EventParam param_;
};
} // namespace atb
#endif // ATB_EVENT_RUNNER_H