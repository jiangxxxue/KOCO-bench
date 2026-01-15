/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_LCCL_RUNNER_H
#define ATB_LCCL_RUNNER_H

#include <lcal_api.h>
#include <memory>
#include "atb/runner/runner.h"
#include "atb/runner/lcal_runner.h"
#include "atb/runner/runner_type.h"
#include "atb/infer_op_params.h"

namespace atb {
class LcclRunner : public LcalRunner {
public:
    explicit LcclRunner(const std::string &name, RunnerType runnerType, int32_t rank, int32_t rankSize,
                        const infer::CommMode commMode, Context &context, const std::string &commDomain = "");
    ~LcclRunner() override;

protected:
    std::shared_ptr<Lcal::Lccl> lccl_;

private:
    void Initlccl();
};
} // namespace atb
#endif
