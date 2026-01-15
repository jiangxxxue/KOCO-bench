/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_LCAL_RUNNER_H
#define ATB_LCAL_RUNNER_H

#include <lcal.h>
#include <memory>
#include <utility>
#include "atb/runner/runner.h"
#include "atb/runner/runner_type.h"
#include "atb/infer_op_params.h"

namespace atb {
class LcalRunner : public Runner {
public:
    explicit LcalRunner(const std::string &name, RunnerType runnerType, int32_t rank, int32_t rankSize,
                        const infer::CommMode commMode, const std::string &commDomain, Context &context);
    ~LcalRunner() override;

protected:
    Lcal::LcalComm *GetLcalComm();
    Status SetupImpl(RunnerVariantPack &runnerVariantPack) override;

protected:
    RunnerType runnerType_ = RUNNER_TYPE_UNDEFINED;
    int32_t rank_ = 0;
    int32_t rankSize_ = 0;
    infer::CommMode commMode_ = infer::CommMode::COMM_MULTI_PROCESS;
    std::string commDomain_;

private:
    void InitLcalComm();
    std::pair<int32_t, int32_t> ParseCommDomain(const std::string &commDomain) const;
    std::shared_ptr<Lcal::LcalComm> CreateLcalComm();

private:
    std::shared_ptr<Lcal::LcalComm> lcalComm_;
    int32_t lcalErrorCode_ = 0;
    bool magicNumberDisabled_ = false;
};
} // namespace atb
#endif
