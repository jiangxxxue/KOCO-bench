/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/runner/lcoc_runner.h"
#include <atb/utils/log.h>

namespace atb {
LcocRunner::LcocRunner(const std::string &name, RunnerType runnerType, int32_t rank, int32_t rankSize,
                       const infer::CommMode commMode, Context &context, const std::string &commDomain)
    : LcalRunner(name, runnerType, rank, rankSize, commMode, commDomain, context)
{
    ATB_LOG(INFO) << GetLogPrefix() << "LcocRunner::LcocRunner called, rank : " << rank << "/" << rankSize;
    InitLcoc();
}

LcocRunner::~LcocRunner() {}

void LcocRunner::InitLcoc()
{
    Lcal::LcalComm *lcalComm = GetLcalComm();
    if (!lcalComm) {
        ATB_LOG(ERROR) << GetLogPrefix() << "GetLcalComm failed, rank: " << rank_;
        return;
    }
    lcoc_ = std::make_shared<Lcal::Lcoc>(*lcalComm);
    if (!lcoc_) {
        ATB_LOG(ERROR) << GetLogPrefix() << "failed to new Lcoc of rank: " << rank_;
        return;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "InitLcoc success, rank : " << rank_ << "/" << rankSize_;
}
} // namespace atb
