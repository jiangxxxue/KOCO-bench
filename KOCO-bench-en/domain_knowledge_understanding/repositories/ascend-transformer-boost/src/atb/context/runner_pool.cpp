/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "atb/context/runner_pool.h"
#include "atb/utils/config.h"
#include "atb/runner/runner.h"
#include "atb/utils/singleton.h"

static const uint32_t DEFAULT_RUNNER_POOL_SIZE = 64;

namespace atb {
RunnerPool::RunnerPool()
{
    lock_ = std::make_shared<std::mutex>();
    poolItems_.resize(DEFAULT_RUNNER_POOL_SIZE);
}

void RunnerPool::FreeRunner(Runner *runner)
{
    if (lock_ == nullptr) {
        ATB_LOG(ERROR) << "Lock is empty!";
        return;
    }
    std::lock_guard<std::mutex> lock(*lock_);
    for (auto &poolItem : poolItems_) {
        if (poolItem.runner.get() == runner) {
            poolItem.isUsed = false;
            break;
        }
    }
}

void RunnerPool::SetRunnerParam(PoolItem &item, const Mki::Any &param) const
{
    item.runner->SetParam(param);
}

RunnerPool::RunnerPool(const RunnerPool &other)
{
    poolItems_ = other.poolItems_;
    lock_ = std::make_shared<std::mutex>();
}

RunnerPool &RunnerPool::operator=(const RunnerPool &other)
{
    poolItems_ = other.poolItems_;
    lock_ = std::make_shared<std::mutex>();
    return *this;
}
} // namespace atb