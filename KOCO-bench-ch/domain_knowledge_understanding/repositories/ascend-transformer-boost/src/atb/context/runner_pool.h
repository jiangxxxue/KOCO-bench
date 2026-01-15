/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_RUNNER_POOL_H
#define ATB_RUNNER_POOL_H

#include <memory>
#include <vector>
#include <mutex>
#include <mki/utils/any/any.h>
#include "atb/utils/log.h"

namespace atb {
class Runner;
struct PoolItem {
    bool isUsed = false;
    std::shared_ptr<Runner> runner;
};

class RunnerPool {
public:
    RunnerPool();
    ~RunnerPool() = default;
    RunnerPool(const RunnerPool &other);
    RunnerPool &operator=(const RunnerPool &other);
    template <typename RunnerClass, typename ParamType> Runner *MallocRunner(ParamType param)
    {
        if (lock_ == nullptr) {
            ATB_LOG(ERROR) << "Lock is empty!";
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(*lock_);
        for (size_t i = 0; i < poolItems_.size(); i++) {
            auto &poolItem = poolItems_.at(i);
            if (!poolItem.isUsed) {
                poolItem.isUsed = true;
                if (poolItem.runner) {
                    ATB_LOG(INFO) << "Get pool old runner!";
                    SetRunnerParam(poolItem, param);
                } else {
                    ATB_LOG(INFO) << "Pool create new runner!";
                    poolItem.runner = std::make_shared<RunnerClass>(param);
                }
                return poolItem.runner.get();
            }
        }
        return nullptr;
    }
    void FreeRunner(Runner *runner);

private:
    void SetRunnerParam(PoolItem &item, const Mki::Any &param) const;

private:
    std::vector<PoolItem> poolItems_;
    std::shared_ptr<std::mutex> lock_;
};
} // namespace atb
#endif