/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_STATISTIC_H
#define ATB_STATISTIC_H
#include <string>

namespace atb {
struct OpSetupStatistic {
    uint64_t totalTime = 0;
    uint64_t runnerCreateTime = 0;
    uint64_t runnerSetupTime = 0;
    uint64_t runnerFillHostTilingTime = 0;
    uint64_t setupTotalCount = 0;
    uint64_t setupCacheHitCount = 0;
    uint64_t setupCacheMissCount = 0;
    uint64_t opsInitLuanchBufferTime = 0;
    uint64_t tilingCacheHitCount = 0;
    uint64_t tilingCacheMissCount = 0;
    uint64_t tilingLocalCacheHitCount = 0;
    uint64_t tilingGlobalCacheHitCount = 0;
    uint64_t kernelCacheGetTilingTime = 0;
    uint64_t kernelCacheAddTilingTime = 0;
    uint64_t kernelCacheCompareRunInfoTime = 0;
    uint64_t kernelCacheGetRunInfoTime = 0;

    std::string ToString() const;
    void Reset();
};

struct OpExecuteStatistic {
    uint64_t totalTime = 0;
    uint64_t streamSyncTime = 0;
    uint64_t tillingCopyTime = 0;
    uint64_t kernelExecuteTime = 0;
    uint64_t launchTime = 0;
    uint64_t preLaunchTime = 0;
    std::string ToString() const;
    void Reset();
};

OpSetupStatistic &GetOpSetupStatistic();
OpExecuteStatistic &GetOpExecuteStatistic();
} // namespace atb
#endif