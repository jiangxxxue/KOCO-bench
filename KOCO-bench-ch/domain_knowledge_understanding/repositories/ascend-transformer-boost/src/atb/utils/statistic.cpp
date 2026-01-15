/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/utils/statistic.h"

namespace atb {
thread_local OpSetupStatistic g_opSetupStatistic;
thread_local OpExecuteStatistic g_opExecuteStatistic;

std::string OpSetupStatistic::ToString() const
{
    return "totalTime:" + std::to_string(totalTime) + ", runnerCreateTime:" + std::to_string(runnerCreateTime) +
           ", runnerSetupTime:" + std::to_string(runnerSetupTime) +
           ", runnerFillHostTilingTime:" + std::to_string(runnerFillHostTilingTime) +
           ", setupTotalCount:" + std::to_string(setupTotalCount) +
           ", setupCacheHitCount:" + std::to_string(setupCacheHitCount) +
           ", setupCacheMissCount:" + std::to_string(setupCacheMissCount) +
           ", opsInitLuanchBufferTime:" + std::to_string(opsInitLuanchBufferTime) +
           ", tilingCacheHitCount:" + std::to_string(tilingCacheHitCount) +
           ", tilingCacheMissCount:" + std::to_string(tilingCacheMissCount) +
           ", tilingLocalCacheHitCount:" + std::to_string(tilingLocalCacheHitCount) +
           ", tilingGlobalCacheHitCount:" + std::to_string(tilingGlobalCacheHitCount) +
           ", kernelCacheGetTilingTime:" + std::to_string(kernelCacheGetTilingTime) +
           ", kernelCacheAddTilingTime:" + std::to_string(kernelCacheAddTilingTime) +
           ", kernelCacheCompareRunInfoTime:" + std::to_string(kernelCacheCompareRunInfoTime) +
           ", kernelCacheGetRunInfoTime:" + std::to_string(kernelCacheGetRunInfoTime);
}

void OpSetupStatistic::Reset()
{
    totalTime = 0;
    runnerCreateTime = 0;
    runnerSetupTime = 0;
    runnerFillHostTilingTime = 0;
    setupTotalCount = 0;
    setupCacheHitCount = 0;
    setupCacheMissCount = 0;
    tilingLocalCacheHitCount = 0;
    tilingGlobalCacheHitCount = 0;
    opsInitLuanchBufferTime = 0;
    tilingCacheHitCount = 0;
    tilingCacheMissCount = 0;
    kernelCacheGetTilingTime = 0;
    kernelCacheAddTilingTime = 0;
    kernelCacheCompareRunInfoTime = 0;
    kernelCacheGetRunInfoTime = 0;
}


std::string OpExecuteStatistic::ToString() const
{
    return "totalTime:" + std::to_string(totalTime) + ", streamSyncTime:" + std::to_string(streamSyncTime) +
           ", tillingCopyTime:" + std::to_string(tillingCopyTime) +
           ", kernelExecuteTime:" + std::to_string(kernelExecuteTime) +
           ", launchTime:" + std::to_string(launchTime) +
           ", preLaunchTime:" + std::to_string(preLaunchTime);
}

void OpExecuteStatistic::Reset()
{
    totalTime = 0;
    streamSyncTime = 0;
    tillingCopyTime = 0;
    kernelExecuteTime = 0;
    launchTime = 0;
    preLaunchTime = 0;
}

OpSetupStatistic &GetOpSetupStatistic()
{
    return g_opSetupStatistic;
}

OpExecuteStatistic &GetOpExecuteStatistic()
{
    return g_opExecuteStatistic;
}
} // namespace atb