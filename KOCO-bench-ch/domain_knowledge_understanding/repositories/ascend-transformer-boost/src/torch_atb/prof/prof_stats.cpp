/*
 *  Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "prof_stats.h"
#include <thread>

namespace TorchAtb {

ProfStats::ProfStats() {}

ProfStats::~ProfStats() {}

ProfStats &ProfStats::GetProfStats()
{
    thread_local ProfStats instance;
    return instance;
}

void ProfStats::SetRunTime(const std::string &opName, uint64_t runTime)
{
    auto &runTimes = runTimeStatsMap[opName];
    if (runTimes.size() >= MAX_RUN_TIMES) {
        runTimes.erase(runTimes.begin());
    }
    runTimes.push_back(runTime);
}

std::vector<uint64_t> ProfStats::GetRunTimeStats(const std::string &opName)
{
    if (runTimeStatsMap.find(opName) != runTimeStatsMap.end()) {
        return runTimeStatsMap.at(opName);
    } else {
        return {};
    }
}

} // namespace TorchAtb