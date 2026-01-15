/*
 *  Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TORCH_ATB_PROF_STATS_H
#define TORCH_ATB_PROF_STATS_H

#include <string>
#include <map>
#include <vector>

namespace TorchAtb {
constexpr size_t MAX_RUN_TIMES = 1000;

class ProfStats {
public:
    ProfStats();
    ~ProfStats();
    static ProfStats &GetProfStats();
    void SetRunTime(const std::string &opName, uint64_t runTime);
    std::vector<uint64_t> GetRunTimeStats(const std::string &opName);

private:
    std::map<std::string, std::vector<uint64_t>> runTimeStatsMap;
};
} // namespace TorchAtb
#endif