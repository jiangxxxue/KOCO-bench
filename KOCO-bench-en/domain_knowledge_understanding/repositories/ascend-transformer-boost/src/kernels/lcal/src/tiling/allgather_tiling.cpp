/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include "tiling.h"
#include "tiling_910B.h"
#include "tiling_91093.h"
#include "tiling_func.h"
#include "lcoc_func.h"

namespace Lcal {
void CoCAllGatherMatmulTilingFunc::GetDefaultTiling(const TaskParam &taskParam)
{
    CoCTilingFunc::GetDefaultTiling(taskParam);
    if (Is91093(taskParam.chipName)) {
        if (cocTilingData.rankSize == RANKSIZE_EIGHT) {
            AllGatherNPU91093EightRankFP16Tiling(cocTilingData);
            return;
        } else if (cocTilingData.rankSize == RANKSIZE_SIXTEEN) {
            AllGatherNPU91093SixteenRankFP16Tiling(cocTilingData);
            return;
        } else if (cocTilingData.rankSize == RANKSIZE_TWO &&
                   taskParam.cocParamDesc.mmInfo.isInt8) {
            AllGatherNPU91093TwoRankINT8Tiling(cocTilingData);
            return;
        } else if (cocTilingData.rankSize == RANKSIZE_TWO) {
            AllGatherNPU91093TwoRankFP16Tiling(cocTilingData);
            return;
        }
    } else if (Is910B(taskParam.chipName)) {
        if (cocTilingData.rankSize == RANKSIZE_EIGHT) {
            AllGatherEightRankFP16GetDefaultTiling(cocTilingData);
            return;
        } else if (cocTilingData.rankSize == RANKSIZE_FOUR) {
            AllGatherFourRankINT8Tiling(cocTilingData); // INT8
            return;
        }
    }
    AllGatherGetDefaultTiling(cocTilingData);
}

void CoCAllGatherMatmulV2TilingFunc::GetDefaultTiling(const TaskParam &taskParam)
{
    CoCTilingFunc::GetDefaultTiling(taskParam);
    auto coreNum = cocTilingData.blockDim;
    if (Is91093(taskParam.chipName)) {
        if (cocTilingData.rankSize == RANKSIZE_EIGHT) {
            AllGatherV2NPU91093EightRankFP16Tiling(cocTilingData);
            return;
        } else if (cocTilingData.rankSize == RANKSIZE_SIXTEEN) {
            AllGatherV2NPU91093SixteenRankFP16Tiling(cocTilingData);
            return;
        } else if (cocTilingData.rankSize == RANKSIZE_TWO) {
            AllGatherV2NPU91093TwoRankFP16Tiling(cocTilingData);
            return;
        }
    }
    if (coreNum >= ALLGATHERV2_CORENUM_SIXTEEN + cocTilingData.rankSize) {
        AllGatherV2EightRankFP16Core16GetDefaultTiling(cocTilingData);
        return;
    }
    AllGatherV2EightRankFP16GetDefaultTiling(cocTilingData);
}

bool CheckKValue(const TaskParam &taskParam, const CoCTilingData &data)
{
    auto blockCount = data.is91093 ? BLOCK_COUNT_3 : MAX_BLOCK_COUNT;
    int32_t maxPeerMemPerRank = (taskParam.bufferSize * 1024 * 1024) / INPUT_DTYPE / data.rankSize / blockCount;
    if (data.pValue * data.m0 * data.k0 * data.kLoop >= maxPeerMemPerRank) {
        std::string str = "The k value is too large and is currently not supported. "
                          "pValue: " + std::to_string(data.pValue) + ", m0: " + std::to_string(data.m0) +
                          ", k0: " + std::to_string(data.k0) + ", kLoop: " + std::to_string(data.kLoop) +
                          "maxPeerMemPerRank: " + std::to_string(maxPeerMemPerRank);
        PrintErrorLog(taskParam.lcalType, str);
        return false;
    }
    return true;
}

bool CoCAllGatherMatmulTilingFunc::CheckTiling(const TaskParam &taskParam)
{
    if (!CoCTilingFunc::CheckTiling(taskParam)) {
        return false;
    }
    if (!CheckKValue(taskParam, cocTilingData)) {
        return false;
    }

    auto rankSize = cocTilingData.rankSize;
    auto commNpuSplit = cocTilingData.commNpuSplit;
    auto commDataSplit = cocTilingData.commDataSplit;
    auto coreNum = cocTilingData.blockDim;
    auto is91093 = cocTilingData.is91093;
    auto minCoreCount = is91093 ? rankSize / A3_DIE_NUM : rankSize;
    int32_t useCoreCount = commNpuSplit * commDataSplit;

    std::vector<std::tuple<std::string, int, int, int>> paramCheckList = {
        {"commNpuSplit * commDataSplit", useCoreCount, minCoreCount, coreNum}
    };
    return CheckParamScopeList(paramCheckList);
}

bool CoCAllGatherMatmulV2TilingFunc::CheckTiling(const TaskParam &taskParam)
{
    if (!CoCTilingFunc::CheckTiling(taskParam)) {
        return false;
    }
    if (!CheckKValue(taskParam, cocTilingData)) {
        return false;
    }

    auto commNpuSplit = cocTilingData.commNpuSplit;
    auto commDataSplit = cocTilingData.commDataSplit;
    auto coreNum = cocTilingData.blockDim;
    int32_t useCoreCount = commNpuSplit * commDataSplit;

    std::vector<std::tuple<std::string, int, int, int>> paramCheckList = {
        {"commNpuSplit * commDataSplit", useCoreCount, PARAM_CHECK_MIN_VALUE_ONE, coreNum-1}
    };
    return CheckParamScopeList(paramCheckList);
}
}