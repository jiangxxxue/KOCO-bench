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
#
namespace Lcal {
void CoCAllToAllAllGatherMatmulTilingFunc::GetDefaultTiling(const TaskParam &tilingInfo)
{
    CoCTilingFunc::GetDefaultTiling(tilingInfo);
    cocTilingData.m0 = DEFAULT_ROW;
    cocTilingData.n0 = DEFAULT_COL;
    cocTilingData.k0 = DEFAULT_COL;
    constexpr int32_t pValue = 1;
    cocTilingData.pValue = pValue;
    constexpr int32_t ubMove = 28672;
    cocTilingData.ubMoveNum = ubMove;
    return;
}

bool CheckPValue(const TaskParam &tilingInfo, const CoCTilingData &data)
{
    auto blockCount = MAX_BLOCK_COUNT;
    int32_t bufferSize = tilingInfo.bufferSize * 1024 * 1024;
    int32_t maxPeerMemPerRank = bufferSize / INPUT_DTYPE / data.rankSize / blockCount;
    if (data.pValue * data.m0 * data.k0 * data.kLoop >= maxPeerMemPerRank) {
        std::string str = "The k value is too large and is currently not supported. "
                          "pValue: " + std::to_string(data.pValue) + ", m0: " + std::to_string(data.m0) +
                          ", k0: " + std::to_string(data.k0) + ", kLoop: " + std::to_string(data.kLoop) +
                          "maxPeerMemPerRank: " + std::to_string(maxPeerMemPerRank);
        PrintErrorLog(tilingInfo.lcalType, str);
        return false;
    }
    return true;
}

bool CoCAllToAllAllGatherMatmulTilingFunc::CheckTiling(const TaskParam &tilingInfo)
{
    if (!CoCTilingFunc::CheckTiling(tilingInfo)) {
        return false;
    }
    if (!CheckPValue(tilingInfo, cocTilingData)) {
        return false;
    }

    int32_t rankSize = cocTilingData.rankSize;
    int32_t ep = tilingInfo.cocParamDesc.moeInfo.EP;
    int32_t tp = tilingInfo.cocParamDesc.moeInfo.TP;
    int32_t expertPerRank = tilingInfo.cocParamDesc.moeInfo.local_expert_nums;

    if (ep * tp != rankSize) {
        std::string str = "The ep * tp != rankSize. "
                            "rankSize: " + std::to_string(rankSize) + ", ep: " + std::to_string(ep) +
                            " , tp: " + std::to_string(tp);
        PrintErrorLog(tilingInfo.lcalType, str);
        return false;
    }

    std::vector<std::tuple<std::string, int, int, int>> paramCheckList = {
        {"expertPerrank", expertPerRank, 1, 20}
    };
    return CheckParamScopeList(paramCheckList);
}
}