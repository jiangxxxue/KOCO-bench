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
void CoCMatmulReduceScatterAllToAllHiddenTilingFunc::GetDefaultTiling(const TaskParam &tilingInfo)
{
    CoCTilingFunc::GetDefaultTiling(tilingInfo);
    auto n = tilingInfo.cocParamDesc.mmInfo.n;
    auto maxOutputSize = tilingInfo.cocParamDesc.moeInfo.maxOutputSize;
    int32_t maxPvalue = (n + 255) / 256;

    cocTilingData.m0 = DEFAULT_ROW;
    cocTilingData.n0 = DEFAULT_COL;
    cocTilingData.k0 = DEFAULT_COL;
    int32_t m = tilingInfo.cocParamDesc.mmInfo.m;
    auto blockCount = MAX_BLOCK_COUNT;
    int32_t bufferSize = tilingInfo.bufferSize * 1024 * 1024;
    int32_t maxPeerMemPerRank = bufferSize / INPUT_DTYPE / blockCount;
    constexpr int32_t Four = 4;
    cocTilingData.pValue = Four;
    if (cocTilingData.pValue > maxPvalue) {
        cocTilingData.pValue = maxPvalue;
    }

    if (m < DEFAULT_ROW) {
        cocTilingData.pValue = (n + cocTilingData.n0 - 1) / cocTilingData.n0;
    }

    if (cocTilingData.pValue * cocTilingData.n0 * maxOutputSize > maxPeerMemPerRank) {
        cocTilingData.pValue = maxPeerMemPerRank / maxOutputSize / cocTilingData.n0;
    }

    cocTilingData.ubMoveNum = AllTOAll_HIDDEN_UBMOVENUM;
    constexpr int32_t two = 2;
    int32_t maxUbPingPongSize = cocTilingData.ubMoveNum / two;
    if (cocTilingData.pValue * cocTilingData.n0 > maxUbPingPongSize) {
        cocTilingData.pValue = maxUbPingPongSize / cocTilingData.n0;
    }
    return;
}
bool CoCMatmulReduceScatterAllToAllHiddenTilingFunc::CheckTiling(const TaskParam &tilingInfo)
{
    int32_t rankSize = cocTilingData.rankSize;
    int32_t ep = tilingInfo.cocParamDesc.moeInfo.EP;
    int32_t tp = tilingInfo.cocParamDesc.moeInfo.TP;
    int32_t expertPerRank = tilingInfo.cocParamDesc.moeInfo.local_expert_nums;
    int32_t n = tilingInfo.cocParamDesc.mmInfo.n;
    auto maxOutputSize = tilingInfo.cocParamDesc.moeInfo.maxOutputSize;

    auto blockCount = MAX_BLOCK_COUNT;
    int32_t maxPeerMemPerRank = (tilingInfo.bufferSize * 1024 * 1024) / INPUT_DTYPE / blockCount;
    if ((cocTilingData.pValue - 1) * cocTilingData.n0 > n) {
        return false;
    }
    if (cocTilingData.pValue * cocTilingData.n0 * maxOutputSize > maxPeerMemPerRank) {
        std::string str = "The k value is too large and is currently not supported. "
                          "pValue: " + std::to_string(cocTilingData.pValue) + ", n0: " +
                          std::to_string(cocTilingData.n0) + "maxPeerMemPerRank: " + std::to_string(maxPeerMemPerRank);
        PrintErrorLog(tilingInfo.lcalType, str);
        return false;
    }
    constexpr int32_t Two = 2;
    int32_t maxUbPingPongSize = cocTilingData.ubMoveNum / Two;
    if (cocTilingData.pValue * cocTilingData.n0 > maxUbPingPongSize) {
        std::string str = "The k value is too large and is currently not supported. "
                          "pValue: " + std::to_string(cocTilingData.pValue) + ", n0: " +
                          std::to_string(cocTilingData.n0) + "maxUbPingPongSize: " + std::to_string(maxUbPingPongSize);
        PrintErrorLog(tilingInfo.lcalType, str);
        return false;
    }

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
