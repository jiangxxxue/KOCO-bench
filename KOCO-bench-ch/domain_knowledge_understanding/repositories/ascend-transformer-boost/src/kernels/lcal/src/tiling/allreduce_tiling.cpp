/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "tiling.h"
#include "lcoc_func.h"
#include "tiling_910B.h"
#include "tiling_91093.h"
#include "tiling_func.h"

namespace Lcal {
const int ALLREDUCE_M_EDGE = 3072;
const int ALLREDUCE_N_EDGE = 3072;

void CoCMatmulAllReduceTilingFunc::GetDefaultTiling(const TaskParam &taskParam)
{
    CoCTilingFunc::GetDefaultTiling(taskParam);
    if (Is91093(taskParam.chipName)) {
        if (cocTilingData.rankSize == RANKSIZE_EIGHT) {
            AllReduceNPU91093EightRankFP16Tiling(cocTilingData);
            return;
        } else if (cocTilingData.rankSize == RANKSIZE_SIXTEEN) {
            AllReduceNPU91093SixteenRankFP16Tiling(cocTilingData);
            return;
        }
    }
    if (cocTilingData.rankSize == RANKSIZE_FOUR) {
        if (taskParam.cocParamDesc.mmInfo.isInt8) {
            AllReduceFourRankInt8GetDefaultTiling(cocTilingData);
            return;
        } else {
            AllReduceFourRankFP16GetDefaultTiling(cocTilingData);
            return;
        }
    } else if (cocTilingData.rankSize == RANKSIZE_TWO) {
        AllReduceTwoRankFP16Tiling(cocTilingData);
        return;
    }
    AllReduceGetDefaultTiling(cocTilingData);
}

void CoCMatmulAllReduceDeterTilingFunc::GetDefaultTiling(const TaskParam &taskParam)
{
    CoCTilingFunc::GetDefaultTiling(taskParam);
    if (cocTilingData.rankSize == RANKSIZE_FOUR) {
        if (taskParam.cocParamDesc.mmInfo.isInt8) {
            AllReduceFourRankInt8GetDefaultTiling(cocTilingData);
        } else {
            AllReduceFourRankFP16GetDefaultTiling(cocTilingData);
        }
    } else {
        if (taskParam.cocParamDesc.mmInfo.isInt8) {
            AllReduceEightRankINT8GetDefaultTiling(cocTilingData);
        } else {
            AllReduceEightRankFP16GetDefaultTiling(cocTilingData);
        }
    }
    if (cocTilingData.m * cocTilingData.n >= ALLREDUCE_M_EDGE * ALLREDUCE_N_EDGE) {
        cocTilingData.lenPerLoop = ALLREDUCE_LENPERLOOP_DEFAULT / RANKSIZE_EIGHT * cocTilingData.rankSize;
        cocTilingData.lenPerLoop = RoundNum(cocTilingData.lenPerLoop, HALF_KBYTE);
        cocTilingData.ubMoveNum = cocTilingData.lenPerLoop;
        cocTilingData.extraLenPerLoop = cocTilingData.lenPerLoop;
        cocTilingData.extraUbMoveNum = cocTilingData.ubMoveNum;
    }
    if (cocTilingData.lenPerLoop > TREE_LEN_PER_LOOP) {
        cocTilingData.lenPerLoop = TREE_LEN_PER_LOOP;
        cocTilingData.ubMoveNum = TREE_LEN_PER_LOOP;
        cocTilingData.extraLenPerLoop = cocTilingData.lenPerLoop;
        cocTilingData.extraUbMoveNum = cocTilingData.ubMoveNum;
    }
}

bool CheckCMatrix(const TaskParam &taskParam, const CoCTilingData &data)
{
    constexpr int32_t BUFFER_UNIT = 1024;
    if (data.withSerialMode != 0 &&
        data.batchSize * data.m * data.n >= (taskParam.bufferSize * BUFFER_UNIT * BUFFER_UNIT)
        / INPUT_DTYPE / MAX_BLOCK_COUNT) {
        std::string str = "The matrix c is too large to support serial. "
                          "withSerialMode: " + std::to_string(data.withSerialMode) +
                          ", batchSize: " + std::to_string(data.batchSize) +
                          ", m: " + std::to_string(data.m) +
                          ", n: " + std::to_string(data.n);
        PrintErrorLog(taskParam.lcalType, str);
        return false;
    }
    return true;
}

bool CoCMatmulAllReduceTilingFunc::CheckTiling(const TaskParam &taskParam)
{
    if (!CoCTilingFunc::CheckTiling(taskParam)) {
        return false;
    }
    if (!CheckCMatrix(taskParam, cocTilingData)) {
        return false;
    }

    auto rankSize = cocTilingData.rankSize;
    auto commNpuSplit = cocTilingData.commNpuSplit;
    auto commDataSplit = cocTilingData.commDataSplit;
    auto coreNum = cocTilingData.blockDim;
    int32_t useCoreCount = commNpuSplit * commDataSplit;

    std::vector<std::tuple<std::string, int, int, int>> paramCheckList = {
        {"commNpuSplit * commDataSplit", useCoreCount, rankSize, coreNum},
        {"commNpuSplit", commNpuSplit, PARAM_CHECK_MIN_VALUE_ONE, rankSize}
    };
    return CheckParamScopeList(paramCheckList);
}

bool CoCMatmulAllReduceDeterTilingFunc::CheckTiling(const TaskParam &taskParam)
{
    if (!CoCMatmulAllReduceTilingFunc::CheckTiling(taskParam)) {
        return false;
    }

    auto commNpuSplit = cocTilingData.commNpuSplit;
    if (commNpuSplit != 1) {
        std::string str = "The product of commNpuSplit must equal 1. commNpuSplit: " + std::to_string(commNpuSplit);
        PrintErrorLog(taskParam.lcalType, str);
        return false;
    }
    return true;
}
}