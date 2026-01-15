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
void CoCMatmulReduceScatterTilingFunc::GetDefaultTiling(const TaskParam &taskParam)
{
    CoCTilingFunc::GetDefaultTiling(taskParam);
    if (Is91093(taskParam.chipName)) {
        if (cocTilingData.rankSize == RANKSIZE_EIGHT) {
            ReduceScatterNPU91093EightRankFP16Tiling(cocTilingData);
            return;
        } else if (cocTilingData.rankSize == RANKSIZE_SIXTEEN) {
            ReduceScatterNPU91093SixteenRankFP16Tiling(cocTilingData);
            return;
        } else if (cocTilingData.rankSize == RANKSIZE_TWO &&
                   taskParam.cocParamDesc.mmInfo.isInt8) {
            ReduceScatterNPU91093TwoRankINT8Tiling(cocTilingData);
            return;
        } else if (cocTilingData.rankSize == RANKSIZE_TWO) {
            ReduceScatterNPU91093TwoRankFP16Tiling(cocTilingData);
            return;
        } else if (cocTilingData.rankSize == RANKSIZE_FOUR) {
            ReduceScatterNPU91093FourRankFP16Tiling(cocTilingData);
            return;
        }
    } else if (Is910B(taskParam.chipName)) {
        if (cocTilingData.rankSize == RANKSIZE_FOUR) {
            ReduceScatterFourRankINT8Tiling(cocTilingData); // INT8
            return;
        }
    }
    ReduceScatterEightRankFP16GetDefaultTiling(cocTilingData);
}

bool CoCMatmulReduceScatterTilingFunc::CheckTiling(const TaskParam &taskParam)
{
    if (!CoCTilingFunc::CheckTiling(taskParam)) {
        return false;
    }
    auto pValue = cocTilingData.pValue;
    auto rankSize = cocTilingData.rankSize;
    auto blockDim = cocTilingData.blockDim;
    if ((pValue * blockDim) % rankSize != 0) {
        std::string str = "The product of pValue and blockDim must be divisible by rankSize."
                          " pValue: " + std::to_string(pValue) + " blockDim: " + std::to_string(blockDim) +
                          " rankSize: " + std::to_string(rankSize);
        PrintErrorLog(taskParam.lcalType, str);
        return false;
    }
    return true;
}
}