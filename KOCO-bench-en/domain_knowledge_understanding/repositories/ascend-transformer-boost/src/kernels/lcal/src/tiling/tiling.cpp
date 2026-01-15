/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "tiling_func.h"
#include "mki/utils/log/log.h"
#include "tiling.h"
namespace Lcal {
CoCTilingData CoCTilingFunc::GenerateTiling(const TaskParam &taskParam, const CoCTiling &tiling)
{
    SetTilingInputParam(taskParam, cocTilingData);

    cocTilingData.SetDefaultValue();

    this->GetDefaultTiling(taskParam);

    // 设置Tiling策略参数
    SetTilingData(taskParam, tiling, cocTilingData);

    return cocTilingData;
}

bool CoCTilingFunc::CheckTiling(const TaskParam &taskParam)
{
    (void) taskParam;
    return CheckCoCTilingData(cocTilingData);
}

void CoCTilingFunc::GetDefaultTiling(const TaskParam &taskParam)
{
    (void) taskParam;
    cocTilingData.ubMoveNum = VALID_UB_MOVE_NUM;
    cocTilingData.commNpuSplit = cocTilingData.rankSize;
    cocTilingData.commDataSplit = COMMDATASPLIT_ONE;
    cocTilingData.commDirect = COMM_DATA_DIRECT;
    cocTilingData.lenPerLoop = LENPERLOOP_DEFAULT;
}
}