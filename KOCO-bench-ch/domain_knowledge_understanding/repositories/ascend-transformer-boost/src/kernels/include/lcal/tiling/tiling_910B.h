/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LCAL_TILING_910B_H
#define LCAL_TILING_910B_H

#include "tiling_args.h"
namespace Lcal {
    void AllGatherGetDefaultTiling(CoCTilingData &cocTilingData);
    void AllGatherEightRankFP16GetDefaultTiling(CoCTilingData &cocTilingData);
    void AllGatherFourRankINT8Tiling(CoCTilingData &cocTilingData);

    void AllGatherV2EightRankFP16GetDefaultTiling(CoCTilingData &cocTilingData);
    void AllGatherV2EightRankFP16Core16GetDefaultTiling(CoCTilingData &cocTilingData);

    void AllReduceGetDefaultTiling(CoCTilingData &cocTilingData);
    void AllReduceFourRankInt8GetDefaultTiling(CoCTilingData &cocTilingData);
    void AllReduceFourRankFP16GetDefaultTiling(CoCTilingData &cocTilingData);
    void AllReduceEightRankFP16GetDefaultTiling(CoCTilingData &cocTilingData);
    void AllReduceEightRankINT8GetDefaultTiling(CoCTilingData &cocTilingData);
    void AllReduceTwoRankFP16Tiling(CoCTilingData &cocTilingData);

    void ReduceScatterEightRankFP16GetDefaultTiling(CoCTilingData &cocTilingData);
    void ReduceScatterFourRankINT8Tiling(CoCTilingData &cocTilingData);
}
#endif // LCAL_TILING_910B_H