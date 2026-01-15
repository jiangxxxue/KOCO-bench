/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LCAL_TILING_91093_H
#define LCAL_TILING_91093_H

#include "tiling_args.h"
namespace Lcal {
    void AllGatherNPU91093EightRankFP16Tiling(CoCTilingData &cocTilingData);
    void AllGatherNPU91093SixteenRankFP16Tiling(CoCTilingData &cocTilingData);
    void AllGatherNPU91093TwoRankFP16Tiling(CoCTilingData &cocTilingData);
    void AllGatherNPU91093TwoRankINT8Tiling(CoCTilingData &cocTilingData);

    void AllGatherV2NPU91093EightRankFP16Tiling(CoCTilingData &cocTilingData);
    void AllGatherV2NPU91093SixteenRankFP16Tiling(CoCTilingData &cocTilingData);
    void AllGatherV2NPU91093TwoRankFP16Tiling(CoCTilingData &cocTilingData);

    void AllReduceNPU91093EightRankFP16Tiling(CoCTilingData &cocTilingData);
    void AllReduceNPU91093SixteenRankFP16Tiling(CoCTilingData &cocTilingData);

    void ReduceScatterNPU91093EightRankFP16Tiling(CoCTilingData &cocTilingData);
    void ReduceScatterNPU91093SixteenRankFP16Tiling(CoCTilingData &cocTilingData);
    void ReduceScatterNPU91093TwoRankFP16Tiling(CoCTilingData &cocTilingData);
    void ReduceScatterNPU91093TwoRankINT8Tiling(CoCTilingData &cocTilingData);
    void ReduceScatterNPU91093FourRankFP16Tiling(CoCTilingData &cocTilingData);

    void CoCAllgatherMatmulReduceScatterAgEightRsTwoTiling(CoCTilingData &cocTilingData);
    void CoCAllgatherMatmulReduceScatterDefaultTiling(CoCTilingData &cocTilingData, int32_t rsDim);
}
#endif // LCAL_TILING_91093_H
