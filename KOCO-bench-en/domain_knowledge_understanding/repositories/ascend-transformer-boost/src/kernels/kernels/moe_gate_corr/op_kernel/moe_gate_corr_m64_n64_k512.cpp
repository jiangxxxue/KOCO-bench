/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "moe_gate_corr.h"

extern "C" __global__ __aicore__ void moe_gate_corr_m64_n64_k512(
    GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR tiling)
{
    auto tilingData = GetTilingData(tiling);

    GemmCoord problemShape{tilingData.m, tilingData.n, tilingData.k};
    layout::RowMajor layoutA{tilingData.m, tilingData.k};
    layout::ColumnMajor layoutB{tilingData.k, tilingData.n};
    layout::RowMajor layoutC{tilingData.m, tilingData.n};

    using L1TileShape = GemmShape<64, 64, 512>;
    using L0TileShape = GemmShape<64, 64, 128>;

    MoeGateCorr<L1TileShape, L0TileShape, layout::RowMajor, layout::ColumnMajor, layout::RowMajor, float, float>(
        problemShape,
        gmA, layoutA,
        gmB, layoutB,
        gmC, layoutC
    );
}
