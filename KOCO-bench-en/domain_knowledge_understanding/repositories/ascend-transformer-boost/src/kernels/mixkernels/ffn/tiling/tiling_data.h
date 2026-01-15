/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ASCEND_OPS_FFN_TILING_DATA_H
#define ASCEND_OPS_FFN_TILING_DATA_H

#include <cstdint>

namespace AtbOps {
struct FFNMatmulTilingData {
    uint32_t coreLoops{0};
    uint32_t m{0};
    uint32_t k{0};
    uint32_t n{0};
    uint32_t baseM{0};
    uint32_t baseK{0};
    uint32_t baseN{0};
    uint32_t mLoops{0};
    uint32_t kLoops{0};
    uint32_t nLoops{0};
    uint32_t swizzlCount{0};
    uint32_t swizzlDirect{0};
};

struct FFNTilingData {
    uint32_t activationType{0};
    uint32_t scaleOffset{0};
    uint32_t biasOffset{0};
    uint32_t syncOffset{0};
    uint32_t cOffset{0};
    uint32_t geluOffset{0};
    FFNMatmulTilingData mm1;
    FFNMatmulTilingData mm2;
};

}  // namespace AtbOps

#endif