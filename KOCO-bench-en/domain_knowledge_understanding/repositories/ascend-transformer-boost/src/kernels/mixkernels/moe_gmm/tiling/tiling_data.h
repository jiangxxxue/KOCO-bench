/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ASCEND_OPS_MOE_GMM_TILING_DATA
#define ASCEND_OPS_MOE_GMM_TILING_DATA

#include <cstdint>

namespace AtbOps {
struct MoeGmmTilingData {
    uint32_t batch{0};
    uint32_t m{0};
    uint32_t k{0};
    uint32_t n{0};
    uint32_t m0{0};
    uint32_t k0{0};
    uint32_t n0{0};
    uint32_t mLoop{0};
    uint32_t kLoop{0};
    uint32_t nLoop{0};
    uint32_t coreLoop{0};
    uint32_t swizzlCount{0};
    uint32_t tilingKey{0};
    uint32_t blockDim{1};
    uint32_t swizzlDirect{0};
    uint32_t splitk{0};
    uint32_t enShuffleK{0};
    uint32_t allM{1};
    uint32_t moeUp{0};
};
} // namespace AtbOps

#endif // ASCEND_OPS_MOE_GMM_TILING_DATA
