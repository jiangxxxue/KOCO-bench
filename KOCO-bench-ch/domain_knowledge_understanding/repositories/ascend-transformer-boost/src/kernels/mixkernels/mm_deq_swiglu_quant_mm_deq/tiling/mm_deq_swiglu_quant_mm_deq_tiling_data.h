/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_MM_DEQ_SWIGLU_QUANT_MM_DEQ_TILING_DATA_H
#define ASCEND_OPS_MM_DEQ_SWIGLU_QUANT_MM_DEQ_TILING_DATA_H

#include <cstdint>

constexpr uint32_t GM_ALIGN_BYTE = 512;

constexpr uint32_t CUSTOM_PRELOAD_STAGES = 1;
constexpr uint32_t CUSTOM_L1_STAGES = 2;
constexpr uint32_t CUSTOM_L0A_STAGES = 2;
constexpr uint32_t CUSTOM_L0B_STAGES = 2;
constexpr uint32_t CUSTOM_L0C_STAGES = 1;
constexpr bool CUSTOM_ENABLE_UNIT_FLAG = true;
constexpr bool CUSTOM_ENABLE_SHUFFLE_K = true;

enum PermuteType {
    PERMUTE_N256,
    PERMUTE_N128,
    PERMUTE_INVALID
};

template <PermuteType PERMUTE_TYPE>
struct Mm1TileArgs {
};

template <>
struct Mm1TileArgs<PERMUTE_N256> {
    static constexpr uint32_t L1M = 128;
    static constexpr uint32_t L1N = 256;
    static constexpr uint32_t EPIM = 32;
};

template <>
struct Mm1TileArgs<PERMUTE_N128> {
    static constexpr uint32_t L1M = 256;
    static constexpr uint32_t L1N = 128;
    static constexpr uint32_t EPIM = 64;
};

constexpr uint32_t MM1_L1K = 512;
constexpr uint32_t MM1_L0K = 128;
constexpr uint32_t MM1_SWIZZLE_OFFSET = 3;
constexpr uint32_t MM1_SWIZZLE_DIRECTION = 0;

constexpr uint32_t MM2_L1A_STAGES = 4;
constexpr uint32_t MM2_L1B_STAGES = 2;
constexpr uint32_t MM2_L0A_STAGES = 4;
constexpr uint32_t MM2_L0B_STAGES = 2;

constexpr uint32_t MM2_L1M = 128;
constexpr uint32_t MM2_L1N = 256;
constexpr uint32_t MM2_L1K = 512;
constexpr uint32_t MM2_L0K = 128;
constexpr uint32_t MM2_EPIM = 32;
constexpr uint32_t MM2_SWIZZLE_OFFSET = 3;
constexpr uint32_t MM2_SWIZZLE_DIRECTION = 0;

constexpr uint32_t WORKSPACE_STAGES = 4;
constexpr uint32_t MAX_TILE_SIZE = 128 * 256;

namespace AtbOps {
struct MmDeqSwigluQuantMmDeqTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t k;
};
}  // namespace AtbOps

#endif