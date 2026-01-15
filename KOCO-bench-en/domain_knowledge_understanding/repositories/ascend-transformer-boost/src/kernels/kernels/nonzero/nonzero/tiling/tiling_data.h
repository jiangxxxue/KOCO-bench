/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASDOPS_NONZRERO_TILING_DATA
#define ASDOPS_NONZRERO_TILING_DATA

#include <cstdint>

constexpr uint32_t MAX_DIM_NUM = 8;
namespace AsdOps {
struct NonzeroTilingData {
    uint32_t xdimLength{1};
    uint32_t xNumel{1};
    uint32_t xDims[MAX_DIM_NUM];
};
}
#endif