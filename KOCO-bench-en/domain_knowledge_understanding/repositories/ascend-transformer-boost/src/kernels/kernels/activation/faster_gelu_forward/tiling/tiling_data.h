/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_OP_COMMON_LIB_TILING_DATA_H
#define ASCEND_OP_COMMON_LIB_TILING_DATA_H

#include <cstdint>

namespace AsdOps {
constexpr uint32_t FASTER_GELU_FORWARD_BUFF_NUM = 2;
constexpr uint32_t MAX_CORE_SIZE = 128;
struct FasterGeluForwardTilingData {
    uint32_t usedCoreNum = 0;
    uint32_t maxTileLen = 0;
    uint32_t alignDataNum = 0;
    uint32_t singleCoreDataLen[MAX_CORE_SIZE] = {0};
};
} // namespace AsdOps
#endif // ASCEND_OP_COMMON_LIB_TILING_DATA_H
