/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ASCEND_OPS_MOE_GATE_CORR_TILING_DATA_H
#define ASCEND_OPS_MOE_GATE_CORR_TILING_DATA_H

#include <cstdint>

namespace AsdOps {

struct MoeGateCorrTilingData {
    uint32_t m{0};
    uint32_t k{0};
    uint32_t n{0};
};

}  // namespace AsdOps

#endif