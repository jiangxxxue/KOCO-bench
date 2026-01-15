/*
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ASCEND_OPS_FUSED_ADD_TOPK_DIV_TILING_DATA_H
#define ASCEND_OPS_FUSED_ADD_TOPK_DIV_TILING_DATA_H

#include <cstdint>

namespace AtbOps {
    struct FusedAddTopkDivTilingData {
        uint32_t firstDimSize{0};
        uint32_t secondDimSize{0};
        uint32_t addNumDimSize{0};
        uint32_t groupNum{0};
        uint32_t groupTopk{0};
        uint32_t n{0};
        uint32_t k{0};
        uint32_t activateType{0};
        uint32_t isNorm{0};
        float scale{1.0};
        uint32_t groupEles{0};
        uint32_t blockNum{0};
        uint32_t ubFactorElement{0};
        uint32_t batchPerCore{0};
        uint32_t tailBatch{0};
        uint32_t tilingKey{0};
        uint32_t dtype{1};
        uint32_t tempSize;
        uint32_t enableExpertMapping{0};
        uint32_t expertNum{256};
        uint32_t tableDim{1};
        uint32_t reserved{0};
        uint64_t workspacePerCore;
    };
}
#endif  // ASCEND_OPS_FUSED_ADD_TOPK_DIV_TILING_DATA_H