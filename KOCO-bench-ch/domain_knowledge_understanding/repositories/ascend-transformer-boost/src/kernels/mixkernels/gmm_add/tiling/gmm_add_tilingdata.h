/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ASCEND_OPS_GMM_ADD_TILING_DATA_H
#define ASCEND_OPS_GMM_ADD_TILING_DATA_H

#include <cstdint>
#include "tiling/tiling_api.h"

namespace AtbOps {
constexpr uint16_t MAX_TENSOR_LIST_SIZE = 128;

struct GmmBaseParam {
    uint32_t groupNum{0};
    uint32_t coreNum{0};
    uint32_t activeType{0};
    uint32_t ubBaseK{0};
    uint32_t ubBaseN{0};
    uint32_t ubCalSize{0};
    uint32_t ubRestBytes{0};
    uint32_t singleWeight{0};
    uint32_t singleX{0};
    uint32_t singleY{0};
    int32_t groupType{2};
    uint32_t singleN{0};
    uint32_t isPerTokenQuant{0};
    uint32_t groupListType{0};
    uint64_t workspaceSize{0};
};

struct GmmArray {
    int32_t mList[MAX_TENSOR_LIST_SIZE];
    int32_t kList[MAX_TENSOR_LIST_SIZE];
    int32_t nList[MAX_TENSOR_LIST_SIZE];
};

struct GmmTilingData {
    GmmBaseParam gmmBaseParams;
    GmmArray gmmArray;
    optiling::TCubeTiling mmTilingData;
};
} // namespace AtbOps

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_GMM_ADD_H