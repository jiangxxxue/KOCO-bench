/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ASCEND_GENATTENTIONMASK_TILING_DATA
#define ASCEND_GENATTENTIONMASK_TILING_DATA

#include <cstdint>

namespace AtbOps {
const uint32_t GEN_ATTENTION_MASK_MAX_BATCH = 32;
struct GenAttentionMaskTilingData {
    int32_t batch;
    int32_t maxSeqLen;
    int32_t blockNum;
    int32_t headNum;
    int32_t qSeqLen[GEN_ATTENTION_MASK_MAX_BATCH];
};
}
#endif // ASCEND_OPS_GENATTENTIONMASK_TILING_DATA_H