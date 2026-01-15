/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATBOPS_PARAMS_KVCACHE_H
#define ATBOPS_PARAMS_KVCACHE_H

#include <cstdint>
#include <string>
#include <sstream>
#include <mki/utils/SVector/SVector.h>

namespace AtbOps {
namespace OpParam {
struct KVCache {
    enum Type {
        KVCACHE_ND = 0,
        KVCACHE_NZ = 3,
        KVCACHE_DYNAMIC_BATCH = 5,
        KVCACHE_ND_PARAMS = 7,
        KVCACHE_NZ_PARAMS = 9,
        KVCACHE_DYNAMIC_BATCH_PARAMS = 11,
    };
    Type type;
    std::vector<int32_t> qSeqLen;
    std::vector<int32_t> kvSeqLen;
    std::vector<int32_t> batchRunStatus;

    std::vector<int32_t> seqLen;
    std::vector<int32_t> tokenOffset;

    bool operator==(const KVCache &other) const
    {
        return this->type == other.type && this->qSeqLen == other.qSeqLen && this->kvSeqLen == other.kvSeqLen &&
               this->batchRunStatus == other.batchRunStatus;
    }
};
} // namespace OpParam
} // namespace AtbOps

#endif // ATBOPS_PARAMS_KVCACHE_H