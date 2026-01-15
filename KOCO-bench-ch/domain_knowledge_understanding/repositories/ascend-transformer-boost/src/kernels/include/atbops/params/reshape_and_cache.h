/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATBOPS_PARAMS_RESHAPE_AND_CACHE_H
#define ATBOPS_PARAMS_RESHAPE_AND_CACHE_H


namespace AtbOps {
namespace OpParam {
struct ReshapeAndCache {
    enum Type {
        RESHAPE_AND_CACHE_ND = 0,
        RESHAPE_AND_CACHE_NZ = 1,
        RESHAPE_AND_CACHE_WINS = 2,
        RESHAPE_AND_CACHE_WINS_ROPE = 3,
        RESHAPE_AND_CACHE_ND_SISO = 4,
        RESHAPE_AND_CACHE_OMNI_COMPRESS = 5
    };
    Type type = RESHAPE_AND_CACHE_ND;

    bool operator==(const ReshapeAndCache &other) const
    {
        return this->type == other.type;
    }
};
} // namespace OpParam
} // namespace AtbOps

#endif // ATBOPS_PARAMS_RESHAPE_AND_CACHE_H