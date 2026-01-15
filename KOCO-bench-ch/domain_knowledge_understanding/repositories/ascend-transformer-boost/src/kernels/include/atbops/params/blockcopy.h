/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATBOPS_PARAMS_BLOCKCOPY_H
#define ATBOPS_PARAMS_BLOCKCOPY_H

namespace AtbOps {
namespace OpParam {
struct BlockCopy {
    enum Type {
        BLOCK_COPY_CACHE_ND = 0,
        BLOCK_COPY_CACHE_NZ = 1
    };
    Type type = BLOCK_COPY_CACHE_ND;
    bool operator==(const BlockCopy &other) const
    {
        return this->type == other.type;
    }
};
} // namespace OpParam
} // namespace AtbOps

#endif // ATBOPS_PARAMS_BLOCKCOPY_H