/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ASCEND_OPS_PAGED_CACHE_LOAD_TILING_DEPENDENCY_H
#define ASCEND_OPS_PAGED_CACHE_LOAD_TILING_DEPENDENCY_H

namespace AtbOps {

struct PagedCacheLoadTilingData {
    int32_t blockSize;
    int32_t numTokens;
    int32_t numblkTabCol;
    int32_t tokenSizeK;
    int32_t tokenSizeV;
    int32_t typeByte;
    int32_t hasSeqStarts;
    int32_t cuSeqLens;
};
}

#endif // ASCEND_OPS_PAGED_CACHE_LOAD_TILING_DEPENDENCY_H
