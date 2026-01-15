/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ASCEND_OPS_STRIDEDBATCHMATMUL_TILING_DATA_H
#define ASCEND_OPS_STRIDEDBATCHMATMUL_TILING_DATA_H

#include <cstdint>

namespace AtbOps {
    struct StridedBatchMatmulTilingData {
        int32_t transA = 0;
        int32_t transB = 0;
        int32_t batch = 0;
        int32_t headNum = 0;
        int32_t blockNum = 0;
    };
    struct StridedBatchMatmulSampleTilingData {
        int32_t batchOffsetA = 0;
        int32_t batchOffsetB = 0;
        int32_t batchOffsetC = 0;
        int32_t strideA = 0;
        int32_t strideB = 0;
        int32_t strideC = 0;
    };
}
#endif // ASCEND_OPS_STRIDEDBATCHMATMUL_TILING_DATA_H