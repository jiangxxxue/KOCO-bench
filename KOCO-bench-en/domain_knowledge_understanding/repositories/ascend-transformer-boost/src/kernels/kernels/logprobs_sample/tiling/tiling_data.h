/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASDOPS_LOGPROBS_SAMPLE_TILING_DATA_H
#define ASDOPS_LOGPROBS_SAMPLE_TILING_DATA_H
#include <cstdint>

namespace AsdOps {
struct LogprobsSampleTilingData {
    uint32_t batchSize;           // 输入sortedProbs第一维大小，也是输出第一维的大小
    uint32_t probsSize;           // 输入sortedProbs第二维大小，用于计算拷贝输入的offset
};
}

#endif // ASDOPS_LOGPROBS_SAMPLE_TILING_DATA_H
