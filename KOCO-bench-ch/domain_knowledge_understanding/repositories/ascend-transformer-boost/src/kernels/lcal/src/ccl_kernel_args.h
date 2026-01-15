/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LCAL_CCL_KERNEL_ARGS_H
#define LCAL_CCL_KERNEL_ARGS_H

#include "lcal_types.h"
#include "comm_args.h"

namespace Lcal {
struct AscendCCLKernelArgs {
    const void *input = nullptr;
    const void *output = nullptr;
    const void *commArgsPtr = nullptr;
    int64_t count = 0;
    int64_t magic = 0;
    int op = 0;
    int root = 0;
    int cycleCount = 0;
    const void *scale = nullptr;
    int64_t scaleCount = 0;
    const void *offset = nullptr;
};

struct CCLGatherArgs {
    const void *embTable = nullptr;
    const void *lookup = nullptr;
    const void *revData = nullptr;
    int64_t lookupLen = 0;
    int64_t embTableLen = 0;
    int64_t embTableDim = 0;
};
}
#endif // LCAL_CCL_KERNEL_ARGS_H
