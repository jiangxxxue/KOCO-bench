/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDOPS_DYNAMIC_QUANT_TILING_DATA
#define ASDOPS_DYNAMIC_QUANT_TILING_DATA

#include <stdint.h>

namespace AsdOps {
struct DynamicQuantTilingData {
    uint32_t numCore{0};
    uint32_t sizeH{0}; // value of last dim
    uint32_t sizeX{0}; // value of 32byte alignment for last dim of intensor x
    uint32_t sizeZOut{0}; // value of 32byte alignment for last dim of outtensor z
    uint32_t sizeCopyRow{0}; // value of 32byte alignment for numCopyRow
    uint32_t numCopyRow{0}; // copy num row for one time
    uint32_t numHeadCore{0}; // the core doing one more session
    uint32_t numTailCore{0}; // the core doing less one session
    uint32_t numHeadTimes{0}; // the times for the core doing one more session
    uint32_t numTailTimes{0}; // the times for the core doing less one session
    uint32_t numLastTailRow{0}; // the num of row for tail batch
    uint32_t alignType{0}; // 0: DynamicQuantAlign, 1: 310p unalign, 2: 910b unalign
    uint32_t asymmetric{0}; // false : symmetric，true : asymmetric
};
} // namespace AsdOps
#endif