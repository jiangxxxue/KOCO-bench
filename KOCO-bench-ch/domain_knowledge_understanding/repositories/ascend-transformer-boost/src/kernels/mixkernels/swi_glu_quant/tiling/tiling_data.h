/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * AscendOpCommonLib is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 * http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef ATBOPS_SWI_GLU_QUANT_TILING_DATA
#define ATBOPS_SWI_GLU_QUANT_TILING_DATA

#include <cstdint>

namespace AtbOps {

struct SwiGluQuantTilingData {
    uint32_t groupLen;
    uint32_t rowLen;
    uint32_t colLen;
    uint32_t rowLenPerHeadCore;
    uint32_t rowLenPerTailCore;
    uint32_t basicRowLenHeadCore;
    uint32_t basicRowLenTailCore;
    uint32_t basicColLen;
    uint32_t headCoreNum;
    uint32_t realCoreNum;
    uint32_t totalCore = 1; // 后面的是在tiling中使用
    uint32_t ubSize = 0;
    uint32_t groupLength = 1;
    uint32_t dataNumSingleUb = 1;      // UB空间可处理的最大数据量
    uint32_t blockNum;            // 32B对齐使用
    uint32_t cacheLineLen = 1;         // 512B对齐使用
    uint32_t coreNumUsed = 0;
    uint32_t optBaseColLen = 1;
    uint32_t optBaseRowLenHeadCore = 1;
    uint32_t optBaseRowLenTailCore = 1;
};
}
#endif