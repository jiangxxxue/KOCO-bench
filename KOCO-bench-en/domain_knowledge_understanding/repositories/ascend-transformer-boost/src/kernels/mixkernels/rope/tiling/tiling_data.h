/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ATBOPS_ROPE_TILING_DATA
#define ATBOPS_ROPE_TILING_DATA

#include <cstdint>

namespace AtbOps {
struct RopeTilingData {
    uint32_t hiddenSizeQ{16};
    uint32_t hiddenSizeK{16};
    uint32_t headDim{1}; // qk头长度的最大值
    uint32_t headNumQ{1};
    uint32_t headNumK{1};
    uint32_t rotaryCoeff{4}; // 旋转系数
    uint32_t ntokens{1}; // 总token数
    uint32_t realCore{0}; // 实际用到核数
    uint32_t cosFormat{0}; // 是否复用cos sin
    uint32_t batch{32}; // 几个batch
    uint32_t maxUbSize{0}; // 最大UB内存
    uint32_t multiple{1}; // ntokens减小，qk增大的倍率
};
}
#endif