/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ASCEND_OPS_GATING_TILING_DATA
#define ASCEND_OPS_GATING_TILING_DATA

#include <cstdint>

// 实际用到的最大核数，通过该值设置数组长度，保证大于或等于实际NPU核数即可
constexpr int32_t MAX_CORE_NUM = 50;

namespace AtbOps {
struct GatingTilingData {
    int32_t topkExpertNum;                                          // 每个token选的专家数量
    int32_t topKNum;                                                // 输入长度，即一维tensor中元素数量
    int32_t topKNumPadded;                                          // 输入长度topKNum按2048个元素对齐后的长度
    int32_t cumSumNum;                                              // topk之后的专家总数量
    int32_t cumSumNum32BytesPadded;                                 // cumSumNum按32字节对齐
    int32_t actualCoreNum;                                          // 实际用到核数量
    int32_t blockNum;                                               // 数据块总数量
    int32_t tailBlockDataSize;                                      // 尾块有效数据长度，单位为元素
    int32_t syncSize;                                               // 供IBSet/IBWait使用的额外空间大小，单位字节
    int32_t deviceExpertNum;                                        // 有效的专家数量
    int32_t blockNumPerCore[MAX_CORE_NUM] = {0};                    // 每个核分配块数量
    int32_t beginBlockIndexPerCore[MAX_CORE_NUM] = {0};             // 每个核起始块下标
    int32_t offSetPerCore[MAX_CORE_NUM] = {0};                      // 每个核需要计算数据的偏移，单位为元素
    uint32_t cumSumInt64;                                           // cumSumNum是否输出为int64格式
    int32_t deviceExpert[1025] = {0};                               // 从小到大排序的有效的专家列表
};
}

#endif // ASCEND_OPS_GATING_TILING_DATA
