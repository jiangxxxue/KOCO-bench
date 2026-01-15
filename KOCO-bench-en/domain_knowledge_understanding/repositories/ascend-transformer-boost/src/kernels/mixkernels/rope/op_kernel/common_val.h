/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef COMMON_VAL_H
#define COMMON_VAL_H

constexpr uint32_t NUM_TWO = 2; // 2
constexpr uint32_t NUM_FOUR = 4; // 4
constexpr uint32_t BLK_SIZE = 32; // 一个block字节数
constexpr uint32_t ELE_NUM_FP16 = 16; // 一个block fp16元素个数
constexpr uint32_t ELE_NUM_FP32 = 8; // 一个block字节数 fp32元素个数
constexpr uint32_t MAX_REPEAT_TIME = 255; // copy的repeatTimes不能超过255
constexpr uint32_t REPEAT_SIZE_FP32 = 64;
constexpr uint32_t MAX_LEN_FP16 = 8192; // 非fp16情况下最大长度（hiddensize）
constexpr uint8_t DEFAULT_REPEAT_STRIDE = 8; // 默认stride, 8 * 32 = 256
constexpr int64_t REG_910B = 48; // 饱和模式寄存器位置
constexpr int64_t REG_310P = 53; // 饱和模式寄存器位置
constexpr int64_t SLICE_SIZE = 4096; // 切片大小
constexpr int64_t SLICE_SIZE_FP16 = 12288;
constexpr int64_t SLICE_SIZE_FP16_LARGE_NTOKENS = 4096;
#endif