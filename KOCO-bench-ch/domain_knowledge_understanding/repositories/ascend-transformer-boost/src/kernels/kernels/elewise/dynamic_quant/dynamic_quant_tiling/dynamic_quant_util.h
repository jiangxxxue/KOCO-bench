/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef DYNAMIC_QUANT_UTIL_H
#define DYNAMIC_QUANT_UTIL_H

namespace AsdOps {
constexpr uint32_t DYNAMIC_QUANT_BUFFER_NUM_ONE = 1;
constexpr uint32_t DYNAMIC_QUANT_BUFFER_NUM_TWO = 2;
constexpr uint32_t DYNAMIC_QUANT_SIZE_OF_INT8 = 1;
constexpr uint32_t DYNAMIC_QUANT_SIZE_OF_HALF = 2;
constexpr uint32_t DYNAMIC_QUANT_SIZE_OF_FLOAT = 4;
constexpr uint32_t DYNAMIC_QUANT_ALIGN_SIZE = 32;
constexpr uint32_t DYNAMIC_QUANT_ALIGN_NUM_X = 16;
constexpr uint32_t DYNAMIC_QUANT_ALIGN_NUM_SCALE = 8;
constexpr uint32_t DYNAMIC_QUANT_MAX_VALUE = 127;
constexpr uint32_t DYNAMIC_QUANT_MIN_VALUE = -128;
constexpr uint32_t DYNAMIC_QUANT_ASYMMETRIC_VALUE = 255;
constexpr uint32_t DYNAMIC_QUANT_HEADSPACE = 10240;
constexpr uint32_t DYNAMIC_QUANT_COPY_ROW_SCALE = 5;
constexpr uint32_t DYNAMIC_QUANT_FP16_BUF_SCALE = 2;
/**
 * rowSuit = Utils::RoundUp(213 - sizeX * 2 / 7, 8);
 * This formula is summarized after testing. from 512->64 and 64->192.
 */
constexpr uint32_t DYNAMIC_QUANT_COPY_ROW_LONG = 64;
constexpr uint32_t DYNAMIC_QUANT_COPY_ROW_SHORT = 192;
constexpr uint32_t DYNAMIC_QUANT_LEN_H_LONG = 512;
constexpr uint32_t DYNAMIC_QUANT_LEN_H_SHORT = 64;
constexpr uint32_t DYNAMIC_QUANT_ROW_SUIT_MUL = 2;
constexpr uint32_t DYNAMIC_QUANT_ROW_SUIT_DIV = 7;
constexpr uint32_t DYNAMIC_QUANT_ROW_SUIT_ADD = 213;

constexpr uint32_t DYNAMIC_QUANT_STATUS_ALIGN = 0;
constexpr uint32_t DYNAMIC_QUANT_STATUS_UNALIGN_310P = 1;
constexpr uint32_t DYNAMIC_QUANT_STATUS_UNALIGN_910B = 2;
constexpr float DYNAMIC_QUANT_EPSINON = 1e-12;

constexpr uint32_t DYNAMIC_QUANT_BF16_LAST_DIM_LIMITATION = 7552; // This formula is summarized after testing.
constexpr uint32_t DYNAMIC_QUANT_FP16_LAST_DIM_LIMITATION_310P = 4096; // This formula is summarized after testing.
}

#endif