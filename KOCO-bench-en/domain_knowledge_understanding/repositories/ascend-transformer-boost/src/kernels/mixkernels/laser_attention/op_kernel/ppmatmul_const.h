/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __PPMATMUL_CONST_BF16_H__
#define __PPMATMUL_CONST_BF16_H__

constexpr int AICFLAGID = 0;
constexpr int AIVFLAGID = 1;
constexpr int AIC2AIVFLAGID = 2;
constexpr int AIV2AICFLAGID = 3;

using T_OUTPUT = float;

constexpr int32_t PING_PONG_NUM = 2;
constexpr int32_t L0AB_PINGPONG_BUFFER_LEN = 16384; // 32 KB
constexpr int32_t BLOCK_SIZE = 16;
constexpr int32_t CUBE_MATRIX_SIZE = 256;          // 16 * 16
constexpr int64_t L1_PINGPONG_BUFFER_LEN = 16384;  // 32 KB
constexpr int64_t L0C_PINGPONG_BUFFER_LEN = 16384; // 64 KB

constexpr int32_t BASE_BLOCK_SIZE = 16384;     // BASE_BLOCK shape ：[128 * 128]
constexpr int32_t BASE_BLOCK_SIZE_192 = 24576; // BASE_BLOCK shape ：[128 * 192]
constexpr int32_t BASE_BLOCK_SIZE_256 = 32768; // BASE_BLOCK shape ：[128 * 256]
constexpr int32_t BASE_BLOCK_SIDE_LEN = 128;   // BASE_BLOCK  row adn column  size

constexpr int32_t B16_SIZE = 2;
constexpr int32_t B32_SIZE = 4;
constexpr int32_t CUBE2_LENGTH_M = 128;
constexpr int32_t CUBE2_LENGTH_K = 128;
constexpr int32_t CUBE2_LENGTH_N = 128;
constexpr int32_t MAX_SWITCH_TIME = 16; // 一个core最多处理16个基本块，因此最多只能有16段

constexpr int32_t VEC_NUM_PER_CUBE = 2;
constexpr int32_t BASE_BLOCK_LENGTH = 128;
// 基本块是方阵，长和宽
constexpr int BASE_BLOCK_SIZE_DOUBLE = BASE_BLOCK_SIDE_LEN * 2;
constexpr int HEAD_DIM = 128;                                                  // head的维度
constexpr int BASE_BLOCK_DATA_NUM = BASE_BLOCK_SIDE_LEN * BASE_BLOCK_SIDE_LEN; // 基本块含有数据量
constexpr int MAX_LENG_PER_UB_PROC = 8192; // UB一次处理的最大长度 (单个ping)
constexpr int MAX_BLOCK_PER_ONE_PROC = MAX_LENG_PER_UB_PROC / BASE_BLOCK_SIDE_LEN;
constexpr int BLOCK_NUM_FOR_VMAX = 16; // 折半计算的基准block数量
constexpr int SHORT_SEQ_THRESHOLD = 8192;
constexpr int MDDIUM_SEQ_THRESHOLD = 32768;
constexpr int BASIC_GAP = BASE_BLOCK_DATA_NUM - BASE_BLOCK_SIDE_LEN;

constexpr float PADDING_FOR_MAX = -1e30; // 非2的幂长度，折半计算vmax时，需要padding

constexpr int TRI_MATRIX_NONE = 0;
constexpr int TRI_MATRIX_TAIL = 1;
constexpr int TRI_MATRIX_HEAD = 2;
constexpr int TRI_MATRIX_HEAD_AND_TAIL = 3;


template <typename TYPE, typename WORKSPACE_TYPE> struct PhyAddrCube1 {
    __gm__ TYPE *left;
    __gm__ TYPE *right;
    __gm__ WORKSPACE_TYPE *out;
    int32_t k = 0;
};

template <typename TYPE, typename WORKSPACE_TYPE> struct PhyAddrCube2Rowsum {
    __gm__ WORKSPACE_TYPE *left;
    __gm__ TYPE *right;
    __gm__ float *out;
    __gm__ float *rowsumOut;
    int32_t k = 0;
};

#endif
