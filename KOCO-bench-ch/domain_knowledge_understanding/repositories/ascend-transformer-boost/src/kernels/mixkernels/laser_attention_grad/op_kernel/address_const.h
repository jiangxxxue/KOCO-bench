/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
/*
 * address_const头文件用于定义在寻址模块中使用到的常量
 */

#ifndef __ADDRESS_CONST_H__
#define __ADDRESS_CONST_H__

namespace Address {
// 定义常量
const int SIZE_128 = 128;
const int SIZE_256 = 256;

// 基本块的length
const int BASE_BLOCK_LENGTH = SIZE_128;

// 读取query,key,value的block size
const int QUERY_BLOCK_SIZE = SIZE_128 * BASE_BLOCK_LENGTH;
const int KEY_BLOCK_SIZE = SIZE_128 * BASE_BLOCK_LENGTH;
const int VALUE_BLOCK_SIZE = SIZE_128 * BASE_BLOCK_LENGTH;
const int ATTENTION_SCORE_BLOCK_SIZE = BASE_BLOCK_LENGTH * BASE_BLOCK_LENGTH;

// 反向最长的分段数
const int MAX_LENGTH = 16;

// 反向寻址预处理的结构体
struct BackWardAddr {
    int32_t b;    // batch索引
    int32_t n;    // head索引
    int32_t iR;  // 行索引
    int32_t iC;  // 列索引
    int32_t kx;   // 行数
    int32_t ky;   // 列数
    int32_t k;    // 基本块的数量
};

// 反向cube1寻址模块
template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
struct PhyAddrBackwardCube1 {
    __gm__ T_LEFT *left;      // left起始位置
    __gm__ T_RIGHT *right;    // right起始位置
    __gm__ T_OUTPUT *out;     // out起始位置
    int32_t kx = 0;           // x方向的长度
    int32_t ky = 0;           // y方向的长度
    int32_t k = 0;            // 总共的块数
    int32_t lineStride = 0;  // 行和行的间隔
    bool lowerLeft;          // 左下角是否不需要计算：true代表不需要计算，false代表需要计算
    bool upperRight;         // 右上角是否不需要计算： true代表不需要计算，false代表需要计算
};

// 反向cube2寻址模块
template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
struct PhyAddrBackwardCube2 {
    __gm__ T_LEFT *left;      // left起始位置
    __gm__ T_RIGHT *right;    // right起始位置
    __gm__ T_OUTPUT *out;     // out起始位置
    int32_t kx = 0;           // x方向的长度
    int32_t ky = 0;           // y方向的长度
    int32_t k = 0;            // 总共的块数
    int32_t lineStride = 0;  // 行和行的间隔
    bool lowerLeft;          // 左下角是否不需要计算：true代表不需要计算，false代表需要计算
    bool upperRight;         // 右上角是否不需要计算： true代表不需要计算，false代表需要计算
};

// 反向cube3寻址模块
template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
struct PhyAddrBackwardCube3 {
    __gm__ T_LEFT *left;      // left起始位置
    __gm__ T_RIGHT *right;    // right起始位置
    __gm__ T_OUTPUT *out;     // out起始位置
    int32_t kx = 0;           // x方向的长度
    int32_t ky = 0;           // y方向的长度
    int32_t k = 0;            // 总共的块数
    int32_t lineStride = 0;  // 行和行的间隔
    bool lowerLeft;          // 左下角是否不需要计算：true代表不需要计算，false代表需要计算
    bool upperRight;         // 右上角是否不需要计算： true代表不需要计算，false代表需要计算
};

// 反向vector寻址预处理的结构体
struct VectorAddr {
    int32_t b = 0;    // batch索引
    int32_t n = 0;    // head索引
    int32_t iR = 0;  // 行索引
    int32_t iC = 0;  // 列索引
    int32_t kx = 0;   // 行数
    int32_t ky = 0;   // 列数
    int32_t k = 0;    // 基本块的数量
};

// vector的全局信息
struct GLOBAL_INFO {
    int64_t cubeNum = 0;           // cube数量
    int64_t blockNumPerCube = 0;   // 每个cube处理block的数量
    int64_t headNum = 0;           // head数量
    int64_t batchNum = 0;          // batch数量
    int64_t seqLenQ = 0;           // query的序列长度
    int64_t seqLenK = 0;           // key、value的序列长度
    int64_t headDim = 0;           // head-dim = 128;
    bool triMatrix = false;        // 是否是三角阵
    int32_t blockRows = 0;         // 负载均衡前的行数
    int32_t blockCols = 0;         // 负载均衡前的列数
    int64_t blockNumPerRow = 0;    // 一行有几个block（三角阵和非三角阵不同）
    int64_t blockNumPerCol = 0;    // 一列有几个block（也就是一个head有几个行）
    int64_t blockNumPerLoop = 0;   // 一个loop处理block的数量
    int64_t blockNumPerHead = 0;   // 一个head包含的block数量
    int64_t blockNumPerBatch = 0;  // 一个batch包含的block数量
    int64_t tailBlockNum = 0;      // 最后一次循环处理的block数量
    int64_t isSparse = 0;          // 是否为sparse 魔道e
    int64_t windowLength = 0;      // 滑动窗口的length
    int64_t windowsBlockNum = 0;   // 滑窗大小, sparse使用
};

// vector的当前信息
struct LOCAL_INFO {
    int64_t cubeIndex = 0;                // 所属的cube分组（第几个cube很重要）
    int64_t vectorIdx = 0;                // 当前vector在cube内的编号（0或1）
    int64_t startLineInBaseBlock = 0;  // 处理基本块的起始行（都是2个vector处理一个基本块）
    bool procTail = false;                // 是否参与尾块处理
    int64_t procTailBlockNum = 0;       // 处理尾块中的block数量
};

// vector的分段信息
struct SECTION_INFO {
    int64_t sectionNum = 0;                               // 当前处理的block条，包含几个块
    int64_t section_start_block[MAX_SWITCH_TIME] = {0};    // 起始block编号
    int64_t head_skip_block[MAX_SWITCH_TIME] = {0};        // 当前section的头尾是否要跳过计算
    int64_t tail_skip_block[MAX_SWITCH_TIME] = {0};        // 当前section的头尾是否要跳过计算
    int64_t global_lines_in_heads[MAX_SWITCH_TIME] = {0};  // 在所有heads中的起始行
    int64_t len[MAX_SWITCH_TIME] = {0};                    // 每一段的长度
    bool head_apply_mask[MAX_SWITCH_TIME] = {false};
    bool tail_apply_mask[MAX_SWITCH_TIME] = {false};
    int64_t O_dO_offset[MAX_SWITCH_TIME] = {0};  // 矩阵O dO的偏移量
    int64_t S_dP_offset[MAX_SWITCH_TIME] = {0};  // 矩阵S dP的偏移量
    int64_t processLines = {0};                 // 处理的行数
};
}  // namespace Address
#endif