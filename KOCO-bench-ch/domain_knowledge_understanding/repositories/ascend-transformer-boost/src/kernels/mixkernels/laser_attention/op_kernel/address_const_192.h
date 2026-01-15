/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ADDRESS_CONST_192_H__
#define __ADDRESS_CONST_192_H__

namespace Address_192 {
    // 定义常量
    const int SIZE_128 = 128;
    const int SIZE_192 = 192;
    const int SIZE_256 = 256;
    const int DIVISOR_2 = 2;

    // 基本块的length
    const int BASE_BLOCK_LENGTH = SIZE_128;

    // 读取query,key,value的block size
    const int QUERY_BLOCK_SIZE = SIZE_192 * BASE_BLOCK_LENGTH;
    const int KEY_BLOCK_SIZE = SIZE_256 * BASE_BLOCK_LENGTH;
    const int VALUE_BLOCK_SIZE = SIZE_128 * BASE_BLOCK_LENGTH;
    const int ROWSUM_BLOCK_SIZE = SIZE_128;
    const int ATTENTION_SCORE_BLOCK_SIZE = BASE_BLOCK_LENGTH * BASE_BLOCK_LENGTH;
    const int OUTPUT_BLOCK_SIZE = SIZE_128 * BASE_BLOCK_LENGTH;

    // vector的全局信息
    struct GLOBAL_INFO {
        int64_t cubeNum = 0;              // cube数量
        int64_t blockNumPerCube = 0;      // 每个cube处理block的数量
        int64_t headNum = 0;              // head数量
        int64_t batchNum = 0;             // batch数量
        int64_t seqLenQ = 0;              // query的序列长度
        int64_t seqLenK = 0;              // key、value的序列长度
        int64_t headDim = 0;              // headDim = 192;
        bool triMatix = false;            // 是否是三角阵
        int64_t blockRows;                // 行数
        int64_t blockCols;                // 列数
        int64_t blockNumPerRow = 0;    // 一行有几个block（三角阵和非三角阵不同）
        int64_t blockNumPerCol = 0;    // 一列有几个block（也就是一个head有几个行）
        int64_t blockNumPerLoop = 0;   // 一个loop处理block的数量
        int64_t blockNumPerHead = 0;   // 一个head包含的block数量
        int64_t blockNumPerBatch = 0;  // 一个batch包含的block数量
        int64_t loopTimes = 0;           // 大循环次数
        int64_t tailBlockNum = 0;       // 最后一次循环处理的block数量
        int64_t isSparse = 0;            // 是否为sparse 魔道e
        int64_t windowLength = 0;        // 滑动窗口的length
        int64_t windowsBlockNum = 0;    // 滑窗大小, sparse使用
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
    const int MAX_SWITCH_TIME = 16;
    struct SECTION_INFO {
        int64_t sectionNum = 0;                               // 当前处理的block条，包含几个块
        int64_t sectionStartBlock[MAX_SWITCH_TIME] = {0};    // 起始block编号
        int64_t headSkipBlock[MAX_SWITCH_TIME] = {0};          // 当前section的头尾是否要跳过计算
        int64_t tailSkipBlock[MAX_SWITCH_TIME] = {0};          // 当前section的头尾是否要跳过计算
        int64_t globalLinesInHeads[MAX_SWITCH_TIME] = {0};  // 在所有heads中的起始行
        int64_t len[MAX_SWITCH_TIME] = {0};                    // 每一段的长度
        bool headApplyMask[MAX_SWITCH_TIME] = {false};
        bool tailApplyMask[MAX_SWITCH_TIME] = {false};
        int64_t dOutputAndOutputOffset[MAX_SWITCH_TIME] = {0};  // 矩阵O dO的偏移量
        int64_t dSoftmaxScoreAndScoreOffset[MAX_SWITCH_TIME] = {0};  // 矩阵S dP的偏移量
        int64_t processLines = {0};                 // 处理的行数
    };

    // 前向cube1寻址的结构体
    template<typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
    struct PHY_ADDR_FORWARD_CUBE1 {
        __gm__ T_LEFT *left;                      // 左矩阵的起始地址，基本块为 128 * 192
        __gm__ T_RIGHT *right;                    // 右矩阵的起始地址， 基本块为 128 * 192
        __gm__ T_OUTPUT *out;                     // 输出的起始地址， 基本块为 128 * 128
        int32_t k = 0;                            // 连续计算的基本块数量
    };

    // 前向cube2寻址的结构体
    template<typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
    struct PHY_ADDR_FORWARD_ROWSUM_CUBE2 {
        __gm__ T_LEFT *left;                      // 左矩阵的起始地址，基本块为 128 * 128
        __gm__ T_RIGHT *right;                    // 右矩阵的起始地址， 基本块为 128 * 192
        __gm__ T_OUTPUT *out;                     // cube2输出的起始地址， 基本块为 128 * 192
        __gm__ T_OUTPUT *rowsumOut;                  // row_sum输出的起始地址， 基本块为 128 * 1
        int32_t k = 0;                            // 连续计算的基本块数量
    };
}
#endif