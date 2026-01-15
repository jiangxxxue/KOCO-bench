/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ADDRESS_MODULE_ADDRESSMAPPING_FORWARD_H
#define ADDRESS_MODULE_ADDRESSMAPPING_FORWARD_H

#include "address_const_192.h"

namespace Address_192 {
    template<typename TYPE>
    class AddressMappingForward {
    public:
        // B N S D 格式的基础信息
        int64_t batchSize;                    // batch批次的大小
        int64_t headNum;                      // head数量
        int64_t gqaGroupNum;                 // GQA场景的组数
        int64_t querySequenceLen;            // query序列长度
        int64_t keyValueSequenceLen;        // key、value序列长度
        int64_t outputSequenceLen;           // 输出O的序列长度
        int64_t rowSumSequenceLen;          // row_sum的序列长度
        int64_t headDim;                      // head_dim的长度

        // 核数、核组的信息
        int64_t coreNum;                      // 使用的核心数量
        int64_t coreGroupNum;                // 核数数量
        int64_t coreNumPerGroup;            // 每一核组的核心数量
        int64_t curCoreIndex;                // 当前核心的序号
        int64_t curGroupIndex;               // 当前组的序号
        int64_t groupLocalIndex;             // 组内核心的序号

        // 偏移相关信息
        bool isTriangle;                      // 倒三角mask的标志
        int64_t forwardBlockNumPerCol;     // 负载均衡后attention的行数
        int64_t forwardBlockNumPerRow;     // 负载均衡后attention的列数
        int64_t forwardBlockRowsPerHead;   // 前向每个head的基本行块数量
        int64_t forwardBlockRowsPerBatch;  // 前向每个batch的基本行块数量
        int64_t forwardTotalRows;            // 前向计算的总行块数
        int64_t forwardTotalRounds;          // 前向总共的轮次
        int64_t blockNumPerCoreForward;    // 前向每个核心平均处理的基本块数量
        int64_t remainBlockNumForward;      // 前向尾块的数量

    public:
        __aicore__ __inline__ void set_balance_info()
        {
            // no-mask场景
            this->forwardBlockNumPerCol = this->querySequenceLen / SIZE_128;
            this->forwardBlockNumPerRow = this->keyValueSequenceLen / SIZE_128;

            // 倒三角mask场景：
            if (this->isTriangle) {
                this->forwardBlockNumPerCol = this->querySequenceLen / SIZE_128 / DIVISOR_2;
                this->forwardBlockNumPerRow = this->keyValueSequenceLen / SIZE_128 + 1;
            }

            // 计算head、batch的基本块
            this->forwardBlockRowsPerHead = this->forwardBlockNumPerCol;
            this->forwardBlockRowsPerBatch = this->forwardBlockRowsPerHead * this->headNum;
            this->forwardTotalRows = this->forwardBlockRowsPerBatch * this->batchSize;

            // 平均块和尾块数量的计算
            this->blockNumPerCoreForward = this->forwardBlockNumPerRow / this->coreNumPerGroup;
            this->remainBlockNumForward = this->forwardBlockNumPerRow % this->coreNumPerGroup;

            // 轮次
            this->forwardTotalRounds =
                    (this->forwardTotalRows + this->coreGroupNum - 1) / this->coreGroupNum;
        }

    public:
        __aicore__ __inline__ void
        init(int64_t batchSize, int64_t headNum, int64_t gqaGroupNum, int64_t querySequenceLen,
            int64_t keyValueSequenceLen, int64_t headDim, bool isTriangle)
        {
            this->batchSize = batchSize;
            this->headNum = headNum;
            this->gqaGroupNum = gqaGroupNum;
            this->querySequenceLen = querySequenceLen;
            this->keyValueSequenceLen = keyValueSequenceLen;
            this->headDim = headDim;
            this->isTriangle = isTriangle;

            this->outputSequenceLen = this->querySequenceLen;
            this->rowSumSequenceLen = this->querySequenceLen;
        }

        __aicore__ __inline__ void
        set_core_info(int64_t coreNum, int64_t coreGroupNum, int64_t coreNumPerGroup, int64_t curCoreIndex,
                      int64_t curGroupIndex, int64_t groupLocalIndex)
        {
            this->coreNum = coreNum;
            this->coreGroupNum = coreGroupNum;
            this->coreNumPerGroup = coreNumPerGroup;
            this->curCoreIndex = curCoreIndex;
            this->curGroupIndex = curGroupIndex;
            this->groupLocalIndex = groupLocalIndex;

            // 计算轮次等信息
            set_balance_info();
        }

        __aicore__ __inline__ int64_t get_total_round()
        {
            return this->forwardTotalRounds;
        }

        __aicore__ __inline__ bool is_running(int64_t round_id)
        {
            return (round_id * this->coreGroupNum + this->curGroupIndex) < this->forwardTotalRows;
        }

        template<typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
        __aicore__ __inline__ void addrMapping_cube1(__gm__ T_LEFT *__restrict__ left,
                                                     __gm__ T_RIGHT *__restrict__ right, __gm__
                                                     T_OUTPUT *__restrict__ out,
                                                     int64_t round_id,
                                                     PHY_ADDR_FORWARD_CUBE1<T_LEFT, T_RIGHT, T_OUTPUT> *src,
                                                     PHY_ADDR_FORWARD_CUBE1<T_LEFT, T_RIGHT, T_OUTPUT> *remain)
        {
            if (this->isTriangle) {
                addrMapping_cube1_mask(left, right, out, round_id, src, remain);
            } else {
                addrMapping_cube1_nomask(left, right, out, round_id, src, remain);
            }
        }

        template<typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
        __aicore__ __inline__ void addrMapping_cube1_nomask(__gm__ T_LEFT *__restrict__ left,
                                                            __gm__ T_RIGHT *__restrict__ right, __gm__
                                                            T_OUTPUT *__restrict__ out,
                                                            int64_t round_id,
                                                            PHY_ADDR_FORWARD_CUBE1<T_LEFT, T_RIGHT, T_OUTPUT> *src,
                                                            PHY_ADDR_FORWARD_CUBE1<T_LEFT, T_RIGHT, T_OUTPUT> *remain)
            {
            // 当前的row
            int64_t curCalcRow =
                    round_id * this->coreGroupNum + this->curGroupIndex;
            // workspace
            int64_t curCalcRowP =
                    this->coreGroupNum * (round_id % 2) + this->curGroupIndex;
            // 当前行所在的batch号
            int64_t b = curCalcRow /
                        this->forwardBlockRowsPerBatch;
            // 当前batch下的head号
            int64_t n = curCalcRow % this->forwardBlockRowsPerBatch /
                        this->forwardBlockNumPerCol;
            // 当前head下的行号
            int64_t i_r = curCalcRow % this->forwardBlockRowsPerBatch %
                          this->forwardBlockNumPerCol;
            // 当前core处理的起始列
            int64_t i_c =
                    this->groupLocalIndex *
                    this->blockNumPerCoreForward;
            // 尾块的起始列
            int64_t begin_remain =
                    this->forwardBlockNumPerRow - this->remainBlockNumForward;

            // batch head_num的偏移量
            int64_t left_bn_offset = (b * this->headNum + n) * this->querySequenceLen * this->headDim;
            int64_t g = n / (this->headNum / this->gqaGroupNum);
            int64_t right_bn_offset_GQA =
                    (b * this->gqaGroupNum + g) * this->keyValueSequenceLen * SIZE_256;
            int64_t out_bn_offset = (curCalcRowP * this->forwardBlockNumPerRow) * ATTENTION_SCORE_BLOCK_SIZE;

            src[0].left = left + left_bn_offset + i_r * QUERY_BLOCK_SIZE;
            src[0].right = right + right_bn_offset_GQA + i_c * KEY_BLOCK_SIZE;
            src[0].out = out + out_bn_offset + i_c * ATTENTION_SCORE_BLOCK_SIZE;
            src[0].k = this->blockNumPerCoreForward;

            if (this->remainBlockNumForward != 0) {
                remain[0].left = src[0].left;
                remain[0].right = right + right_bn_offset_GQA + begin_remain * KEY_BLOCK_SIZE;
                remain[0].out = out + out_bn_offset + begin_remain * ATTENTION_SCORE_BLOCK_SIZE;
                remain[0].k = this->remainBlockNumForward;
            }
        }

        template<typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
        __aicore__ __inline__ void addrMapping_cube1_mask(__gm__ T_LEFT *__restrict__ left,
                                                          __gm__ T_RIGHT *__restrict__ right, __gm__
                                                          T_OUTPUT *__restrict__ out,
                                                          int64_t round_id,
                                                          PHY_ADDR_FORWARD_CUBE1<T_LEFT, T_RIGHT, T_OUTPUT> *src,
                                                          PHY_ADDR_FORWARD_CUBE1<T_LEFT, T_RIGHT, T_OUTPUT> *remain)
        {
            // 当前的row
            int64_t curCalcRow =
                    round_id * this->coreGroupNum + this->curGroupIndex;
            // workspace
            int64_t curCalcRowP =
                    this->coreGroupNum * (round_id % 2) + this->curGroupIndex;
            // 当前行所在的batch号
            int64_t b = curCalcRow /
                        this->forwardBlockRowsPerBatch;
            // 当前batch下的head号
            int64_t n = curCalcRow % this->forwardBlockRowsPerBatch /
                        this->forwardBlockNumPerCol;
            // 当前head下的行号
            int64_t i_r = curCalcRow % this->forwardBlockRowsPerBatch %
                          this->forwardBlockNumPerCol;
            // 当前core处理的起始列
            int64_t i_c =
                    this->groupLocalIndex *
                    this->blockNumPerCoreForward;
            // 尾块的起始列
            int64_t begin_remain =
                    this->forwardBlockNumPerRow - this->remainBlockNumForward;

            // 跳变点的位置
            int64_t q_left_index = this->forwardBlockNumPerCol + i_r;
            int64_t q_right_index = this->forwardBlockNumPerCol - 1 - i_r;
            int64_t switch_index = q_left_index;

            // batch、head_num的偏移量
            int64_t left_bn_offset = (b * this->headNum + n) * this->querySequenceLen * this->headDim;
            int64_t g = n / (this->headNum / this->gqaGroupNum);
            int64_t right_bn_offset_GQA =
                    (b * this->gqaGroupNum + g) * this->keyValueSequenceLen * SIZE_256;
            int64_t out_bn_offset = curCalcRowP * this->forwardBlockNumPerRow * ATTENTION_SCORE_BLOCK_SIZE;

            if (switch_index < i_c) {
                src[0].left = left + left_bn_offset + q_right_index * QUERY_BLOCK_SIZE;
                src[0].right = right + right_bn_offset_GQA + (i_c - switch_index - 1) * KEY_BLOCK_SIZE;
                src[0].out = out + out_bn_offset + i_c * ATTENTION_SCORE_BLOCK_SIZE;
                src[0].k = this->blockNumPerCoreForward;

                if (this->remainBlockNumForward != 0) {
                    remain[0].left = src[0].left;
                    remain[0].right =
                            right + right_bn_offset_GQA + (begin_remain - switch_index - 1) * KEY_BLOCK_SIZE;
                    remain[0].out = out + out_bn_offset + begin_remain * ATTENTION_SCORE_BLOCK_SIZE;
                    remain[0].k = this->remainBlockNumForward;
                }
            } else if (i_c <= switch_index && i_c + this->blockNumPerCoreForward > switch_index + 1) {
                src[0].left = left + left_bn_offset + q_left_index * QUERY_BLOCK_SIZE;
                src[0].right = right + right_bn_offset_GQA + i_c * KEY_BLOCK_SIZE;
                src[0].out = out + out_bn_offset + i_c * ATTENTION_SCORE_BLOCK_SIZE;
                src[0].k = switch_index - i_c + 1;

                src[1].left = left + left_bn_offset + q_right_index * QUERY_BLOCK_SIZE;
                src[1].right = right + right_bn_offset_GQA;
                src[1].out = src[0].out + src[0].k * ATTENTION_SCORE_BLOCK_SIZE;
                src[1].k = this->blockNumPerCoreForward - src[0].k;

                if (this->remainBlockNumForward != 0) {
                    remain[0].left = src[1].left;
                    remain[0].right =
                            right + right_bn_offset_GQA + (begin_remain - switch_index - 1) * KEY_BLOCK_SIZE;
                    remain[0].out = out + out_bn_offset + begin_remain * ATTENTION_SCORE_BLOCK_SIZE;
                    remain[0].k = this->remainBlockNumForward;
                }
            } else {
                src[0].left = left + left_bn_offset + q_left_index * QUERY_BLOCK_SIZE;
                src[0].right = right + right_bn_offset_GQA + i_c * KEY_BLOCK_SIZE;
                src[0].out = out + out_bn_offset + i_c * ATTENTION_SCORE_BLOCK_SIZE;
                src[0].k = this->blockNumPerCoreForward;

                if (this->remainBlockNumForward != 0) {
                    if (switch_index < begin_remain) {
                        remain[0].left = left + left_bn_offset + q_right_index * QUERY_BLOCK_SIZE;
                        remain[0].right =
                                right + right_bn_offset_GQA + (begin_remain - switch_index - 1) * KEY_BLOCK_SIZE;
                        remain[0].out = out + out_bn_offset + begin_remain * ATTENTION_SCORE_BLOCK_SIZE;
                        remain[0].k = this->remainBlockNumForward;
                    } else {
                        remain[0].left = left + left_bn_offset + q_left_index * QUERY_BLOCK_SIZE;
                        remain[0].right = right + right_bn_offset_GQA + begin_remain * KEY_BLOCK_SIZE;
                        remain[0].out = out + out_bn_offset + begin_remain * ATTENTION_SCORE_BLOCK_SIZE;
                        remain[0].k = switch_index - begin_remain + 1;

                        remain[1].left = left + left_bn_offset + q_right_index * QUERY_BLOCK_SIZE;
                        remain[1].right = right + right_bn_offset_GQA;
                        remain[1].out = remain[0].out + remain[0].k * ATTENTION_SCORE_BLOCK_SIZE;
                        remain[1].k = this->remainBlockNumForward - remain[0].k;
                    }
                }
            }
        }

        template<typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
        __aicore__ __inline__ void addrMapping_cube2_rowsum(
                __gm__ T_LEFT *__restrict__ left,
                __gm__ T_RIGHT *__restrict__ right, __gm__
                T_OUTPUT *__restrict__ out,
                __gm__ T_OUTPUT *__restrict__ rowsumOut,
                int64_t round_id, int64_t &src_length,
                int64_t &remain_length,
                PHY_ADDR_FORWARD_ROWSUM_CUBE2<T_LEFT, T_RIGHT, T_OUTPUT> *src,
                PHY_ADDR_FORWARD_ROWSUM_CUBE2<T_LEFT, T_RIGHT, T_OUTPUT> *remain)
        {
            if (this->isTriangle) {
                addrMapping_cube2_rowsum_mask(left, right, out, rowsumOut, round_id, src_length, remain_length, src,
                                              remain);
            } else {
                addrMapping_cube2_rowsum_nomask(left, right, out, rowsumOut, round_id, src_length, remain_length, src,
                                                remain);
            }
        }

        template<typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
        __aicore__ __inline__ void addrMapping_cube2_rowsum_nomask(
                __gm__ T_LEFT *__restrict__ left,
                __gm__ T_RIGHT *__restrict__ right, __gm__
                T_OUTPUT *__restrict__ out,
                __gm__ T_OUTPUT *__restrict__ rowsumOut,
                int64_t round_id, int64_t &src_length,
                int64_t &remain_length,
                PHY_ADDR_FORWARD_ROWSUM_CUBE2<T_LEFT, T_RIGHT, T_OUTPUT> *src,
                PHY_ADDR_FORWARD_ROWSUM_CUBE2<T_LEFT, T_RIGHT, T_OUTPUT> *remain)
        {
            // 当前的row
            int64_t curCalcRow =
                    round_id * this->coreGroupNum + this->curGroupIndex;
            // workspace
            int64_t curCalcRowP =
                    this->coreGroupNum * (round_id % 2) + this->curGroupIndex;
            // 当前行所在的batch号
            int64_t b = curCalcRow /
                        this->forwardBlockRowsPerBatch;
            // 当前batch下的head号
            int64_t n = curCalcRow % this->forwardBlockRowsPerBatch /
                        this->forwardBlockNumPerCol;
            // 当前head下的行号
            int64_t i_r = curCalcRow % this->forwardBlockRowsPerBatch %
                          this->forwardBlockNumPerCol;
            // 当前core处理的起始列
            int64_t i_c =
                    this->groupLocalIndex *
                    this->blockNumPerCoreForward;
            // 尾块的起始列
            int64_t begin_remain =
                    this->forwardBlockNumPerRow - this->remainBlockNumForward;

            // batch、head_num的偏移量
            int64_t g = n / (this->headNum / this->gqaGroupNum);
            int64_t left_bn_out_offset =
                    (curCalcRowP * this->forwardBlockNumPerRow) * ATTENTION_SCORE_BLOCK_SIZE;
            int64_t right_bn_offset_GQA = (b * this->gqaGroupNum + g) * keyValueSequenceLen * SIZE_128;
            int64_t cube2_out_bn_offset = (b * this->headNum + n) * outputSequenceLen * SIZE_128;
            int64_t rowsum_out_bn_offset = (b * this->headNum + n) * rowSumSequenceLen;

            src[0].left = left + left_bn_out_offset + i_c * ATTENTION_SCORE_BLOCK_SIZE;
            src[0].right = right + right_bn_offset_GQA + i_c * VALUE_BLOCK_SIZE;
            src[0].out = out + cube2_out_bn_offset + i_r * OUTPUT_BLOCK_SIZE;
            src[0].rowsumOut = rowsumOut + rowsum_out_bn_offset + i_r * ROWSUM_BLOCK_SIZE;
            src[0].k = this->blockNumPerCoreForward;
            src_length = 1;
            if (this->remainBlockNumForward != 0) {
                remain[0].left = left + left_bn_out_offset + begin_remain * ATTENTION_SCORE_BLOCK_SIZE +
                                 (BASE_BLOCK_LENGTH / this->coreNumPerGroup) * this->groupLocalIndex *
                                 BASE_BLOCK_LENGTH;
                remain[0].right = right + right_bn_offset_GQA + begin_remain * VALUE_BLOCK_SIZE;
                remain[0].out = out + cube2_out_bn_offset + i_r * OUTPUT_BLOCK_SIZE +
                                (BASE_BLOCK_LENGTH / this->coreNumPerGroup) * this->groupLocalIndex *
                                BASE_BLOCK_LENGTH;
                remain[0].rowsumOut =
                        rowsumOut + rowsum_out_bn_offset + i_r * ROWSUM_BLOCK_SIZE +
                        (BASE_BLOCK_LENGTH / this->coreGroupNum) * this->groupLocalIndex;
                remain[0].k = this->remainBlockNumForward;
                remain_length = 1;
            }
        }

        template<typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
        __aicore__ __inline__ void addrMapping_cube2_rowsum_mask(
            __gm__ T_LEFT *__restrict__ left,
            __gm__ T_RIGHT *__restrict__ right,
            __gm__ T_OUTPUT *__restrict__ out,
            __gm__ T_OUTPUT *__restrict__ rowsumOut,
            int64_t round_id, int64_t &src_length,
            int64_t &remain_length,
            PHY_ADDR_FORWARD_ROWSUM_CUBE2<T_LEFT, T_RIGHT, T_OUTPUT> *src,
            PHY_ADDR_FORWARD_ROWSUM_CUBE2<T_LEFT, T_RIGHT, T_OUTPUT> *remain) {
            // 当前的row
            int64_t curCalcRow =
                    round_id * this->coreGroupNum + this->curGroupIndex;
            // workspace
            int64_t curCalcRowP =
                    this->coreGroupNum * (round_id % 2) + this->curGroupIndex;
            // 当前行所在的batch号
            int64_t b = curCalcRow /
                        this->forwardBlockRowsPerBatch;
            // 当前batch下的head号
            int64_t n = curCalcRow % this->forwardBlockRowsPerBatch /
                        this->forwardBlockNumPerCol;
            // 当前head下的行号
            int64_t i_r = curCalcRow % this->forwardBlockRowsPerBatch %
                          this->forwardBlockNumPerCol;
            // 当前core处理的起始列
            int64_t i_c =
                    this->groupLocalIndex *
                    this->blockNumPerCoreForward;
            // 尾块的起始列
            int64_t begin_remain =
                    this->forwardBlockNumPerRow - this->remainBlockNumForward;

            // 跳变点的位置
            int64_t q_left_index = this->forwardBlockNumPerCol + i_r;
            int64_t q_right_index = this->forwardBlockNumPerCol - 1 - i_r;
            int64_t switch_index = q_left_index;

            // batch、head_num的偏移量
            int64_t g = n / (this->headNum / this->gqaGroupNum);
            int64_t left_bn_out_offset =
                    (curCalcRowP * this->forwardBlockNumPerRow) * ATTENTION_SCORE_BLOCK_SIZE;
            int64_t right_bn_offset_GQA = (b * this->gqaGroupNum + g) * keyValueSequenceLen * SIZE_128;
            int64_t cube2_out_bn_offset = (b * this->headNum + n) * outputSequenceLen * SIZE_128;
            int64_t rowsum_out_bn_offset = (b * this->headNum + n) * rowSumSequenceLen;

            if (switch_index < i_c) {
                src[0].left = left + left_bn_out_offset + i_c * ATTENTION_SCORE_BLOCK_SIZE;
                src[0].right = right + right_bn_offset_GQA + (i_c - switch_index - 1) * VALUE_BLOCK_SIZE;
                src[0].out = out + cube2_out_bn_offset + q_right_index * OUTPUT_BLOCK_SIZE;
                src[0].rowsumOut = rowsumOut + rowsum_out_bn_offset + q_right_index * ROWSUM_BLOCK_SIZE;
                src[0].k = this->blockNumPerCoreForward;
                src_length = 1;

                if (this->remainBlockNumForward != 0) {
                    remain[0].left =
                            left + left_bn_out_offset + begin_remain * ATTENTION_SCORE_BLOCK_SIZE +
                            (BASE_BLOCK_LENGTH / this->coreNumPerGroup) * this->groupLocalIndex *
                            BASE_BLOCK_LENGTH;
                    remain[0].right =
                            right + right_bn_offset_GQA + (begin_remain - switch_index - 1) * VALUE_BLOCK_SIZE;
                    remain[0].out = src[0].out +
                                    (BASE_BLOCK_LENGTH / this->coreNumPerGroup) * this->groupLocalIndex *
                                    BASE_BLOCK_LENGTH;
                    remain[0].rowsumOut =
                            src[0].rowsumOut +
                            (BASE_BLOCK_LENGTH / this->coreNumPerGroup) * this->groupLocalIndex;
                    remain[0].k = this->remainBlockNumForward;
                    remain_length = 1;
                }
            } else if (i_c <= switch_index && i_c + this->blockNumPerCoreForward > switch_index + 1) {
                src[0].left = left + left_bn_out_offset + i_c * ATTENTION_SCORE_BLOCK_SIZE;
                src[0].right = right + right_bn_offset_GQA + i_c * VALUE_BLOCK_SIZE;
                src[0].out = out + cube2_out_bn_offset + q_left_index * OUTPUT_BLOCK_SIZE;
                src[0].rowsumOut = rowsumOut + rowsum_out_bn_offset + q_left_index * ROWSUM_BLOCK_SIZE;
                src[0].k = switch_index - i_c + 1;

                src[1].left = src[0].left +
                              src[0].k * ATTENTION_SCORE_BLOCK_SIZE;
                src[1].right = right + right_bn_offset_GQA;
                src[1].out = out + cube2_out_bn_offset + q_right_index * OUTPUT_BLOCK_SIZE;
                src[1].rowsumOut = rowsumOut + rowsum_out_bn_offset + q_right_index * ROWSUM_BLOCK_SIZE;
                src[1].k = this->blockNumPerCoreForward - src[0].k;

                src_length = 2;

                if (this->remainBlockNumForward != 0) {
                    remain[0].left =
                            left + left_bn_out_offset + begin_remain * ATTENTION_SCORE_BLOCK_SIZE +
                            (BASE_BLOCK_LENGTH / this->coreNumPerGroup) * this->groupLocalIndex *
                            BASE_BLOCK_LENGTH;
                    remain[0].right =
                            right + right_bn_offset_GQA + (begin_remain - switch_index - 1) * VALUE_BLOCK_SIZE;
                    remain[0].out =
                            src[1].out + (BASE_BLOCK_LENGTH / this->coreNumPerGroup) * this->groupLocalIndex *
                                         BASE_BLOCK_LENGTH;
                    remain[0].rowsumOut =
                            src[1].rowsumOut +
                            (BASE_BLOCK_LENGTH / this->coreNumPerGroup) * this->groupLocalIndex;
                    remain[0].k = this->remainBlockNumForward;
                    remain_length = 1;
                }
            } else {
                src[0].left = left + left_bn_out_offset + i_c * ATTENTION_SCORE_BLOCK_SIZE;
                src[0].right = right + right_bn_offset_GQA + i_c * VALUE_BLOCK_SIZE;
                src[0].out = out + cube2_out_bn_offset + q_left_index * OUTPUT_BLOCK_SIZE;
                src[0].rowsumOut = rowsumOut + rowsum_out_bn_offset + q_left_index * ROWSUM_BLOCK_SIZE;
                src[0].k = this->blockNumPerCoreForward;
                src_length = 1;

                if (this->remainBlockNumForward != 0) {
                    if (switch_index < begin_remain) {
                        remain[0].left =
                                left + left_bn_out_offset + begin_remain * ATTENTION_SCORE_BLOCK_SIZE +
                                (BASE_BLOCK_LENGTH / this->coreNumPerGroup) * this->groupLocalIndex *
                                BASE_BLOCK_LENGTH;
                        remain[0].right =
                                right + right_bn_offset_GQA + (begin_remain - switch_index - 1) * VALUE_BLOCK_SIZE;
                        remain[0].out =
                                out + cube2_out_bn_offset + q_right_index * OUTPUT_BLOCK_SIZE +
                                (BASE_BLOCK_LENGTH / this->coreNumPerGroup) * this->groupLocalIndex *
                                BASE_BLOCK_LENGTH;
                        remain[0].rowsumOut =
                                rowsumOut + rowsum_out_bn_offset + q_right_index * ROWSUM_BLOCK_SIZE +
                                (BASE_BLOCK_LENGTH / this->coreNumPerGroup) * this->groupLocalIndex;
                        remain[0].k = this->remainBlockNumForward;
                        remain_length = 1;
                    } else {
                        remain[0].left =
                                left + left_bn_out_offset + begin_remain * ATTENTION_SCORE_BLOCK_SIZE +
                                (BASE_BLOCK_LENGTH / this->coreNumPerGroup) * this->groupLocalIndex *
                                BASE_BLOCK_LENGTH;
                        remain[0].right = right + right_bn_offset_GQA + begin_remain * VALUE_BLOCK_SIZE;
                        remain[0].out =
                                out + cube2_out_bn_offset + q_left_index * OUTPUT_BLOCK_SIZE +
                                (BASE_BLOCK_LENGTH / this->coreNumPerGroup) * this->groupLocalIndex *
                                BASE_BLOCK_LENGTH;
                        remain[0].rowsumOut =
                                rowsumOut + rowsum_out_bn_offset + q_left_index * ROWSUM_BLOCK_SIZE +
                                (BASE_BLOCK_LENGTH / this->coreNumPerGroup) * this->groupLocalIndex;
                        remain[0].k = switch_index - begin_remain + 1;

                        remain[1].left = remain[0].left + remain[0].k * ATTENTION_SCORE_BLOCK_SIZE;
                        remain[1].right = right + right_bn_offset_GQA;
                        remain[1].out =
                                out + cube2_out_bn_offset + q_right_index * OUTPUT_BLOCK_SIZE +
                                (BASE_BLOCK_LENGTH / this->coreNumPerGroup) * this->groupLocalIndex *
                                BASE_BLOCK_LENGTH;
                        remain[1].rowsumOut =
                                rowsumOut + rowsum_out_bn_offset + q_right_index * BASE_BLOCK_LENGTH +
                                (BASE_BLOCK_LENGTH / this->coreNumPerGroup) * this->groupLocalIndex;
                        remain[1].k = this->remainBlockNumForward - remain[0].k;

                        remain_length = 2;
                    }
                }
            }
        }
    };
}
#endif
