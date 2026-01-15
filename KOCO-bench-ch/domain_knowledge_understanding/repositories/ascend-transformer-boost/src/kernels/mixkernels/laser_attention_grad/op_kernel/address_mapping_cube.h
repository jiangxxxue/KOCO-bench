/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef __ADDRESS_MAPPING_H__
#define __ADDRESS_MAPPING_H__

#include <cstdint>
#include "address_const.h"

namespace Address {
template <typename TYPE>
class AddressMappingCube {
public:
    __aicore__ __inline__ void BackWardAddrMapping_pre(BackWardAddr *addr, int64_t &addrLen, int64_t roundId)
    {
        // 当前核计算起始块的索引
        int32_t curBlockId = roundId * this->backward_core_num_ * this->backward_block_num_per_core_ +
                               this->backward_local_core_index_ * this->backward_block_num_per_core_;
        int32_t rowNumPerRound = this->backward_block_num_ky_;
        int32_t colNumPerRound = this->backward_block_num_per_core_ / this->backward_block_num_ky_;

        // 最后轮次的尾块处理：
        int32_t remain = this->backward_block_num_per_core_;
        if (curBlockId + this->backward_block_num_per_core_ > this->backward_total_blocks_) {
            remain = this->backward_total_blocks_ - curBlockId;
        }

        // 设置x，y方向的基本块数量
        int32_t Ky = this->backward_block_num_ky_;
        int32_t Kx = remain / Ky;

        // 当前轮次的(b,n,ir_begin,ic_begin)
        int32_t b = curBlockId / this->backward_block_num_per_batch_;
        int32_t n = curBlockId % this->backward_block_num_per_batch_ / this->backward_block_num_per_head_;
        int32_t blockRow =
            curBlockId % this->backward_block_num_per_head_ / (rowNumPerRound * this->backward_block_num_per_row_);
        int32_t ir = blockRow * rowNumPerRound;
        int32_t ic = (roundId * this->backward_core_num_ * colNumPerRound +
                         this->backward_local_core_index_ * colNumPerRound) %
                     (this->backward_block_num_per_batch_) % this->backward_block_num_per_row_;

        // 处理边界：
        addr[0].b = b;
        addr[0].n = n;
        addr[0].iR = ir;
        addr[0].iC = ic;
        addr[0].kx = Kx;
        addr[0].ky = Ky;
        addr[0].k = remain;

        int index = 0;
        for (; remain > 0;) {
            if (addr[index].iC + addr[index].kx > this->backward_block_num_per_row_) {  // 换行
                addr[index].kx = this->backward_block_num_per_row_ - addr[index].iC;
                addr[index].k = addr[index].kx * addr[index].ky;

                addr[index + 1].b = addr[index].b;
                addr[index + 1].n = addr[index].n;
                addr[index + 1].iR = addr[index].iR + addr[index].ky;
                addr[index + 1].iC = 0;
                addr[index + 1].k = remain - addr[index].k;
                addr[index + 1].ky = addr[index].ky;
                addr[index + 1].kx = addr[index + 1].k / addr[index + 1].ky;
                if (addr[index + 1].iR >= this->backward_block_num_per_column_) {  // 换head
                    addr[index + 1].n = addr[index].n + 1;
                    addr[index + 1].iR = addr[index + 1].iR % this->backward_block_num_per_column_;
                    if (addr[index + 1].n >= this->head_num_) {  // 换batch
                        addr[index + 1].b = addr[index].b + 1;
                        addr[index + 1].n = 0;
                    }
                }
            }

            remain -= addr[index].k;
            ++index;
        }
        addrLen = index;
    }

    template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
    __aicore__ __inline__ void addrMapping_rectangular_cube1(__gm__ T_LEFT *__restrict__ left,
        __gm__ T_RIGHT *__restrict__ right, __gm__ T_OUTPUT *__restrict__ out, const BackWardAddr *addr,
        PhyAddrBackwardCube1<T_LEFT, T_RIGHT, T_OUTPUT> *src,
        int64_t srcLen, int64_t roundId, int64_t dim_size_q, int64_t dim_size_k)
    {
        // 开启work space
        auto outOffsetRoundEven =
            out + this->backward_local_core_index_ * this->backward_block_num_per_core_ * ATTENTION_SCORE_BLOCK_SIZE;
        auto outOffsetRoundOdd =
            out + this->backward_core_num_ * this->backward_block_num_per_core_ * ATTENTION_SCORE_BLOCK_SIZE +
            this->backward_local_core_index_ * this->backward_block_num_per_core_ * ATTENTION_SCORE_BLOCK_SIZE;

        for (int32_t i = 0; i < srcLen; ++i) {
            auto b = addr[i].b;
            auto n = addr[i].n;
            auto ir = addr[i].iR;
            auto ic = addr[i].iC;
            auto Kx = addr[i].kx;
            auto Ky = addr[i].ky;
            auto k = addr[i].k;

            // 设置偏移量
            auto bnLeftOffset = left + (b * this->head_num_ + n) * this->query_sequence_len_ * dim_size_q;
            auto bnRightOffset = right + (b * this->head_num_ + n) * this->key_value_sequence_len_ * dim_size_k;
            int64_t gIndex = n / (this->head_num_ / this->gqa_group_num_);
            auto bnRightOffsetGqa =
                right + (b * this->gqa_group_num_ + gIndex) * this->key_value_sequence_len_ * dim_size_k;
            src[i].left = bnLeftOffset + ir * SIZE_128 * dim_size_q;
            src[i].right = bnRightOffsetGqa + ic * SIZE_128 * dim_size_k;
            src[i].out = ((roundId + 1) % 2) ? outOffsetRoundEven : outOffsetRoundOdd;
            src[i].kx = Kx;
            src[i].ky = Ky;
            src[i].k = k;
            src[i].lineStride = Kx * ATTENTION_SCORE_BLOCK_SIZE;
            src[i].lowerLeft = false;
            src[i].upperRight = false;

            // 多段时，workspoce的偏移量更新
            outOffsetRoundEven += k * ATTENTION_SCORE_BLOCK_SIZE;
            outOffsetRoundOdd += k * ATTENTION_SCORE_BLOCK_SIZE;
        }
    }

    template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
    __aicore__ __inline__ void addrMapping_triangular_cube1(__gm__ T_LEFT *__restrict__ left,
        __gm__ T_RIGHT *__restrict__ right, __gm__ T_OUTPUT *__restrict__ out, const BackWardAddr *addr,
        PhyAddrBackwardCube1<T_LEFT, T_RIGHT, T_OUTPUT> *src,
        int64_t &srcLen, int64_t roundId, int64_t dim_size_q, int64_t dim_size_k)
    {
        // 开启work space

        int64_t index = 0;
        int64_t triBlockNumPerColumn = this->backward_block_num_per_column_ - 2 * this->is_odd_;
        for (int32_t i = 0; i < srcLen; ++i) {
            int64_t iR = addr[i].iR;
            int64_t iC = addr[i].iC;
            int64_t kx = addr[i].kx;
            int64_t k = addr[i].k;

            // 倒三角和非倒三角这三个变量不一样

            int64_t switchIndex = triBlockNumPerColumn + iR + (iR + 1) % 2;
            int64_t rowOffset = (iR + 1) % 2 == 1 ? -1 : 1;
            int64_t rowIndexLeftSection = triBlockNumPerColumn + iR;  // 倒三角 ：非倒三角
            int64_t rowIndexRightSection = triBlockNumPerColumn - 1 - iR + rowOffset;

            int64_t colIndexLeftSection = iC;
            int64_t colIndexRightSection = iC - switchIndex - 1;

            // GQA设置：在cube1中K（右矩阵）进行修改
            int64_t gIndex = addr[i].n / (this->head_num_ / this->gqa_group_num_);
            int64_t bnOffsetGqaRightMatrix = (addr[i].b * this->gqa_group_num_ + gIndex) *
                                                 this->key_value_sequence_len_ * dim_size_k;  // for gqa mode
            int64_t bnOffsetRightMatrix =
                (addr[i].b * this->head_num_ + addr[i].n) * this->key_value_sequence_len_ * dim_size_k;
            int64_t bnOffsetLeftMatrix =
                (addr[i].b * this->head_num_ + addr[i].n) * this->query_sequence_len_ * dim_size_q;

            int64_t qLeftOffsetSection = rowIndexLeftSection * SIZE_128 * dim_size_q;
            int64_t qRightOffsetSection = rowIndexRightSection * SIZE_128 * dim_size_q;
            int64_t kLeftOffsetSection = colIndexLeftSection * SIZE_128 * dim_size_k;
            int64_t kRightOffsetSection = colIndexRightSection * SIZE_128 * dim_size_k;

            // sparse场景 isTri == true 以及 sparseMode == 1
            bool sparseFlag = false;
            int64_t windowBlockSize = this->window_length_ / 128;
            if (this->is_triangle_ && this->sparse_mode_ == 1) {
                sparseFlag = iR > ((windowBlockSize - 1) / 2) ? true : false;
                switchIndex = (windowBlockSize / 2) + iR;
                rowIndexLeftSection = (windowBlockSize / 2) + iR;
                rowIndexRightSection = (windowBlockSize / 2) - 1 - iR;
                colIndexLeftSection = iC;
                colIndexRightSection = iC - switchIndex - 1;
                qLeftOffsetSection = rowIndexLeftSection * ATTENTION_SCORE_BLOCK_SIZE;
                qRightOffsetSection = rowIndexRightSection * ATTENTION_SCORE_BLOCK_SIZE;
                kLeftOffsetSection = colIndexLeftSection * ATTENTION_SCORE_BLOCK_SIZE;
                kRightOffsetSection = colIndexRightSection * ATTENTION_SCORE_BLOCK_SIZE;
            }
            int64_t rowIndexSparseSection = iR + (windowBlockSize / 2);
            int64_t colIndexSparseSection = iR + iC - (windowBlockSize / 2);
            int64_t qSparseOffsetSection = rowIndexSparseSection * ATTENTION_SCORE_BLOCK_SIZE;
            int64_t kSparseOffsetSection = colIndexSparseSection * ATTENTION_SCORE_BLOCK_SIZE;

            int64_t outOffset = ((addr[i].b * this->head_num_ + addr[i].n) * this->backward_block_num_per_row_ *
                                         this->backward_block_num_per_column_ +
                                     (iR * this->backward_block_num_per_row_)) *
                                 ATTENTION_SCORE_BLOCK_SIZE;

            int64_t dbOffset = (roundId % 2) * (this->backward_core_num_ * this->backward_block_num_per_core_ *
                                                     ATTENTION_SCORE_BLOCK_SIZE);
            if (index == 0) {
                src[index].out =
                    out + dbOffset +
                    this->backward_local_core_index_ * this->backward_block_num_per_core_ * ATTENTION_SCORE_BLOCK_SIZE;
            } else {
                src[index].out = src[index - 1].out + src[index - 1].k * ATTENTION_SCORE_BLOCK_SIZE;
            }

            if (!sparseFlag && switchIndex < iC) {
                src[index].left = left + bnOffsetLeftMatrix + qRightOffsetSection;
                src[index].right = right + bnOffsetGqaRightMatrix + kRightOffsetSection;
                // src[index].out = gm_c + outOffset + iC * BASE_BLOCK_SIZE; // 使用 workspace double buffer 后删除
                src[index].kx = kx;
                src[index].ky = this->backward_block_num_ky_;
                src[index].k = k;
                src[index].upperRight = iC + src[index].kx >= backward_block_num_per_row_ ? true : false;
                src[index].lowerLeft = false;
                src[index].lineStride = src[index].kx * ATTENTION_SCORE_BLOCK_SIZE;
                index++;
            } else if (!sparseFlag && iC <= switchIndex && iC + kx - 1 > switchIndex) {
                src[index].left = left + bnOffsetLeftMatrix + qLeftOffsetSection;
                src[index].right = right + bnOffsetGqaRightMatrix + kLeftOffsetSection;
                // src[index].out = gm_c + outOffset + iC * BASE_BLOCK_SIZE;  // 使用 workspace double buffer 后删除
                src[index].kx = switchIndex - iC + 1;
                src[index].ky = this->backward_block_num_ky_;
                src[index].k = src[index].kx * this->backward_block_num_ky_;
                src[index].upperRight = true;
                src[index].lowerLeft = false;
                src[index].lineStride = src[index].kx * ATTENTION_SCORE_BLOCK_SIZE;
                src[index + 1].left = left + bnOffsetLeftMatrix + qRightOffsetSection;
                src[index + 1].right = right + bnOffsetGqaRightMatrix;
                src[index + 1].out = src[index].out + src[index].k * ATTENTION_SCORE_BLOCK_SIZE;  // 保留
                src[index + 1].kx = kx - src[index].kx;
                src[index + 1].ky = this->backward_block_num_ky_;
                src[index + 1].k = src[index + 1].kx * this->backward_block_num_ky_;
                src[index + 1].upperRight =
                    switchIndex + src[index + 1].kx >= backward_block_num_per_row_ - 1 ? true : false;
                src[index + 1].lowerLeft = false;
                src[index + 1].lineStride = src[index + 1].kx * ATTENTION_SCORE_BLOCK_SIZE;
                index += 2;
            } else if (!sparseFlag && iC <= switchIndex && iC + kx - 1 <= switchIndex) {
                src[index].left = left + bnOffsetLeftMatrix + qLeftOffsetSection;
                src[index].right = right + bnOffsetGqaRightMatrix + kLeftOffsetSection;
                src[index].kx = kx;
                src[index].ky = this->backward_block_num_ky_;
                src[index].k = k;
                src[index].upperRight = iC + src[index].kx - 1 >= switchIndex ? true : false;
                src[index].lowerLeft = false;
                src[index].lineStride = src[index].kx * ATTENTION_SCORE_BLOCK_SIZE;
                index++;
            } else {  // not finished
                src[index].left = left + bnOffsetLeftMatrix + qSparseOffsetSection;
                src[index].right = right + bnOffsetGqaRightMatrix + kSparseOffsetSection;
                src[index].k = k;
                index++;
            }
        }
        srcLen = index;
    }

    template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
    __aicore__ __inline__ void addrMapping_rectangular_cube2(__gm__ T_LEFT *__restrict__ left,
        __gm__ T_RIGHT *__restrict__ right, __gm__ T_OUTPUT *__restrict__ out, const BackWardAddr *addr,
        PhyAddrBackwardCube2<T_LEFT, T_RIGHT, T_OUTPUT> *src,
        int64_t srcLen, int32_t roundId, int64_t dim_size_q, int64_t dim_size_k)
    {
        // 开启work space
        auto leftOffsetRoundEven =
            left + this->backward_local_core_index_ * this->backward_block_num_per_core_ * ATTENTION_SCORE_BLOCK_SIZE;
        auto leftOffsetRoundOdd =
            left + this->backward_core_num_ * this->backward_block_num_per_core_ * ATTENTION_SCORE_BLOCK_SIZE +
            this->backward_local_core_index_ * this->backward_block_num_per_core_ * ATTENTION_SCORE_BLOCK_SIZE;

        for (int32_t i = 0; i < srcLen; ++i) {
            auto b = addr[i].b;
            auto n = addr[i].n;
            auto ir = addr[i].iR;
            auto ic = addr[i].iC;
            auto Kx = addr[i].kx;
            auto Ky = addr[i].ky;
            auto k = addr[i].k;

            // 设置偏移量
            auto bnRightOffset = right + (b * this->head_num_ + n) * this->key_value_sequence_len_ * dim_size_k;
            int64_t gIndex = n / (this->head_num_ / this->gqa_group_num_);
            auto bnRightOffsetGqa =
                right + (b * this->gqa_group_num_ + gIndex) * this->key_value_sequence_len_ * dim_size_k;
            auto bnOutOffset = out + (b * this->head_num_ + n) * this->query_sequence_len_ * dim_size_q;

            src[i].left = ((roundId + 1) % 2) ? leftOffsetRoundEven : leftOffsetRoundOdd;
            src[i].right = bnRightOffsetGqa + ic * SIZE_128 * dim_size_k;
            src[i].out = bnOutOffset + ir * SIZE_128 * dim_size_q;
            src[i].kx = Kx;
            src[i].ky = Ky;
            src[i].k = k;
            src[i].lineStride = Kx * ATTENTION_SCORE_BLOCK_SIZE;
            src[i].lowerLeft = false;
            src[i].upperRight = false;

            // 多段时，work space偏移量更新
            leftOffsetRoundEven += k * ATTENTION_SCORE_BLOCK_SIZE;
            leftOffsetRoundOdd += k * ATTENTION_SCORE_BLOCK_SIZE;
        }
    }

    template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
    __aicore__ __inline__ void addrMapping_triangular_cube2(__gm__ T_LEFT *__restrict__ left,
        __gm__ T_RIGHT *__restrict__ right, __gm__ T_OUTPUT *__restrict__ out, const BackWardAddr *addr,
        PhyAddrBackwardCube2<T_LEFT, T_RIGHT, T_OUTPUT> *src,
        int64_t &srcLen, int32_t roundId, int64_t dim_size_q, int64_t dim_size_k)
    {
        // 负载均衡的地址偏移
        int64_t index = 0;
        int64_t triBlocksPerColumn = this->backward_block_num_per_column_ - 2 * this->is_odd_;
        for (int i = 0; i < srcLen; ++i) {
            // left、right、out的地址配置
            int64_t iR = addr[i].iR;
            int64_t iC = addr[i].iC;
            int64_t kx = addr[i].kx;
            int64_t k = addr[i].k;
            // 倒三角和非倒三角这三个变量不一样

            int64_t switchIndex = triBlocksPerColumn + iR + (iR + 1) % 2;
            int64_t rowIndexLeftSection = triBlocksPerColumn + iR;  // 倒三角 ：非倒三角
            int64_t rowOffset = (iR + 1) % 2 == 1 ? -1 : 1;
            int64_t rowIndexRightSection = triBlocksPerColumn - 1 - iR + rowOffset;
            int64_t colIndexLeftSection = iC;
            int64_t colIndexRightSection = iC - switchIndex - 1;
            // GQA设置：在cube1中K（右矩阵）进行修改
            int64_t gIndex = addr[i].n / (this->head_num_ / this->gqa_group_num_);
            int64_t bnOffsetGqaRightMatrix =
                (addr[i].b * this->gqa_group_num_ + gIndex) * this->key_value_sequence_len_ * dim_size_k;
            int64_t bnOffsetLeftMatrix =
                ((addr[i].b * this->head_num_ + addr[i].n) * this->backward_block_num_per_row_ *
                    this->backward_block_num_per_column_) *
                ATTENTION_SCORE_BLOCK_SIZE;  // 当前b,n下的偏移量
            int64_t bnOffsetRightMatrix = (addr[i].b * this->head_num_ + addr[i].n) * this->key_value_sequence_len_ *
                                             dim_size_k;  // 当前b,n下的偏移量
            int64_t bn_offset_out = (addr[i].b * this->head_num_ + addr[i].n) * this->query_sequence_len_ *
                                    dim_size_q;  // 当前b,n下的偏移量

            // sparse场景：is_tri == true 以及 sparseMode == 1
            bool sparseFlag = false;
            int64_t windowBlockSize = this->window_length_ / 128;
            if (this->is_triangle_ && this->sparse_mode_ == 1) {
                sparseFlag = iR > ((windowBlockSize - 1) / 2) ? true : false;
                switchIndex = (windowBlockSize / 2) + iR;
                rowIndexLeftSection = (windowBlockSize / 2) + iR;
                rowIndexRightSection = (windowBlockSize / 2) - 1 - iR;
                colIndexLeftSection = iC;
                colIndexRightSection = iC - switchIndex - 1;
            }
            int64_t rowIndexSparseSection = iR + (windowBlockSize / 2);
            int64_t colIndexSparseSection = iR + iC - (windowBlockSize / 2);

            // 开启double buffer
            int64_t dbOffset = (roundId % 2) * (this->backward_core_num_ * this->backward_block_num_per_core_ *
                                                     ATTENTION_SCORE_BLOCK_SIZE);
            if (index == 0) {
                src[index].left =
                    left + dbOffset +
                    this->backward_local_core_index_ * this->backward_block_num_per_core_ * ATTENTION_SCORE_BLOCK_SIZE;
            } else {
                src[index].left = src[index - 1].left + src[index - 1].k * ATTENTION_SCORE_BLOCK_SIZE;
            }

            if (!sparseFlag && switchIndex < iC) {
                src[index].right =
                    right + bnOffsetGqaRightMatrix + colIndexRightSection * SIZE_128 * dim_size_k;
                src[index].out = out + bn_offset_out + rowIndexRightSection * SIZE_128 * dim_size_q;
                src[index].kx = kx;
                src[index].ky = this->backward_block_num_ky_;
                src[index].k = k;
                src[index].upperRight = iC + src[index].kx >= backward_block_num_per_row_ ? true : false;
                src[index].lowerLeft = false;
                src[index].lineStride = src[index].kx * ATTENTION_SCORE_BLOCK_SIZE;
                ++index;
            } else if (!sparseFlag && iC <= switchIndex && iC + kx - 1 > switchIndex) {
                // src[index].left = left + bnLeftOffset + iR * (H + 1) * BASE_BLOCK_SIZE + iC * BASE_BLOCK_SIZE;
                src[index].right =
                    right + bnOffsetGqaRightMatrix + colIndexLeftSection * SIZE_128 * dim_size_k;
                src[index].out = out + bn_offset_out + rowIndexLeftSection * SIZE_128 * dim_size_q;
                src[index].kx = switchIndex - iC + 1;
                src[index].ky = this->backward_block_num_ky_;
                src[index].k = src[index].kx * this->backward_block_num_ky_;
                src[index].upperRight = true;
                src[index].lowerLeft = false;
                src[index].lineStride = src[index].kx * ATTENTION_SCORE_BLOCK_SIZE;
                src[index + 1].left = src[index].left + src[index].k * ATTENTION_SCORE_BLOCK_SIZE;
                src[index + 1].right = right + bnOffsetGqaRightMatrix;
                src[index + 1].out = out + bn_offset_out + rowIndexRightSection * SIZE_128 * dim_size_q;
                src[index + 1].kx = kx - src[index].kx;
                src[index + 1].ky = this->backward_block_num_ky_;
                src[index + 1].k = src[index + 1].kx * this->backward_block_num_ky_;
                src[index + 1].upperRight =
                    switchIndex + src[index + 1].kx >= backward_block_num_per_row_ - 1 ? true : false;
                src[index + 1].lowerLeft = false;
                src[index + 1].lineStride = src[index + 1].kx * ATTENTION_SCORE_BLOCK_SIZE;
                index += 2;
            } else if (!sparseFlag && iC <= switchIndex && iC + kx - 1 <= switchIndex) {
                src[index].right =
                    right + bnOffsetGqaRightMatrix + colIndexLeftSection * SIZE_128 * dim_size_k;
                src[index].out = out + bn_offset_out + rowIndexLeftSection * SIZE_128 * dim_size_q;
                src[index].kx = kx;
                src[index].ky = this->backward_block_num_ky_;
                src[index].k = k;
                src[index].upperRight = iC + src[index].kx - 1 >= switchIndex ? true : false;
                src[index].lowerLeft = false;
                src[index].lineStride = src[index].kx * ATTENTION_SCORE_BLOCK_SIZE;
                index++;
            } else {
                src[index].right =
                    right + bnOffsetGqaRightMatrix + colIndexSparseSection * ATTENTION_SCORE_BLOCK_SIZE;
                src[index].out = out + bn_offset_out + rowIndexSparseSection * ATTENTION_SCORE_BLOCK_SIZE;
                src[index].k = k;
                ++index;
            }
        }
        srcLen = index;
    }

    template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
    __aicore__ __inline__ void addrMapping_rectangular_cube3(__gm__ T_LEFT *__restrict__ left,
        __gm__ T_RIGHT *__restrict__ right, __gm__ T_OUTPUT *__restrict__ out, const BackWardAddr *addr,
        PhyAddrBackwardCube3<T_LEFT, T_RIGHT, T_OUTPUT> *src,
        int64_t &srcLen, int32_t roundId, int64_t dim_size_q, int64_t dim_size_k)
    {
        // 开启work space
        auto leftOffsetRoundEven =
            left + this->backward_local_core_index_ * this->backward_block_num_per_core_ * ATTENTION_SCORE_BLOCK_SIZE;
        auto leftOffsetRoundOdd =
            left + this->backward_core_num_ * this->backward_block_num_per_core_ * ATTENTION_SCORE_BLOCK_SIZE +
            this->backward_local_core_index_ * this->backward_block_num_per_core_ * ATTENTION_SCORE_BLOCK_SIZE;

        for (int32_t i = 0; i < srcLen; ++i) {
            auto b = addr[i].b;
            auto n = addr[i].n;
            auto ir = addr[i].iR;
            auto ic = addr[i].iC;
            auto Kx = addr[i].kx;
            auto Ky = addr[i].ky;
            auto k = addr[i].k;

            // 设置偏移量
            auto bnRightOffset = right + (b * this->head_num_ + n) * this->query_sequence_len_ * dim_size_q;
            auto bnOutOffset = out + (b * this->head_num_ + n) * this->key_value_sequence_len_ * dim_size_k;
            int64_t gIndex = n / (this->head_num_ / this->gqa_group_num_);
            auto bn_out_offset_gqa =
                out + (b * this->gqa_group_num_ + gIndex) * this->key_value_sequence_len_ * dim_size_k;
            src[i].left = ((roundId + 1) % 2) ? leftOffsetRoundEven : leftOffsetRoundOdd;
            src[i].right = bnRightOffset + ir * SIZE_128 * dim_size_q;
            src[i].out = bn_out_offset_gqa + ic * SIZE_128 * dim_size_k;
            src[i].kx = Kx;
            src[i].ky = Ky;
            src[i].k = k;
            src[i].lineStride = Kx * ATTENTION_SCORE_BLOCK_SIZE;
            src[i].lowerLeft = false;
            src[i].upperRight = false;

            // 多段时，work space偏移量更新
            leftOffsetRoundEven += k * ATTENTION_SCORE_BLOCK_SIZE;
            leftOffsetRoundOdd += k * ATTENTION_SCORE_BLOCK_SIZE;
        }
    }

    template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
    __aicore__ __inline__ void addrMapping_triangular_cube3(__gm__ T_LEFT *__restrict__ left,
        __gm__ T_RIGHT *__restrict__ right, __gm__ T_OUTPUT *__restrict__ out, const BackWardAddr *addr,
        PhyAddrBackwardCube3<T_LEFT, T_RIGHT, T_OUTPUT> *src,
        int64_t &srcLen, int32_t roundId, int64_t dim_size_q, int64_t dim_size_k)
    {
        // 开启work space
        auto leftOffsetRoundEven =
            left + this->backward_local_core_index_ * this->backward_block_num_per_core_ * ATTENTION_SCORE_BLOCK_SIZE;
        auto leftOffsetRoundOdd =
            left + this->backward_core_num_ * this->backward_block_num_per_core_ * ATTENTION_SCORE_BLOCK_SIZE +
            this->backward_local_core_index_ * this->backward_block_num_per_core_ * ATTENTION_SCORE_BLOCK_SIZE;
        int64_t triBlocksPerColumn = this->backward_block_num_per_column_ - 2 * this->is_odd_;
        int64_t index = 0;
        for (int32_t i = 0; i < srcLen; ++i) {
            // left、right、out的地址配置
            int64_t iR = addr[i].iR;
            int64_t iC = addr[i].iC;
            int64_t kx = addr[i].kx;
            int64_t k = addr[i].k;

            // 倒三角和非倒三角这三个变量不一样
            int64_t switchIndex = triBlocksPerColumn + iR + (iR + 1) % 2;
            int64_t rowIndexLeftSection = triBlocksPerColumn + iR;  // 倒三角 ：非倒三角
            int64_t rowOffset = (iR + 1) % 2 == 1 ? -1 : 1;
            int64_t rowIndexRightSection = triBlocksPerColumn - 1 - iR + rowOffset;
            int64_t colIndexLeftSection = iC;
            int64_t colIndexRightSection = iC - switchIndex - 1;

            int64_t bnOffsetLeftMatrix =
                ((addr[i].b * this->head_num_ + addr[i].n) * this->backward_block_num_per_row_ *
                    this->backward_block_num_per_column_) *
                ATTENTION_SCORE_BLOCK_SIZE;  // 当前b,n下的偏移量
            int64_t bnOffsetRightMatrix = (addr[i].b * this->head_num_ + addr[i].n) * this->query_sequence_len_ *
                                             dim_size_q;  // 当前b,n下的偏移量
            int64_t bn_offset_out = (addr[i].b * this->head_num_ + addr[i].n) * this->key_value_sequence_len_ *
                                    dim_size_k;  // 当前b,n下的偏移量
            int64_t gIndex = addr[i].n / (this->head_num_ / this->gqa_group_num_);
            int64_t bn_offset_gqa_out_matrix =
                (addr[i].b * this->gqa_group_num_ + gIndex) * this->key_value_sequence_len_ * dim_size_k;
            // sparse场景：is_tri == true 以及 sparseMode == 1
            bool sparseFlag = false;
            int64_t windowBlockSize = this->window_length_ / 128;
            if (this->is_triangle_ && this->sparse_mode_ == 1) {
                sparseFlag = iR > ((windowBlockSize - 1) / 2) ? true : false;
                switchIndex = (windowBlockSize / 2) + iR;
                rowIndexLeftSection = (windowBlockSize / 2) + iR;
                rowIndexRightSection = (windowBlockSize / 2) - 1 - iR;
                colIndexLeftSection = iC;
                colIndexRightSection = iC - switchIndex - 1;
            }
            int64_t rowIndexSparseSection = iR + (windowBlockSize / 2);
            int64_t colIndexSparseSection = iR + iC - (windowBlockSize / 2);

            // 开启double buffer
            int64_t dbOffset = (roundId % 2) * (this->backward_core_num_ * this->backward_block_num_per_core_ *
                                                     ATTENTION_SCORE_BLOCK_SIZE);
            if (index == 0) {
                src[index].left =
                    left + dbOffset +
                    this->backward_local_core_index_ * this->backward_block_num_per_core_ * ATTENTION_SCORE_BLOCK_SIZE;
            } else {
                src[index].left = src[index - 1].left + src[index - 1].k * ATTENTION_SCORE_BLOCK_SIZE;
            }

            if (!sparseFlag && switchIndex < iC) {
                // BASE_BLOCK_SIZE
                src[index].right =
                    right + bnOffsetRightMatrix + rowIndexRightSection * dim_size_q * SIZE_128;
                src[index].out = out + bn_offset_gqa_out_matrix + colIndexRightSection * SIZE_128 * dim_size_k;
                src[index].kx = kx;
                src[index].ky = this->backward_block_num_ky_;
                src[index].k = k;
                src[index].upperRight = iC + src[index].kx >= backward_block_num_per_row_ ? true : false;
                src[index].lowerLeft = false;
                src[index].lineStride = src[index].kx * ATTENTION_SCORE_BLOCK_SIZE;
                ++index;
            } else if (!sparseFlag && iC <= switchIndex && iC + kx - 1 > switchIndex) {
                src[index].right = right + bnOffsetRightMatrix + rowIndexLeftSection * dim_size_q * SIZE_128;
                src[index].out = out + bn_offset_gqa_out_matrix + colIndexLeftSection * SIZE_128 * dim_size_k;
                src[index].kx = switchIndex - iC + 1;
                src[index].ky = this->backward_block_num_ky_;
                src[index].k = src[index].kx * this->backward_block_num_ky_;
                src[index].upperRight = true;
                src[index].lowerLeft = false;
                src[index].lineStride = src[index].kx * ATTENTION_SCORE_BLOCK_SIZE;

                src[index + 1].left = src[index].left + src[index].k * ATTENTION_SCORE_BLOCK_SIZE;
                src[index + 1].right =
                    right + bnOffsetRightMatrix + rowIndexRightSection * dim_size_q * SIZE_128;
                src[index + 1].out = out + bn_offset_gqa_out_matrix;
                src[index + 1].kx = kx - src[index].kx;
                src[index + 1].ky = this->backward_block_num_ky_;
                src[index + 1].k = src[index + 1].kx * this->backward_block_num_ky_;
                src[index + 1].upperRight =
                    switchIndex + src[index + 1].kx >= backward_block_num_per_row_ - 1 ? true : false;
                src[index + 1].lowerLeft = false;
                src[index + 1].lineStride = src[index + 1].kx * ATTENTION_SCORE_BLOCK_SIZE;
                index += 2;
            } else if (!sparseFlag && iC <= switchIndex && iC + kx - 1 <= switchIndex) {
                src[index].right = right + bnOffsetRightMatrix + rowIndexLeftSection * dim_size_q * SIZE_128;
                src[index].out = out + bn_offset_gqa_out_matrix + colIndexLeftSection * SIZE_128 * dim_size_k;
                src[index].kx = kx;
                src[index].ky = this->backward_block_num_ky_;
                src[index].k = k;
                src[index].upperRight = iC + src[index].kx - 1 >= switchIndex ? true : false;
                src[index].lowerLeft = false;
                src[index].lineStride = src[index].kx * ATTENTION_SCORE_BLOCK_SIZE;
                index++;
            } else {
                src[index].right =
                    right + bnOffsetRightMatrix + rowIndexSparseSection * ATTENTION_SCORE_BLOCK_SIZE;
                src[index].out = out + bn_offset_gqa_out_matrix + colIndexSparseSection * ATTENTION_SCORE_BLOCK_SIZE;
                src[index].k = k;
                ++index;
            }
        }
        srcLen = index;
    }

public:
    __aicore__ __inline__ void init(int32_t batchSize, int32_t headNum, int32_t query_sequence_len,
        int32_t key_value_sequence_len, int32_t headDim, int32_t gqa_group_num, bool isTriangle, int32_t sparseMode,
        int32_t windowLength)
    {
        // B N S D初始化
        this->batch_size_ = batchSize;
        this->head_num_ = headNum;
        this->query_sequence_len_ = query_sequence_len;
        this->key_value_sequence_len_ = key_value_sequence_len;
        this->head_dim_ = headDim;
        this->gqa_group_num_ = gqa_group_num;
        this->is_odd_ = this->query_sequence_len_ / BASE_BLOCK_LENGTH / TILING_DIVIDE % TILING_DIVIDE;
        // 负责均衡前信息的初始化
        this->blockRows = this->query_sequence_len_ / BASE_BLOCK_LENGTH;
        this->blockCols = this->key_value_sequence_len_ / BASE_BLOCK_LENGTH;

        // 初始化负载均衡的信息
        this->is_triangle_ = isTriangle;

        this->sparse_mode_ = sparseMode;
        this->window_length_ = windowLength;
        this->window_size_ = this->window_length_ / SIZE_128;
    }

    __aicore__ __inline__ int32_t get_total_rounds()
    {
        return this->backward_total_rounds_;
    }

    __aicore__ __inline__ bool is_running(int32_t roundId)
    {
        return (roundId * this->backward_block_num_per_core_ * this->backward_core_num_ +
                   this->backward_local_core_index_ * this->backward_block_num_per_core_) <
                this->backward_total_blocks_;
    }

    __aicore__ __inline__ void set_backward_tiling(int32_t backward_core_num, int32_t backward_local_core_index,
        int32_t backward_block_num_per_core, int32_t backward_block_num_ky)
    {
        // 初始化
        this->backward_core_num_ = backward_core_num;
        this->backward_local_core_index_ = backward_local_core_index;
        this->backward_block_num_per_core_ = backward_block_num_per_core;
        this->backward_block_num_ky_ = backward_block_num_ky;

        // 根据负载均衡信息来计算基本块
        this->backward_block_num_per_column_ = this->blockRows;
        this->backward_block_num_per_row_ = this->blockCols;
        if (this->is_triangle_) {
            this->backward_block_num_per_column_ = this->blockRows / TILING_DIVIDE + this->is_odd_;
            this->backward_block_num_per_row_ = this->blockCols + TILING_DIVIDE * (1 - this->is_odd_);
            if (this->sparse_mode_ == 1) {
                this->backward_block_num_per_column_ = this->blockRows - this->window_size_ / TILING_DIVIDE;
                this->backward_block_num_per_row_ = this->window_size_ + 1;
            }
        }

        // 初始化基本块数量
        this->backward_block_num_per_head_ = this->backward_block_num_per_column_ * this->backward_block_num_per_row_;
        this->backward_block_num_per_batch_ = this->head_num_ * this->backward_block_num_per_head_;
        this->backward_total_blocks_ = this->batch_size_ * this->backward_block_num_per_batch_;

        // 轮次的计算
        this->backward_total_rounds_ =
            (this->backward_total_blocks_ + this->backward_block_num_per_core_ * this->backward_core_num_ - 1) /
            (this->backward_block_num_per_core_ * this->backward_core_num_);
    }

    template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
    __aicore__ __inline__ void addrMapping_cube1(__gm__ T_LEFT *__restrict__ left, __gm__ T_RIGHT *__restrict__ right,
        __gm__ T_OUTPUT *__restrict__ out, PhyAddrBackwardCube1<T_LEFT, T_RIGHT, T_OUTPUT> *src, int64_t &srcLen,
        int64_t roundId, int64_t dim_size_q, int64_t dim_size_k)
    {
        // 寻址的预处理
        BackWardAddr BackWardAddr[MAX_SWITCH_TIME];
        BackWardAddrMapping_pre(BackWardAddr, srcLen, roundId);

        // cube1地址偏移的计算
        if (this->is_triangle_) {
            addrMapping_triangular_cube1(left, right, out, BackWardAddr, src, srcLen, roundId, dim_size_q, dim_size_k);
        } else {  // no-mask
            addrMapping_rectangular_cube1(left, right, out, BackWardAddr, src, srcLen, roundId, dim_size_q, dim_size_k);
        }
    }

    template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
    __aicore__ __inline__ void addrMapping_cube2(__gm__ T_LEFT *__restrict__ left, __gm__ T_RIGHT *__restrict__ right,
        __gm__ T_OUTPUT *__restrict__ out, PhyAddrBackwardCube2<T_LEFT, T_RIGHT, T_OUTPUT> *src, int64_t &srcLen,
        int64_t roundId, int64_t dim_size_q, int64_t dim_size_k)
    {
        // 寻址的预处理
        BackWardAddr BackWardAddr[MAX_SWITCH_TIME];
        BackWardAddrMapping_pre(BackWardAddr, srcLen, roundId);

        // cube1地址偏移的计算
        if (this->is_triangle_) {
            addrMapping_triangular_cube2(left, right, out, BackWardAddr, src, srcLen, roundId, dim_size_q, dim_size_k);
        } else {
            addrMapping_rectangular_cube2(left, right, out, BackWardAddr, src, srcLen, roundId, dim_size_q, dim_size_k);
        }
    }

    template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
    __aicore__ __inline__ void addrMapping_cube3(__gm__ T_LEFT *__restrict__ left, __gm__ T_RIGHT *__restrict__ right,
        __gm__ T_OUTPUT *__restrict__ out, PhyAddrBackwardCube3<T_LEFT, T_RIGHT, T_OUTPUT> *src, int64_t &srcLen,
        int64_t roundId, int64_t dim_size_q, int64_t dim_size_k)
    {
        // 寻址的预处理
        BackWardAddr BackWardAddr[MAX_SWITCH_TIME];
        BackWardAddrMapping_pre(BackWardAddr, srcLen, roundId);

        // cube1地址偏移的计算
        if (this->is_triangle_) {
            addrMapping_triangular_cube3(left, right, out, BackWardAddr, src, srcLen, roundId, dim_size_q, dim_size_k);
        } else {
            addrMapping_rectangular_cube3(left, right, out, BackWardAddr, src, srcLen, roundId, dim_size_q, dim_size_k);
        }
    }

private:
    // B N S D的格式
    int32_t batch_size_;
    int32_t head_num_;
    int32_t query_sequence_len_;
    int32_t key_value_sequence_len_;
    int32_t head_dim_;
    int32_t gqa_group_num_;

    int32_t is_odd_;
    // 负载均衡前的信息
    int32_t blockRows;  // 行数
    int32_t blockCols;  // 列数

    // 负责均衡的信息
    bool is_triangle_;       // 三角形的标志
    int32_t sparse_mode_;    // sparse模式
    int32_t window_length_;  // 滑动窗口
    int32_t window_size_;    // 滑动窗口

    // 前向的信息
    int32_t forward_core_num_;            // 前向的核数
    int32_t forward_group_num_;           // 前向核心的组
    int32_t forward_core_num_per_group_;  // 每个组的核心数
    int32_t forward_local_core_index_;    // 当前核心的index
    int32_t forward_total_rounds_;        // 前向的总轮次

    // 反向的信息
    int32_t backward_core_num_;              // 反向的核数
    int32_t backward_local_core_index_;      // 反向当前核的index
    int32_t backward_block_num_per_core_;    // 反向每个核计算的基本块数量
    int32_t backward_block_num_ky_;          // 反向按y方向变量的基本块数量
    int32_t backward_block_num_per_row_;     // 反向每行的基本块数量
    int32_t backward_block_num_per_column_;  // 反向每列的基本块数量
    int32_t backward_block_num_per_head_;    // 反向每个head的基本块数量
    int32_t backward_block_num_per_batch_;   // 反向每个batch的基本块数量
    int32_t backward_total_blocks_;          // 反向计算的总块数
    int32_t backward_total_rounds_;          // 反向总共的轮次
};
}  // namespace Address
#endif