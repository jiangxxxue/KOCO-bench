/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
/**
 * AddressMapping_Vector头文件，用于Vector的寻址
 */
#ifndef ADDRESS_MODULE_ADDRESSMAPPING_VECTOR_H
#define ADDRESS_MODULE_ADDRESSMAPPING_VECTOR_H

#include <cstdint>
#include "address_const.h"

namespace Address {
// 定义常量
const int MODULUS_TWO = 2;
const int CONSTANT_FACTOR_TWO = 2;
const int VECTOR_NUM_PER_CUBE = 2;
const int WINDOW_LEN_ADJUSTOR = 2;

class AddressMappingVector {
public:
    __aicore__ __inline__ void addrMapping_pre(int64_t round_id, VectorAddr *vector_addr, int64_t &addr_len)
    {
        // 起始块的索引id
        int64_t cur_block_id =
            round_id * this->global_.blockNumPerLoop + this->local_.cubeIndex * this->global_.blockNumPerCube;
        int64_t row_num_per_round = this->ky_;
        int64_t col_num_per_round = this->global_.blockNumPerCube / this->ky_;

        // 最后轮次的尾块处理
        int64_t remain = this->global_.blockNumPerCube;
        if (cur_block_id + this->global_.blockNumPerCube > this->mTotalBlocks) {
            remain = this->mTotalBlocks - cur_block_id;
        }

        // 设置x,y方向的基本块数量
        int64_t Ky = this->ky_;
        int64_t Kx = remain / Ky;

        // 计算起始的B N S D
        int64_t b = cur_block_id / this->global_.blockNumPerBatch;
        int64_t n = cur_block_id % this->global_.blockNumPerBatch / this->global_.blockNumPerHead;
        int64_t block_row =
            cur_block_id % this->global_.blockNumPerHead / (row_num_per_round * this->global_.blockNumPerRow);
        int64_t ir = block_row * row_num_per_round;
        int64_t ic = (round_id * this->global_.cubeNum * col_num_per_round + local_.cubeIndex * col_num_per_round) %
                     (this->global_.blockNumPerHead) % this->global_.blockNumPerRow;

        // 处理边界：
        vector_addr[0].b = b;
        vector_addr[0].n = n;
        vector_addr[0].iR = ir;
        vector_addr[0].iC = ic;
        vector_addr[0].kx = Kx;
        vector_addr[0].ky = Ky;
        vector_addr[0].k = remain;

        int64_t index = 0;
        for (; remain > 0;) {
            // 跳变点的位置
            int64_t switch_index = vector_addr[index].iR + this->mBalanceRow +
                                   (vector_addr[index].iR + 1) % MODULUS_TWO;
            if (this->global_.triMatrix && (vector_addr[index].iC <= switch_index) &&
                (vector_addr[index].iC + vector_addr[index].kx - 1 > switch_index)) {  // 沿着跳变点切分
                vector_addr[index].kx = switch_index - vector_addr[index].iC + 1;
                vector_addr[index].k = vector_addr[index].kx * vector_addr[index].ky;

                vector_addr[index + 1].b = vector_addr[index].b;
                vector_addr[index + 1].n = vector_addr[index].n;
                vector_addr[index + 1].iR = vector_addr[index].iR;
                vector_addr[index + 1].iC = switch_index + 1;
                vector_addr[index + 1].k = remain - vector_addr[index].k;
                vector_addr[index + 1].ky = vector_addr[index].ky;
                vector_addr[index + 1].kx = vector_addr[index + 1].k / vector_addr[index + 1].ky;
            }
            if (vector_addr[index].iC + vector_addr[index].kx > this->global_.blockNumPerRow) {  // 换行
                vector_addr[index].kx = this->global_.blockNumPerRow - vector_addr[index].iC;
                vector_addr[index].k = vector_addr[index].kx * vector_addr[index].ky;

                vector_addr[index + 1].b = vector_addr[index].b;
                vector_addr[index + 1].n = vector_addr[index].n;
                vector_addr[index + 1].iR = vector_addr[index].iR + vector_addr[index].ky;
                vector_addr[index + 1].iC = 0;
                vector_addr[index + 1].k = remain - vector_addr[index].k;
                vector_addr[index + 1].ky = vector_addr[index].ky;
                vector_addr[index + 1].kx = vector_addr[index + 1].k / vector_addr[index + 1].ky;
                if (vector_addr[index + 1].iR >= this->global_.blockNumPerCol) {  // 换head
                    vector_addr[index + 1].n = vector_addr[index].n + 1;
                    vector_addr[index + 1].iR = vector_addr[index + 1].iR % this->global_.blockNumPerCol;
                    if (vector_addr[index + 1].n >= this->global_.headNum) {  // 换batch
                        vector_addr[index + 1].b = vector_addr[index].b + 1;
                        vector_addr[index + 1].n = 0;
                    }
                }
            }

            remain -= vector_addr[index].k;
            ++index;
        }

        // 处理空的段
        int64_t pos = 0;
        for (size_t i = 0; i < index; ++i) {
            if (vector_addr[i].k == 0) {
                continue;
            }
            vector_addr[pos++] = vector_addr[i];
        }
        addr_len = pos;
    }

    __aicore__ __inline__ void addrMapping_rectangle_nomark(
        int64_t round_id, VectorAddr *vector_addr, int64_t &addr_len)
    {
        // workspace:
        int64_t work_space_offset_odd =
            this->local_.cubeIndex * this->global_.blockNumPerCube * ATTENTION_SCORE_BLOCK_SIZE;
        int64_t work_space_offset_even =
            this->global_.cubeNum * this->global_.blockNumPerCube * ATTENTION_SCORE_BLOCK_SIZE +
            this->local_.cubeIndex * this->global_.blockNumPerCube * ATTENTION_SCORE_BLOCK_SIZE;
        int64_t work_space_offset = ((round_id + 1) % MODULUS_TWO) ? work_space_offset_odd : work_space_offset_even;

        // 分段的偏移量
        for (size_t i = 0; i < addr_len; ++i) {
            for (size_t j = 0; j < this->ky_; ++j) {
                int64_t b = vector_addr[i].b;
                int64_t n = vector_addr[i].n;
                int64_t ir = vector_addr[i].iR + j;
                int64_t ic = vector_addr[i].iC;
                int64_t kx = vector_addr[i].kx;

                // 更新section信息
                int64_t index = i * this->ky_ + j;
                this->section_.section_start_block[index] =
                    ((i == 0) && (j == 0))
                        ? 0
                        : (this->section_.section_start_block[index - 1] + this->section_.len[index - 1]);
                this->section_.len[index] = kx;
                this->section_.global_lines_in_heads[index] =
                    (b * this->global_.headNum + n) * this->global_.blockRows * BASE_BLOCK_LENGTH +
                    ir * BASE_BLOCK_LENGTH + this->local_.startLineInBaseBlock;
                this->section_.O_dO_offset[index] =
                    this->section_.global_lines_in_heads[index] * this->global_.headDim;
                this->section_.S_dP_offset[index] =
                    work_space_offset + this->section_.section_start_block[index] * ATTENTION_SCORE_BLOCK_SIZE;

                // mask
                this->section_.head_apply_mask[index] = false;
                this->section_.tail_apply_mask[index] = false;
            }
        }
        this->section_.processLines = BASE_BLOCK_LENGTH / VECTOR_NUM_PER_CUBE;
        this->section_.sectionNum = addr_len * this->ky_;
    }

    __aicore__ __inline__ void addrMapping_rectangle_mark(int64_t round_id, VectorAddr *vector_addr, int64_t &addr_len)
    {
        // workspace:
        int64_t work_space_offset_odd =
            this->local_.cubeIndex * this->global_.blockNumPerCube * ATTENTION_SCORE_BLOCK_SIZE;
        int64_t work_space_offset_even =
            this->global_.cubeNum * this->global_.blockNumPerCube * ATTENTION_SCORE_BLOCK_SIZE +
            this->local_.cubeIndex * this->global_.blockNumPerCube * ATTENTION_SCORE_BLOCK_SIZE;
        int64_t work_space_offset = ((round_id + 1) % MODULUS_TWO) ? work_space_offset_odd : work_space_offset_even;

        // 分段的偏移量
        int32_t index = 0;
        for (size_t i = 0; i < addr_len; ++i) {
            for (size_t j = 0; j < vector_addr[i].ky; ++j) {
                int64_t b = vector_addr[i].b;
                int64_t n = vector_addr[i].n;
                int64_t ir = vector_addr[i].iR + j;
                int64_t ic = vector_addr[i].iC;
                int64_t kx = vector_addr[i].kx;

                int64_t left_ir = ir + this->mBalanceRow;
                int64_t row_offset = (ir + 1) % MODULUS_TWO == 1 ? -1 : 1;
                int64_t right_ir = this->mBalanceRow - 1 - ir + row_offset;
                int64_t left_global_line_offset =
                    (b * this->global_.headNum + n) * this->global_.blockRows * BASE_BLOCK_LENGTH +
                    left_ir * BASE_BLOCK_LENGTH + this->local_.startLineInBaseBlock;
                int64_t right_global_line_offset =
                    (b * this->global_.headNum + n) * this->global_.blockRows * BASE_BLOCK_LENGTH +
                    right_ir * BASE_BLOCK_LENGTH + this->local_.startLineInBaseBlock;

                // 跳变点的索引
                int64_t switch_index = ir + this->mBalanceRow + (ir + 1) % MODULUS_TWO;
                if (ir >= this->mBalanceRow) {  // 256奇数块的最后两行
                    this->section_.head_skip_block[index] = 0;
                    this->section_.tail_skip_block[index] =
                        (ir + 1) % MODULUS_TWO && ic + kx == this->global_.blockNumPerRow ? 1 : 0;
                    this->section_.len[index] =
                        (ir + 1) % MODULUS_TWO && ic + kx == this->global_.blockNumPerRow ? kx - 1 : kx;
                    this->section_.section_start_block[index] =
                        (ir + 1) % MODULUS_TWO && index == 0
                            ? 0
                            : (!((ir + 1) % MODULUS_TWO) && index == 0
                                      ? kx
                                      : this->section_.section_start_block[index - 1] + this->section_.len[index - 1] +
                                            this->section_.head_skip_block[index - 1] +
                                            this->section_.tail_skip_block[index - 1]);
                    this->section_.global_lines_in_heads[index] = left_global_line_offset;
                    this->section_.O_dO_offset[index] =
                        this->section_.global_lines_in_heads[index] * this->global_.headDim;
                    this->section_.S_dP_offset[index] =
                        work_space_offset + this->section_.section_start_block[index] * ATTENTION_SCORE_BLOCK_SIZE;
                    // mask
                    this->section_.head_apply_mask[index] = false;
                    this->section_.tail_apply_mask[index] =
                        (ic + kx >= this->global_.blockNumPerRow - 1 && (ir + 1) % MODULUS_TWO) ||
                        (ic + kx == this->global_.blockNumPerRow && !((ir + 1) % MODULUS_TWO));
                    index = this->section_.len[index] != 0 ? index + 1 : index;
                } else if (ir < this->mBalanceRow && switch_index < ic) {
                    this->section_.head_skip_block[index] = 0;
                    this->section_.tail_skip_block[index] =
                        (ir + 1) % MODULUS_TWO && ic + kx == this->global_.blockNumPerRow ? 1 : 0;
                    this->section_.len[index] =
                        (ir + 1) % MODULUS_TWO && ic + kx == this->global_.blockNumPerRow ? kx - 1 : kx;
                    this->section_.section_start_block[index] =
                        (ir + 1) % MODULUS_TWO && index == 0
                            ? 0
                            : (!((ir + 1) % MODULUS_TWO) && index == 0
                                      ? kx
                                      : this->section_.section_start_block[index - 1] + this->section_.len[index - 1] +
                                            this->section_.head_skip_block[index - 1] +
                                            this->section_.tail_skip_block[index - 1]);
                    this->section_.global_lines_in_heads[index] = right_global_line_offset;
                    this->section_.O_dO_offset[index] =
                        this->section_.global_lines_in_heads[index] * this->global_.headDim;
                    this->section_.S_dP_offset[index] =
                        work_space_offset + this->section_.section_start_block[index] * ATTENTION_SCORE_BLOCK_SIZE;
                    // mask
                    this->section_.head_apply_mask[index] = false;
                    this->section_.tail_apply_mask[index] =
                        (ic + kx >= this->global_.blockNumPerRow - 1 && (ir + 1) % MODULUS_TWO) ||
                        (ic + kx == this->global_.blockNumPerRow && !((ir + 1) % MODULUS_TWO));
                    index = this->section_.len[index] != 0 ? index + 1 : index;
                } else {
                    this->section_.head_skip_block[index] = 0;
                    this->section_.tail_skip_block[index] = (ir + 1) % MODULUS_TWO && ic + kx - 1 == switch_index ?
                                                            1 : 0;
                    // len和workspace位置
                    this->section_.len[index] = (ir + 1) % MODULUS_TWO && ic + kx - 1 == switch_index ? kx - 1 : kx;
                    this->section_.section_start_block[index] =
                        (ir + 1) % MODULUS_TWO && index == 0
                            ? 0
                            : (!((ir + 1) % MODULUS_TWO) && index == 0
                                      ? kx
                                      : this->section_.section_start_block[index - 1] + this->section_.len[index - 1] +
                                            this->section_.head_skip_block[index - 1] +
                                            this->section_.tail_skip_block[index - 1]);
                    // line、O 、S的偏移量
                    this->section_.global_lines_in_heads[index] = left_global_line_offset;
                    this->section_.O_dO_offset[index] =
                        this->section_.global_lines_in_heads[index] * this->global_.headDim;
                    this->section_.S_dP_offset[index] =
                        work_space_offset + this->section_.section_start_block[index] * ATTENTION_SCORE_BLOCK_SIZE;
                    // mask
                    this->section_.head_apply_mask[index] = false;
                    this->section_.tail_apply_mask[index] =
                        (ic + kx >= switch_index && (ir + 1) % MODULUS_TWO) ||
                        (ic + kx - 1 == switch_index && !((ir + 1) % MODULUS_TWO));
                    index = this->section_.len[index] != 0 ? index + 1 : index;
                }
            }
        }
        this->section_.processLines = BASE_BLOCK_LENGTH / VECTOR_NUM_PER_CUBE;
        this->section_.sectionNum = index;
    }

public:
    __aicore__ __inline__ void init_global_info(int64_t batchSize, int64_t headNum, int64_t query_sequence_len,
        int64_t key_value_sequence_len, int64_t headDim, bool isTriangle, int64_t sparseMode, int64_t windowLength,
        int64_t cubeNum, int64_t blockNumPerCube, int64_t ky)
    {
        // 初始化
        this->global_.batchNum = batchSize;
        this->global_.headNum = headNum;
        this->global_.seqLenQ = query_sequence_len;
        this->global_.seqLenK = key_value_sequence_len;
        this->global_.headDim = headDim;
        this->global_.triMatrix = isTriangle;
        this->global_.isSparse = sparseMode;
        this->global_.windowLength = windowLength;
        this->global_.windowsBlockNum = windowLength / BASE_BLOCK_LENGTH;
        this->global_.cubeNum = cubeNum;
        this->global_.blockNumPerCube = blockNumPerCube;
        this->ky_ = ky;
        this->mOddFlag = this->global_.seqLenQ / SIZE_256 % MODULUS_TWO;

        // tiling策略
        this->global_.blockRows = this->global_.seqLenQ / BASE_BLOCK_LENGTH;
        this->global_.blockCols = this->global_.seqLenK / BASE_BLOCK_LENGTH;
        this->global_.blockNumPerCol = this->global_.blockRows;
        this->global_.blockNumPerRow = this->global_.blockCols;
        if (this->global_.triMatrix) {  // 三角标志
            if (this->mOddFlag) {
                this->mBalanceRow = (this->global_.seqLenQ - SIZE_256) / SIZE_128 / CONSTANT_FACTOR_TWO;
                this->global_.blockNumPerCol = this->global_.blockRows / CONSTANT_FACTOR_TWO + 1;
                this->global_.blockNumPerRow = this->global_.blockCols;
            } else {
                this->mBalanceRow = this->global_.seqLenQ / SIZE_128 / CONSTANT_FACTOR_TWO;
                this->global_.blockNumPerCol = this->global_.blockRows / CONSTANT_FACTOR_TWO;
                this->global_.blockNumPerRow = this->global_.blockCols + CONSTANT_FACTOR_TWO;
            }
            if (this->global_.isSparse == 1) {  // sparse mode
                this->global_.blockNumPerCol = this->global_.windowsBlockNum + WINDOW_LEN_ADJUSTOR;
                this->global_.blockNumPerRow = this->global_.blockNumPerCol -
                                               this->global_.windowsBlockNum / CONSTANT_FACTOR_TWO;
            }
        }

        // 计算块数、轮次等
        this->global_.blockNumPerLoop = this->global_.cubeNum * this->global_.blockNumPerCube;
        this->global_.blockNumPerHead = this->global_.blockNumPerRow * this->global_.blockNumPerCol;
        this->global_.blockNumPerBatch = this->global_.headNum * this->global_.blockNumPerHead;
        this->mTotalBlocks = this->global_.batchNum * this->global_.headNum * this->global_.blockNumPerHead;
        this->mTotalRounds =
            (this->mTotalBlocks + this->global_.blockNumPerLoop - 1) / this->global_.blockNumPerLoop;
        this->global_.tailBlockNum = this->mTotalBlocks % this->global_.blockNumPerLoop;
    }

    __aicore__ __inline__ void init_local_info(int64_t local_cube_index, int64_t local_vector_index)
    {
        // 初始化
        this->local_.cubeIndex = local_cube_index;
        this->local_.vectorIdx = local_vector_index;

        // 计算
        this->local_.startLineInBaseBlock = BASE_BLOCK_LENGTH / VEC_NUM_PER_CUBE * this->local_.vectorIdx;
        this->local_.procTail = false;
        if (this->global_.tailBlockNum > 0) {
            int64_t cube_used = this->global_.tailBlockNum / this->global_.blockNumPerCube;
            int64_t tail_block = this->global_.tailBlockNum % this->global_.blockNumPerCube;

            if (this->local_.cubeIndex < cube_used) {
                this->local_.procTail = true;
                this->local_.procTailBlockNum = this->global_.blockNumPerCube;
            }
            if (tail_block > 0) {
                this->local_.procTail = true;
                this->local_.procTailBlockNum = tail_block;
            }
        }
    }

    __aicore__ __inline__ int64_t

        get_total_rounds()
    {
        return this->mTotalRounds;
    }

    __aicore__ __inline__ const SECTION_INFO &get_section_info(int64_t round_id)
    {
        clear_section();
        VectorAddr vector_addr[MAX_SWITCH_TIME];
        int64_t addr_len = 0;
        addrMapping_pre(round_id, vector_addr, addr_len);

        if (this->global_.triMatrix) {
            addrMapping_rectangle_mark(round_id, vector_addr, addr_len);
        } else {
            addrMapping_rectangle_nomark(round_id, vector_addr, addr_len);
        }
        return this->section_;
    }

    __aicore__ __inline__ bool is_running(int64_t round_id)
    {
        // 起始块的索引id
        int64_t cur_block_id =
            round_id * this->global_.blockNumPerLoop + this->local_.cubeIndex * this->global_.blockNumPerCube;

        return cur_block_id < this->mTotalBlocks;
    }

protected:
    __aicore__ __inline__ void clear_section()
    {
        this->section_.sectionNum = 0;
        this->section_.section_start_block[MAX_SWITCH_TIME] = {0};
        this->section_.head_skip_block[MAX_SWITCH_TIME] = {0};
        this->section_.tail_skip_block[MAX_SWITCH_TIME] = {0};
        this->section_.global_lines_in_heads[MAX_SWITCH_TIME] = {0};
        this->section_.len[MAX_SWITCH_TIME] = {0};
        this->section_.head_apply_mask[MAX_SWITCH_TIME] = {false};
        this->section_.tail_apply_mask[MAX_SWITCH_TIME] = {false};
        this->section_.O_dO_offset[MAX_SWITCH_TIME] = {0};
        this->section_.S_dP_offset[MAX_SWITCH_TIME] = {0};
        this->section_.processLines = 0;
    }

private:
    // vector全局信息
    GLOBAL_INFO global_;
    // vector当前信息
    LOCAL_INFO local_;
    // vector分段信息
    SECTION_INFO section_;
    // 总块数
    int64_t mTotalBlocks;
    // 轮次
    int64_t mTotalRounds;
    // ky方向
    int64_t ky_;
    // 序列长度是否为256倍数的奇数倍：true表示是
    bool mOddFlag;
    // 负载均衡矩阵的行数
    int64_t mBalanceRow;
};
}  // namespace Address
#endif
