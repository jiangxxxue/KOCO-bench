# 
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# 
import os
import unittest
import numpy as np
import torch
import random
import sys
import logging
np.set_printoptions(threshold=np.inf)

import op_test

class TestReshapeAndCache(op_test.OpTest):
    def setUp(self):
        super().setUp()

    def set_reshape_and_cache_param(self, num_heads, head_size, block_size, num_blocks, dtype, batch):
        self.num_heads = num_heads
        self.head_size = head_size
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.dtype = dtype
        self.batch = batch
        self.seqLen = np.random.randint(4000, 4001, size=(self.batch)).astype(np.int32)
        self.num_tokens = np.sum(self.seqLen)
        self.wins = np.random.randint(1501, 1502, size=(self.batch *self.num_heads)).astype(np.int32)

        if self.dtype == "bfloat16":
            self.key_expect = np.zeros((self.num_blocks, self.block_size, 1, self.head_size)).astype(np.float32)
            self.value_expect = np.zeros((self.num_blocks, self.block_size, 1, self.head_size)).astype(np.float32)
        else:
            self.key_expect = np.zeros((self.num_blocks, self.block_size, 1, self.head_size)).astype(self.dtype)
            self.value_expect = np.zeros((self.num_blocks, self.block_size, 1, self.head_size)).astype(self.dtype)


        if self.dtype == "bfloat16":
            self.key = np.random.randint(1, 11, size=(self.num_tokens, self.num_heads, self.head_size)).astype(np.float32)
            self.value = np.random.randint(1, 11, size=(self.num_tokens, self.num_heads, self.head_size)).astype(np.float32)
        else:
            self.key = np.random.randint(1, 11, size=(self.num_tokens, self.num_heads, self.head_size)).astype(self.dtype)
            self.value = np.random.randint(1, 11, size=(self.num_tokens, self.num_heads, self.head_size)).astype(self.dtype)

        value = []
        sumX = 0
        for i in range(self.batch*self.num_heads):
            x = (self.wins[i] + self.block_size - 1) // self.block_size
            value.append(sumX * self.block_size)
            sumX += x

        self.slot_mapping = np.array(value).astype(np.int32)


    def generate_test_data(self):
        key_expect = self.key
        value_expect = self.value

        self.new_seq = self.seqLen
        self.new_seq[0] = self.seqLen[0]

        for n in range(1, len(self.seqLen)):
            self.new_seq[n] = self.seqLen[n] + self.seqLen[n-1]

        for i, slot in enumerate(self.slot_mapping):
            if slot < 0:
                continue
            curSlot = slot
            win = self.wins[i]
            for j in range(win):
                block_index = curSlot // self.block_size
                block_offset = curSlot % self.block_size
                curSlot += 1

                curBatch = i // self.num_heads
                bsID = self.new_seq[curBatch] - win + j
                headID = i % self.num_heads

                token_key = self.key[bsID][headID]
                token_v = self.value[bsID][headID]
                self.key_expect[block_index][block_offset] = token_key
                self.value_expect[block_index][block_offset] = token_v

        return self.key_expect, self.value_expect

    def golden_calc(self, in_tensors):
        tensor_out1, tensor_out2 = self.generate_test_data()
        logging.debug(f'kv_cache shape: {tensor_out1.shape}, {tensor_out2.shape}')
        if self.dtype == "bfloat16":
            return [torch.tensor(tensor_out1).bfloat16(), torch.tensor(tensor_out2).bfloat16()]
        else:
            return [torch.tensor(tensor_out1), torch.tensor(tensor_out2)]

    def golden_compare(self, out_tensors, golden_out_tensors):
        self.assertTrue(len(out_tensors) == len(golden_out_tensors))
        result = []
        batchID=0
        BS=self.seqLen[0]

        logging.debug("reshapeAndCacheWins GoldenTest")

        for i, slot in enumerate(self.slot_mapping):
            if slot < 0:
                continue
            curSlot = slot
            win = self.wins[i]
            for j in range(win):
                block_index = curSlot // self.block_size
                block_offset = curSlot % self.block_size

                curBatch = i // self.num_heads

                bsID = self.new_seq[curBatch] - win + j
                headID = i % self.num_heads

                token_key = self.key[bsID][headID]
                token_v = self.value[bsID][headID]
                self.key_expect[block_index][block_offset] = token_key
                self.value_expect[block_index][block_offset] = token_v
                curSlot += 1

        for i in range(len(out_tensors)):
            expect = golden_out_tensors[i]
            actual = out_tensors[i]
            abs_diff = torch.abs(expect-actual)
            result.append(not (abs_diff > 0).nonzero().size(0) > 0)

        return all(result)
    
    @op_test.only_910b
    def test_reshape_and_cache_case0(self):
        batch = 2
        num_heads = 4
        head_size = 128
        block_size = 128
        num_blocks = 512 * num_heads
        dtype = "int8"

        OP_NAME = "ReshapeAndCacheOperation"
        OP_PARAM = {"type": 2}
        self.set_reshape_and_cache_param(num_heads, head_size, block_size, num_blocks, dtype, batch)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 7)
        self.set_output_formats([self.format_nd] * 2)

        key = self.key
        value = self.value
        slot_mapping = self.slot_mapping
        key_cache = self.key_expect
        value_cache = self.value_expect
        wins = self.wins
        seqLen = self.seqLen

        return self.execute([torch.tensor(key), torch.tensor(value), torch.tensor(key_cache),
                             torch.tensor(value_cache), torch.tensor(slot_mapping), torch.tensor(wins), torch.tensor(seqLen)],
                             [2, 3])
        
    @op_test.only_910b
    def test_reshape_and_cache_case1(self):
        batch = 1
        num_heads = 4
        head_size = 128
        block_size = 128
        num_blocks = 1024
        dtype = "float16"

        OP_NAME = "ReshapeAndCacheOperation"
        OP_PARAM = {"type": 2}
        self.set_reshape_and_cache_param(num_heads, head_size, block_size, num_blocks, dtype, batch)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 7)
        self.set_output_formats([self.format_nd] * 2)

        key = self.key
        value = self.value
        slot_mapping = self.slot_mapping
        key_cache = self.key_expect
        value_cache = self.value_expect
        wins = self.wins
        seqLen = self.seqLen

        return self.execute([torch.tensor(key), torch.tensor(value), torch.tensor(key_cache),
                             torch.tensor(value_cache), torch.tensor(slot_mapping), torch.tensor(wins), torch.tensor(seqLen)],
                             [2, 3])
        
    @op_test.only_910b
    def test_reshape_and_cache_case2(self):
        batch = 1
        num_heads = 4
        head_size = 128
        block_size = 128
        num_blocks = 1024
        dtype = "bfloat16"

        OP_NAME = "ReshapeAndCacheOperation"
        OP_PARAM = {"type": 2}
        self.set_reshape_and_cache_param(num_heads, head_size, block_size, num_blocks, dtype, batch)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 7)
        self.set_output_formats([self.format_nd] * 2)

        key = self.key
        value = self.value
        slot_mapping = self.slot_mapping
        key_cache = self.key_expect
        value_cache = self.value_expect
        wins = self.wins
        seqLen = self.seqLen

        return self.execute([torch.tensor(key).bfloat16(), torch.tensor(value).bfloat16(), torch.tensor(key_cache).bfloat16(),
                             torch.tensor(value_cache).bfloat16(), torch.tensor(slot_mapping), torch.tensor(wins), torch.tensor(seqLen)],
                             [2, 3])
if __name__ == '__main__':
    unittest.main()