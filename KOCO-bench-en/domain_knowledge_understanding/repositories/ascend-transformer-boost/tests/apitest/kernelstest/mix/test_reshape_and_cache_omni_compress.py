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
sys.path.append('../')
import logging
np.set_printoptions(threshold=np.inf)
from copy import deepcopy

import op_test
from tensor_file import read_tensor
class TestReshapeAndCache(op_test.OpTest):
    def setUp(self):
        super().setUp()

    def genKVData(self, a, b, c): # a: BS, b: N, c: D
        tensor_list = [torch.full((1, c), i) for i in range(b*a)]
        conbined_list = []
        for i in range (a):
            left = b*i
            right = b*(i+1)
            combined_tensor = torch.cat(tensor_list[left:right], dim=0)
            conbined_list.append(combined_tensor)
        combined_tensor = torch.stack(conbined_list, dim=0)
        return combined_tensor.numpy()

    def set_reshape_and_cache_param(self, num_heads, head_size, block_size, num_blocks, dtype, batch):
        self.num_heads = num_heads
        self.head_size = head_size
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.dtype = dtype
        self.batch = batch

        len = random.randint(1024,4096)
        win = random.randint(0, len)
        sub = len - win
        offset = random.randint(-1, sub)
        self.seqLen = torch.randint(low=len,high=len+1,size =[self.batch]).to(torch.int32)
        self.num_tokens = torch.sum(self.seqLen)
        self.wins = torch.randint(low=win, high=win+1, size=[self.batch*(self.num_heads)]).to(torch.int32)
        self.idxOffset = torch.randint(low=offset, high=offset+1, size=[self.batch*self.num_heads]).to(torch.int32)

        k = np.random.uniform(low=-5, high=5, size=(self.num_tokens, self.num_heads, self.head_size)).astype(np.float32)
        v = np.random.uniform(low=-5, high=5, size=(self.num_tokens, self.num_heads, self.head_size)).astype(np.float32)
        if dtype == "bfloat16":
            self.key = torch.from_numpy(k).to(torch.bfloat16)
            self.value = torch.from_numpy(v).to(torch.bfloat16)
            self.key_expect = torch.zeros(size = (self.num_blocks, self.block_size, 1, self.head_size), dtype=torch.bfloat16)
            self.value_expect = torch.zeros(size = (self.num_blocks, self.block_size, 1, self.head_size), dtype=torch.bfloat16)
        if dtype == "half":
            self.key = torch.from_numpy(k).to(torch.half)
            self.value = torch.from_numpy(v).to(torch.half)
            self.key_expect = torch.zeros(size = (self.num_blocks, self.block_size, 1, self.head_size), dtype=torch.half)
            self.value_expect = torch.zeros(size = (self.num_blocks, self.block_size, 1, self.head_size), dtype=torch.half)
        

        cacheStart = []
        cacheOffset = 0
        for i in range(self.batch * self.num_heads):
            cacheStart.append(cacheOffset)
            if (self.idxOffset[i] != -1) :
                cacheOffset += self.seqLen[int(i/self.num_heads)].item() - self.wins[i].item()
            else :
                cacheOffset += self.seqLen[int(i/self.num_heads)].item()
        self.slot_mapping = torch.Tensor(cacheStart).to(torch.int32)

    def generate_test_data(self):
        key_expect_fp32 = self.key_expect.clone().to(torch.float32)
        value_expect_fp32 = self.value_expect.clone().to(torch.float32)

        self.new_seq = self.seqLen.clone()
        self.new_seq[0] = self.seqLen[0]
        for n in range(1, len(self.seqLen)):
            self.new_seq[n] = self.seqLen[n] + self.new_seq[n-1]
        self.new_seq = torch.cat((torch.zeros(1, dtype=torch.int32), self.new_seq), dim=0)

        for i, slot in enumerate(self.slot_mapping):
            if slot < 0:
                continue
            win = self.wins[i].clone()
            idxOffset = self.idxOffset[i]

            bsID = i // self.num_heads
            headID = i % self.num_heads
            headStartIdx = self.new_seq[bsID]
            headEndIdx = idxOffset + self.wins[i]
            bs = self.new_seq[bsID]

            sum_key = torch.zeros(self.head_size, dtype = torch.float32)
            sum_value = torch.zeros(self.head_size, dtype = torch.float32)
            for j in range(self.seqLen[bsID]):
                block_index = torch.div(slot, self.block_size, rounding_mode='trunc')
                block_offset = slot % self.block_size

                token_key = self.key[bs+j][headID]
                token_v = self.value[bs+j][headID]
                if idxOffset == -1 or (j < idxOffset and idxOffset != -1):
                    key_expect_fp32[block_index][block_offset] = token_key
                    value_expect_fp32[block_index][block_offset] = token_v
                    slot+=1
                    
                if j >= headEndIdx and idxOffset != -1:
                    key_expect_fp32[block_index][block_offset] = token_key.to(torch.float32)
                    value_expect_fp32[block_index][block_offset] = token_v.to(torch.float32)
                    slot+=1

        return key_expect_fp32, value_expect_fp32

    def golden_calc(self, in_tensors):
        tensor_out1, tensor_out2 = self.generate_test_data()
        logging.debug(f'kv_cache shape: {tensor_out1.shape}, {tensor_out2.shape}')
        return [tensor_out1, tensor_out2]

    def golden_compare(self, out_tensors, golden_out_tensors):
        self.assertTrue(len(out_tensors) == len(golden_out_tensors))
        logging.debug("reshapeAndCacheWins for RoPE GoldenTest")

        err = 2**(-8)
        if out_tensors[0].dtype == torch.bfloat16:
            err = 2**(-7)
        elif out_tensors[0].dtype == torch.float32:
            err = 2**(-14)     
        
        out_tensors[0] = out_tensors[0].to(torch.float32)
        out_tensors[1] = out_tensors[1].to(torch.float32)
        kdiff = torch.abs(torch.subtract(out_tensors[0], golden_out_tensors[0]))
        tensor_max = torch.maximum(torch.ones(golden_out_tensors[0].shape, dtype=golden_out_tensors[0].dtype), torch.abs(golden_out_tensors[0]))
        if torch.any(torch.greater(kdiff, err * tensor_max)):
            return False
        vdiff = torch.abs(torch.subtract(out_tensors[1], golden_out_tensors[1]))
        tensor_max = torch.maximum(torch.ones(golden_out_tensors[1].shape, dtype=golden_out_tensors[1].dtype), torch.abs(golden_out_tensors[1]))
        if torch.any(torch.greater(vdiff, err * tensor_max)):
            return False
        return True
    
    @op_test.only_910b
    def test_reshape_and_cache_single_batch_single_head(self): # 单batch 单head
        batch = 1
        num_heads = 1
        head_size = 256
        block_size = 256
        num_blocks = 2048
        dtype = "bfloat16"

        OP_NAME = "ReshapeAndCacheOperation"
        OP_PARAM = {"type": 5}
        self.set_reshape_and_cache_param(num_heads, head_size, block_size, num_blocks, dtype, batch)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 8)
        self.set_output_formats([self.format_nd] * 2)

        key = self.key
        value = self.value
        slot_mapping = self.slot_mapping
        key_cache = self.key_expect
        value_cache = self.value_expect
        wins = self.wins
        seqLen = self.seqLen
        idxOffset = self.idxOffset
        self.execute(
            [key, value, key_cache, value_cache, slot_mapping, wins, seqLen, idxOffset],
            [2, 3])

    @op_test.only_910b
    def test_reshape_and_cache_multi_batch_single_head(self): # 多batch 单head
        batch = 40
        num_heads = 1
        head_size = 96
        block_size = 128
        num_blocks = 8192
        dtype = "half"

        OP_NAME = "ReshapeAndCacheOperation"
        OP_PARAM = {"type": 5}
        self.set_reshape_and_cache_param(num_heads, head_size, block_size, num_blocks, dtype, batch)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 8)
        self.set_output_formats([self.format_nd] * 2)

        key = self.key
        value = self.value
        slot_mapping = self.slot_mapping
        key_cache = self.key_expect
        value_cache = self.value_expect
        wins = self.wins
        seqLen = self.seqLen
        idxOffset = self.idxOffset
        self.execute(
            [key, value, key_cache, value_cache, slot_mapping, wins, seqLen, idxOffset],
            [2, 3])

    @op_test.only_910b
    def test_reshape_and_cache_single_batch_multi_head(self): # 单batch 多head
        batch = 1
        num_heads = 24
        head_size = 64
        block_size = 64
        num_blocks = 8192
        dtype = "bfloat16"

        OP_NAME = "ReshapeAndCacheOperation"
        OP_PARAM = {"type": 5}
        self.set_reshape_and_cache_param(num_heads, head_size, block_size, num_blocks, dtype, batch)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 8)
        self.set_output_formats([self.format_nd] * 2)

        key = self.key
        value = self.value
        slot_mapping = self.slot_mapping
        key_cache = self.key_expect
        value_cache = self.value_expect
        wins = self.wins
        seqLen = self.seqLen
        idxOffset = self.idxOffset
        self.execute(
            [key, value, key_cache, value_cache, slot_mapping, wins, seqLen, idxOffset],
            [2, 3])

    @op_test.only_910b
    def test_reshape_and_cache_multi_batch_multi_head(self): # 多batch 多head
        batch = 8
        num_heads = 8
        head_size = 32
        block_size = 128
        num_blocks = 8192
        dtype = "half"

        OP_NAME = "ReshapeAndCacheOperation"
        OP_PARAM = {"type": 5}
        self.set_reshape_and_cache_param(num_heads, head_size, block_size, num_blocks, dtype, batch)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 8)
        self.set_output_formats([self.format_nd] * 2)

        key = self.key
        value = self.value
        slot_mapping = self.slot_mapping
        key_cache = self.key_expect
        value_cache = self.value_expect
        wins = self.wins
        seqLen = self.seqLen
        idxOffset = self.idxOffset
        self.execute(
            [key, value, key_cache, value_cache, slot_mapping, wins, seqLen, idxOffset],
            [2, 3])

if __name__ == '__main__':
    unittest.main()