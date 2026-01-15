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

import op_test

class TestReshapeAndCacheNz(op_test.OpTest):
    def setUp(self):
        super().setUp()

    def set_reshape_and_cache_nz_param(self, num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype):
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.head_size_k = head_size_k
        self.head_size_v = head_size_v
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.dtype = dtype
        if (dtype == "float16"):
            self.last_dim = 16
        elif (dtype == "int8"):
            self.last_dim = 32
        self.key = np.random.uniform(-5.0, 5.0, size=(self.num_tokens, self.num_heads, self.head_size_k)).astype(self.dtype)
        self.value = np.random.uniform(-5.0, 5.0, size=(self.num_tokens, self.num_heads, self.head_size_v)).astype("float16")
        num_slots = self.block_size * self.num_blocks
        slot_list = random.sample(range(-num_slots, num_slots), self.num_tokens)
        self.slot_mapping = np.array(slot_list).astype(np.int32)
        self.slot_mapping[0] = 0

    def generate_test_data(self):
        key_expect_nz = np.zeros((self.num_blocks, self.num_heads * self.head_size_k // self.last_dim, self.block_size, self.last_dim)).astype(self.dtype)
        value_expect_nz = np.zeros((self.num_blocks, self.num_heads * self.head_size_v // 16, self.block_size, 16)).astype("float16")

        for i, slot in enumerate(self.slot_mapping):
            if slot < 0:
                continue
            block_index = slot // self.block_size
            block_offset = slot % self.block_size

            token_key = self.key[i]
            token_v = self.value[i]
            token_key = token_key.reshape(self.num_heads * self.head_size_k)
            token_v = token_v.reshape(self.num_heads * self.head_size_v)
            for k in range(self.num_heads * self.head_size_k // self.last_dim):
                key_expect_nz[block_index][k][block_offset][:] = token_key[k * self.last_dim: k * self.last_dim + self.last_dim]
            for k in range(self.num_heads * self.head_size_v // 16):
                value_expect_nz[block_index][k][block_offset][:] = token_v[k * 16: k * 16 + 16]
        return key_expect_nz, value_expect_nz

    def golden_calc(self, in_tensors):
        tensor_out1, tensor_out2 = self.generate_test_data()
        logging.debug(f'kv_cache shape: , {tensor_out1.shape}, {tensor_out2.shape}')
        return [torch.tensor(tensor_out1), torch.tensor(tensor_out2)]

    def golden_compare(self, out_tensors, golden_out_tensors):
        self.assertTrue(len(out_tensors) == len(golden_out_tensors))
        result = []
        for i in range(len(out_tensors)):
            actual_output = out_tensors[i]
            golden_output = golden_out_tensors[i]
            result.append(torch.allclose(actual_output.half(), golden_output.half(), rtol=0.001, atol=0.001))
        logging.debug(f"result is : {all(result)}")
        return all(result)
    
    def __run_reshape_and_cache_case(self, batch, seq_len, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype):
        num_tokens = batch * seq_len

        OP_NAME = "ReshapeAndCacheOperation"
        OP_PARAM = {"type": 1}
        self.set_reshape_and_cache_nz_param(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nz, self.format_nz, self.format_nd])

        key = self.key
        value = self.value
        slot_mapping = self.slot_mapping
        key_cache_nz = np.zeros((num_blocks, num_heads * head_size_k // self.last_dim, block_size, self.last_dim)).astype(self.dtype)
        value_cache_nz = np.zeros((num_blocks, num_heads * head_size_v // 16, block_size, 16)).astype("float16")

        self.execute([torch.tensor(key), torch.tensor(value), torch.tensor(key_cache_nz),
                      torch.tensor(value_cache_nz), torch.tensor(slot_mapping)], [2, 3])


    def test_reshape_and_cache_nz_case0(self):
        batch = 6144
        seq_len = 1
        num_heads = 1
        head_size_k = 512
        head_size_v = 64
        block_size = 128
        num_blocks = 64
        dtype = "int8"
        self.__run_reshape_and_cache_case(batch, seq_len, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)

    def test_reshape_and_cache_nz_case1(self):
        batch = 8192
        seq_len = 1
        num_heads = 1
        head_size_k = 128
        head_size_v = 128
        block_size = 128
        num_blocks = 96
        dtype = "int8"
        self.__run_reshape_and_cache_case(batch, seq_len, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)
    
    def test_reshape_and_cache_nz_case2(self):
        batch = 6144
        seq_len = 1
        num_heads = 1
        head_size_k = 512
        head_size_v = 64
        block_size = 128
        num_blocks = 64
        dtype = "float16"
        self.__run_reshape_and_cache_case(batch, seq_len, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)

    def test_reshape_and_cache_nz_case3(self):
        batch = 8192
        seq_len = 1
        num_heads = 1
        head_size_k = 128
        head_size_v = 128
        block_size = 128
        num_blocks = 96
        dtype = "float16"
        self.__run_reshape_and_cache_case(batch, seq_len, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)

    @op_test.only_310b
    def test_reshape_and_cache_nz_case4(self):
        batch = 2
        seq_len = 1
        num_heads = 1
        head_size_k = 64
        head_size_v = 64
        block_size = 128
        num_blocks = 3
        dtype = "float16"
        self.__run_reshape_and_cache_case(batch, seq_len, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)

    @op_test.only_310b
    def test_reshape_and_cache_nz_case5(self):
        batch = 8
        seq_len = 1
        num_heads = 1
        head_size_k = 128
        head_size_v = 128
        block_size = 128
        num_blocks = 3
        dtype = "float16"
        self.__run_reshape_and_cache_case(batch, seq_len, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)

    @unittest.skip("not for pipeline")
    def atest_reshape_and_cache_nz_big_batch(self):
        batch = 200
        seq_len = 2048
        num_heads = 1
        head_size = 16
        block_size = 16
        num_blocks = 200 * 2048
        dtype = "float16"
        self.__run_reshape_and_cache_case(batch, seq_len, num_heads, head_size, head_size, block_size, num_blocks, dtype)




if __name__ == '__main__':
    unittest.main()