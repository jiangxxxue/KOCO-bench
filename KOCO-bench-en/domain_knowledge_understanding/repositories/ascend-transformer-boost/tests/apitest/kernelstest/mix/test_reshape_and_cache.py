# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import unittest
import numpy as np
import torch
import torch_npu
import random
import sys
import logging
import op_test

class TestReshapeAndCache(op_test.OpTest):
    def setUp(self):
        super().setUp()

    def set_reshape_and_cache_param(self, num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype):
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.head_size_k = head_size_k
        self.head_size_v = head_size_v
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.dtype = dtype
        if self.dtype == "float16":
            self.key = np.random.randint(1, 11, size=(self.num_tokens, self.num_heads, self.head_size_k)).astype(np.float16)
            self.value = np.random.randint(1, 11, size=(self.num_tokens, self.num_heads, self.head_size_v)).astype(np.float16)
        elif self.dtype == "bfloat16":
            self.key = np.random.randint(1, 11, size=(self.num_tokens, self.num_heads, self.head_size_k)).astype(np.float32)
            self.value = np.random.randint(1, 11, size=(self.num_tokens, self.num_heads, self.head_size_v)).astype(np.float32)
        else:
            self.key = np.random.randint(1, 11, size=(self.num_tokens, self.num_heads, self.head_size_k)).astype(self.dtype)
            self.value = np.random.randint(1, 11, size=(self.num_tokens, self.num_heads, self.head_size_v)).astype(self.dtype)

        num_slots = self.block_size * self.num_blocks
        slot_list = random.sample(range(-num_slots, num_slots), self.num_tokens)
        self.slot_mapping = np.array(slot_list).astype(np.int32)

    def generate_test_data(self):
        if self.dtype == "float16":
            key_expect = np.zeros((self.num_blocks, self.block_size, self.num_heads, self.head_size_k)).astype(np.float16)
            value_expect = np.zeros((self.num_blocks, self.block_size, self.num_heads, self.head_size_v)).astype(np.float16)
        elif self.dtype == "bfloat16":
            key_expect = np.zeros((self.num_blocks, self.block_size, self.num_heads, self.head_size_k)).astype(np.float32)
            value_expect = np.zeros((self.num_blocks, self.block_size, self.num_heads, self.head_size_v)).astype(np.float32)
        else:
            key_expect = np.zeros((self.num_blocks, self.block_size, self.num_heads, self.head_size_k)).astype(self.dtype)
            value_expect = np.zeros((self.num_blocks, self.block_size, self.num_heads, self.head_size_v)).astype(self.dtype)

        for i, slot in enumerate(self.slot_mapping):
            if slot < 0:
                continue
            block_index = slot // self.block_size
            block_offset = slot % self.block_size

            token_key = self.key[i]
            token_v = self.value[i]
            key_expect[block_index][block_offset] = token_key
            value_expect[block_index][block_offset] = token_v
        return key_expect, value_expect

    def golden_calc(self, in_tensors):
        tensor_out1, tensor_out2 = self.generate_test_data()
        logging.debug(f'kv_cache shape: {tensor_out1.shape}, {tensor_out2.shape}')
        if self.dtype == "float16":
            return [torch.tensor(tensor_out1).half(), torch.tensor(tensor_out2).half()]
        elif self.dtype == "bfloat16":
            return [torch.tensor(tensor_out1).bfloat16(), torch.tensor(tensor_out2).bfloat16()]
        else:
            return [torch.tensor(tensor_out1), torch.tensor(tensor_out2)]

    def golden_compare(self, out_tensors, golden_out_tensors):
        self.assertTrue(len(out_tensors) == len(golden_out_tensors))
        result = []
        if self.dtype == "float16":
            logging.debug("float16 GoldenTest")
            for i in range(len(out_tensors)):
                actual_output = out_tensors[i]
                golden_output = golden_out_tensors[i]
                result.append(torch.equal(actual_output.half(), golden_output.half()))
        elif self.dtype == "int8":
            logging.debug("int8 GoldenTest")
            for i in range(len(out_tensors)):
                actual_output = out_tensors[i].to(torch.int8)
                golden_output = golden_out_tensors[i].to(torch.int8)
                result.append(torch.equal(actual_output, golden_output))
        elif self.dtype == "bfloat16":
            logging.debug("bfloat16 GoldenTest")
            for i in range(len(out_tensors)):
                actual_output = out_tensors[i]
                golden_output = golden_out_tensors[i]
                result.append(torch.equal(actual_output.bfloat16(), golden_output.bfloat16()))
        else:
            logging.debug("Unsupport dtype:%s golden compare", self.dtype)
            result.append(False)
        logging.debug(f"result is : {all(result)}")
        return all(result)

    @op_test.only_910b
    def test_reshape_and_cache_case0(self):
        batch = 100
        seq_len = 1
        num_heads = 64
        head_size_k = 576
        head_size_v = 512
        block_size = 128
        num_blocks = 512
        dtype = "int8"

        num_tokens = batch * seq_len
        OP_NAME = "ReshapeAndCacheOperation"
        OP_PARAM = {"type": 0}
        self.set_reshape_and_cache_param(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 5)
        self.set_output_formats([self.format_nd] * 2)

        key = self.key
        value = self.value
        slot_mapping = self.slot_mapping
        key_cache = np.zeros((num_blocks, block_size, num_heads, head_size_k)).astype(self.dtype)
        value_cache = np.zeros((num_blocks, block_size, num_heads, head_size_v)).astype(self.dtype)

        return self.execute([torch.tensor(key), torch.tensor(value), torch.tensor(key_cache),
                             torch.tensor(value_cache), torch.tensor(slot_mapping)], [2, 3])
    
    @op_test.only_910b
    def test_reshape_and_cache_case1(self):
        batch = 100
        seq_len = 1
        num_heads = 64
        head_size_k = 320
        head_size_v = 256
        block_size = 128
        num_blocks = 512
        dtype = "float16"

        num_tokens = batch * seq_len
        OP_NAME = "ReshapeAndCacheOperation"
        OP_PARAM = {"type": 0}
        self.set_reshape_and_cache_param(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 5)
        self.set_output_formats([self.format_nd] * 2)

        key = self.key
        value = self.value
        slot_mapping = self.slot_mapping
        key_cache = np.zeros((num_blocks, block_size, num_heads, head_size_k)).astype(np.float16)
        value_cache = np.zeros((num_blocks, block_size, num_heads, head_size_v)).astype(np.float16)

        return self.execute([torch.tensor(key).half(), torch.tensor(value).half(), torch.tensor(key_cache).half(),
                             torch.tensor(value_cache).half(), torch.tensor(slot_mapping)], [2, 3])

    # 非32位对齐输入
    @op_test.only_910b
    def test_reshape_and_cache_case2(self):
        batch = 100
        seq_len = 1
        num_heads = 64
        head_size_k = 400
        head_size_v = 200
        block_size = 128
        num_blocks = 512
        dtype = "bfloat16"

        num_tokens = batch * seq_len
        OP_NAME = "ReshapeAndCacheOperation"
        OP_PARAM = {"type": 0}
        self.set_reshape_and_cache_param(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 5)
        self.set_output_formats([self.format_nd] * 2)

        key = self.key
        value = self.value
        slot_mapping = self.slot_mapping
        key_cache = np.zeros((num_blocks, block_size, num_heads, head_size_k)).astype(np.float32)
        value_cache = np.zeros((num_blocks, block_size, num_heads, head_size_v)).astype(np.float32)

        return self.execute([torch.tensor(key).bfloat16(), torch.tensor(value).bfloat16(), torch.tensor(key_cache).bfloat16(),
                             torch.tensor(value_cache).bfloat16(), torch.tensor(slot_mapping)], [2, 3])

    @op_test.only_910b
    def test_reshape_and_cache_case3(self):
        batch = 1
        seq_len = 1
        num_heads = 64
        head_size_k = 576
        head_size_v = 576
        block_size = 128
        num_blocks = 512
        dtype = "int8"

        num_tokens = batch * seq_len
        OP_NAME = "ReshapeAndCacheOperation"
        OP_PARAM = {"type": 0}
        self.set_reshape_and_cache_param(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 5)
        self.set_output_formats([self.format_nd] * 2)

        key = self.key
        value = self.value
        slot_mapping = self.slot_mapping
        key_cache = np.zeros((num_blocks, block_size, num_heads, head_size_k)).astype(self.dtype)
        value_cache = np.zeros((num_blocks, block_size, num_heads, head_size_v)).astype(self.dtype)

        return self.execute([torch.tensor(key), torch.tensor(value), torch.tensor(key_cache),
                             torch.tensor(value_cache), torch.tensor(slot_mapping)], [2, 3])
    
    @op_test.only_910b
    def test_reshape_and_cache_case4(self):
        batch = 50
        seq_len = 1
        num_heads = 64
        head_size_k = 320
        head_size_v = 320
        block_size = 128
        num_blocks = 512
        dtype = "float16"

        num_tokens = batch * seq_len
        OP_NAME = "ReshapeAndCacheOperation"
        OP_PARAM = {"type": 0}
        self.set_reshape_and_cache_param(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 5)
        self.set_output_formats([self.format_nd] * 2)

        key = self.key
        value = self.value
        slot_mapping = self.slot_mapping
        key_cache = np.zeros((num_blocks, block_size, num_heads, head_size_k)).astype(np.float16)
        value_cache = np.zeros((num_blocks, block_size, num_heads, head_size_v)).astype(np.float16)

        return self.execute([torch.tensor(key).half(), torch.tensor(value).half(), torch.tensor(key_cache).half(),
                             torch.tensor(value_cache).half(), torch.tensor(slot_mapping)], [2, 3])

    # 非32位对齐输入
    @op_test.only_910b
    def test_reshape_and_cache_case5(self):
        batch = 1
        seq_len = 1
        num_heads = 64
        head_size_k = 400
        head_size_v = 400
        block_size = 128
        num_blocks = 512
        dtype = "bfloat16"

        num_tokens = batch * seq_len
        OP_NAME = "ReshapeAndCacheOperation"
        OP_PARAM = {"type": 0}
        self.set_reshape_and_cache_param(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 5)
        self.set_output_formats([self.format_nd] * 2)

        key = self.key
        value = self.value
        slot_mapping = self.slot_mapping
        key_cache = np.zeros((num_blocks, block_size, num_heads, head_size_k)).astype(np.float32)
        value_cache = np.zeros((num_blocks, block_size, num_heads, head_size_v)).astype(np.float32)

        return self.execute([torch.tensor(key).bfloat16(), torch.tensor(value).bfloat16(), torch.tensor(key_cache).bfloat16(),
                             torch.tensor(value_cache).bfloat16(), torch.tensor(slot_mapping)], [2, 3])

if __name__ == '__main__':
    unittest.main()