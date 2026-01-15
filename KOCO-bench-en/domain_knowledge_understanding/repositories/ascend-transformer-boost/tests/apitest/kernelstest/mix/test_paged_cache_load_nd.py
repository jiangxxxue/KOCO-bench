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
random.seed(3)

MAX_SEQ_LEN = 1024
class TestPagedCacheLoad(op_test.OpTest):
    def setUp(self):
        super().setUp()

    def set_paged_cache_load_param(self, num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype):
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.head_size_k = head_size_k
        self.head_size_v = head_size_v
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.dtype = dtype

        if self.dtype == "float16":
            self.key_cache = np.random.randint(1, 11, size=(num_blocks, block_size, num_heads, head_size_k)).astype(np.float16)
            self.value_cache = np.random.randint(1, 11, size=(num_blocks, block_size, num_heads, head_size_v)).astype(np.float16)
        elif self.dtype == "bfloat16":
            self.key_cache = np.random.randint(1, 11, size=(num_blocks, block_size, num_heads, head_size_k)).astype(np.float32)
            self.value_cache = np.random.randint(1, 11, size=(num_blocks, block_size, num_heads, head_size_v)).astype(np.float32)
        else:
            self.key_cache = np.random.randint(1, 11, size=(num_blocks, block_size, num_heads, head_size_k)).astype(self.dtype)
            self.value_cache = np.random.randint(1, 11, size=(num_blocks, block_size, num_heads, head_size_v)).astype(self.dtype)

        context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_tokens)]
        max_context_len = max(context_lens)
        cu_context_lens = [0]
        for elem in context_lens:
            cu_context_lens.append(cu_context_lens[-1] + elem) 


        max_num_blocks_per_req = (max_context_len + block_size - 1) // block_size + 4
        block_tables = []   # [num_tokens, max_num_blocks_per_seq]
        for _ in range(num_tokens):
            block_table = [
                random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_req)
            ]
            block_tables.append(block_table)

        seq_starts = [random.randint(0, 4) * block_size for _ in range(num_tokens)]

        self.context_lens = np.array(cu_context_lens).astype(np.int32)
        self.block_tables = np.array(block_tables).astype(np.int32)
        self.seq_starts = np.array(seq_starts).astype(np.int32)

    def generate_test_data(self):

        num_heads = self.num_heads
        num_tokens = self.num_tokens
        head_size_k = self.head_size_k
        head_size_v = self.head_size_v
        block_size = self.block_size
        block_tables = self.block_tables
        context_lens = self.context_lens
        seq_starts = self.seq_starts
        key_cache = self.key_cache
        value_cache = self.value_cache
        sum_context_lens = context_lens[-1]
        
        if self.dtype == "float16":
            key_expect = np.zeros((sum_context_lens, num_heads, head_size_k)).astype(np.float16)
            value_expect = np.zeros((sum_context_lens, num_heads, head_size_v)).astype(np.float16)
        elif self.dtype == "bfloat16":
            key_expect = np.zeros((sum_context_lens, num_heads, head_size_k)).astype(np.float32)
            value_expect = np.zeros((sum_context_lens, num_heads, head_size_v)).astype(np.float32)
        else:
            key_expect = np.zeros((sum_context_lens, num_heads, head_size_k)).astype(self.dtype)
            value_expect = np.zeros((sum_context_lens, num_heads, head_size_v)).astype(self.dtype)

        kv_rslt_id = 0
        context_start = 0
        for i in range(num_tokens):
            block_table = block_tables[i]
            context_end = int(context_lens[i+1])
            context_len = context_end - context_start
            context_start = context_end
            block_table_offset = seq_starts[i] // block_size
            for j in range(context_len):
                block_id = int(block_table[block_table_offset + j // block_size])
                block_offset = j % block_size

                if block_id < 0:
                    continue

                temp_k = key_cache[block_id][block_offset]
                temp_v = value_cache[block_id][block_offset]

                key_expect[kv_rslt_id] = temp_k
                value_expect[kv_rslt_id] = temp_v
                kv_rslt_id += 1

        return key_expect, value_expect

    def golden_calc(self, in_tensors):
        tensor_out1, tensor_out2 = self.generate_test_data()
        logging.info(f'kv shape: {tensor_out1.shape}, {tensor_out2.shape}')
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
            logging.info("float16 GoldenTest")
            for i in range(len(out_tensors)):
                actual_output = out_tensors[i]
                golden_output = golden_out_tensors[i]
                result.append(torch.equal(actual_output.half(), golden_output.half()))
        elif self.dtype == "int8":
            logging.info("int8 GoldenTest")
            for i in range(len(out_tensors)):
                actual_output = out_tensors[i].to(torch.int8)
                golden_output = golden_out_tensors[i].to(torch.int8)
                result.append(torch.equal(actual_output, golden_output))
        elif self.dtype == "bfloat16":
            logging.info("bfloat16 GoldenTest")
            for i in range(len(out_tensors)):
                actual_output = out_tensors[i]
                golden_output = golden_out_tensors[i]
                result.append(torch.equal(actual_output.bfloat16(), golden_output.bfloat16()))
        else:
            logging.info("Unsupport dtype:%s golden compare", self.dtype)
            result.append(False)
        logging.info(f"result is : {all(result)}")
        return all(result)

    @op_test.only_910b
    def test_paged_cache_load_case0(self):
        batch = 16
        seq_len = 1
        num_heads = 16
        head_size_k = 144
        head_size_v = 128
        block_size = 128
        num_blocks = 128
        dtype = "float16"

        num_tokens = batch * seq_len
        OP_NAME = "PagedCacheLoadOperation"
        self.set_paged_cache_load_param(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)

        OP_PARAM = {"type": 0, "cuSeqLens": True, "hasSeqStarts": True}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd] * 2)

        key_cache = self.key_cache
        value_cache = self.value_cache
        block_tables = self.block_tables
        context_lens = self.context_lens
        seq_starts = self.seq_starts
        sum_context_lens = context_lens[-1]
        key = np.zeros((sum_context_lens, num_heads, head_size_k)).astype(np.float16)
        value = np.zeros((sum_context_lens, num_heads, head_size_v)).astype(np.float16)
        self.execute([torch.tensor(key_cache).half(), torch.tensor(value_cache).half(),
                    torch.tensor(block_tables).int(), torch.tensor(context_lens).int(),
                    torch.tensor(key).half(), torch.tensor(value).half(),torch.tensor(seq_starts).int()], [4, 5])

    @op_test.only_910b
    def test_paged_cache_load_case1(self):
        batch = 16
        seq_len = 1
        num_heads = 16
        head_size_k = 144
        head_size_v = 128
        block_size = 128
        num_blocks = 128
        dtype = "bfloat16"

        num_tokens = batch * seq_len
        OP_NAME = "PagedCacheLoadOperation"
        self.set_paged_cache_load_param(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)

        OP_PARAM = {"type": 0, "cuSeqLens": True, "hasSeqStarts": True}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd] * 2)

        key_cache = self.key_cache
        value_cache = self.value_cache
        block_tables = self.block_tables
        context_lens = self.context_lens
        seq_starts = self.seq_starts
        sum_context_lens = context_lens[-1]
        key = np.zeros((sum_context_lens, num_heads, head_size_k)).astype(np.float32)
        value = np.zeros((sum_context_lens, num_heads, head_size_v)).astype(np.float32)
        self.execute([torch.tensor(key_cache).bfloat16(), torch.tensor(value_cache).bfloat16(),
                    torch.tensor(block_tables).int(), torch.tensor(context_lens).int(),
                    torch.tensor(key).bfloat16(), torch.tensor(value).bfloat16(),torch.tensor(seq_starts).int()], [4, 5])

    @op_test.only_910b
    def test_paged_cache_load_case2(self):
        batch = 16
        seq_len = 1
        num_heads = 16
        head_size_k = 144
        head_size_v = 128
        block_size = 128
        num_blocks = 128
        dtype = "int8"

        num_tokens = batch * seq_len
        OP_NAME = "PagedCacheLoadOperation"
        self.set_paged_cache_load_param(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)

        OP_PARAM = {"type": 0, "cuSeqLens": True, "hasSeqStarts": True}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd] * 2)

        key_cache = self.key_cache
        value_cache = self.value_cache
        block_tables = self.block_tables
        context_lens = self.context_lens
        seq_starts = self.seq_starts
        sum_context_lens = context_lens[-1]
        key = np.zeros((sum_context_lens, num_heads, head_size_k)).astype(self.dtype)
        value = np.zeros((sum_context_lens, num_heads, head_size_v)).astype(self.dtype)
        self.execute([torch.tensor(key_cache), torch.tensor(value_cache),
                    torch.tensor(block_tables).int(), torch.tensor(context_lens).int(),
                    torch.tensor(key), torch.tensor(value),torch.tensor(seq_starts).int()], [4, 5])

    @op_test.only_910b
    def test_paged_cache_load_case3(self):
        batch = 32
        seq_len = 1
        num_heads = 16
        head_size_k = 192
        head_size_v = 128
        block_size = 64
        num_blocks = 256
        dtype = "float16"

        num_tokens = batch * seq_len
        OP_NAME = "PagedCacheLoadOperation"
        self.set_paged_cache_load_param(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)

        OP_PARAM = {"type": 0, "cuSeqLens": True, "hasSeqStarts": True}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd] * 2)

        key_cache = self.key_cache
        value_cache = self.value_cache
        block_tables = self.block_tables
        context_lens = self.context_lens
        seq_starts = self.seq_starts
        sum_context_lens = context_lens[-1]
        key = np.zeros((sum_context_lens, num_heads, head_size_k)).astype(np.float16)
        value = np.zeros((sum_context_lens, num_heads, head_size_v)).astype(np.float16)
        self.execute([torch.tensor(key_cache).half(), torch.tensor(value_cache).half(),
                    torch.tensor(block_tables).int(), torch.tensor(context_lens).int(),
                    torch.tensor(key).half(), torch.tensor(value).half(),torch.tensor(seq_starts).int()], [4, 5])

    @op_test.only_910b
    def test_paged_cache_load_case4(self):
        batch = 32
        seq_len = 1
        num_heads = 16
        head_size_k = 192
        head_size_v = 128
        block_size = 64
        num_blocks = 256
        dtype = "bfloat16"

        num_tokens = batch * seq_len
        OP_NAME = "PagedCacheLoadOperation"
        self.set_paged_cache_load_param(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)

        OP_PARAM = {"type": 0, "cuSeqLens": True, "hasSeqStarts": True}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd] * 2)

        key_cache = self.key_cache
        value_cache = self.value_cache
        block_tables = self.block_tables
        context_lens = self.context_lens
        seq_starts = self.seq_starts
        sum_context_lens = context_lens[-1]
        key = np.zeros((sum_context_lens, num_heads, head_size_k)).astype(np.float32)
        value = np.zeros((sum_context_lens, num_heads, head_size_v)).astype(np.float32)
        self.execute([torch.tensor(key_cache).bfloat16(), torch.tensor(value_cache).bfloat16(),
                    torch.tensor(block_tables).int(), torch.tensor(context_lens).int(),
                    torch.tensor(key).bfloat16(), torch.tensor(value).bfloat16(),torch.tensor(seq_starts).int()], [4, 5])

    @op_test.only_910b
    def test_paged_cache_load_case5(self):
        batch = 32
        seq_len = 1
        num_heads = 16
        head_size_k = 192
        head_size_v = 128
        block_size = 64
        num_blocks = 256
        dtype = "int8"

        num_tokens = batch * seq_len
        OP_NAME = "PagedCacheLoadOperation"
        self.set_paged_cache_load_param(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)

        OP_PARAM = {"type": 0, "cuSeqLens": True, "hasSeqStarts": True}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd] * 2)

        key_cache = self.key_cache
        value_cache = self.value_cache
        block_tables = self.block_tables
        context_lens = self.context_lens
        seq_starts = self.seq_starts
        sum_context_lens = context_lens[-1]
        key = np.zeros((sum_context_lens, num_heads, head_size_k)).astype(self.dtype)
        value = np.zeros((sum_context_lens, num_heads, head_size_v)).astype(self.dtype)
        self.execute([torch.tensor(key_cache), torch.tensor(value_cache),
                    torch.tensor(block_tables).int(), torch.tensor(context_lens).int(),
                    torch.tensor(key), torch.tensor(value),torch.tensor(seq_starts).int()], [4, 5])


if __name__ == '__main__':
    unittest.main()