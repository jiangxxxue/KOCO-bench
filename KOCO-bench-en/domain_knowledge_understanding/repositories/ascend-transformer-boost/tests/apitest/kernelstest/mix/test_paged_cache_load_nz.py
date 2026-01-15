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
            self.key_cache = np.random.randint(1, 11, size=(num_blocks, num_heads * head_size_k // 16, block_size, 16)).astype(np.float16)
            self.value_cache = np.random.randint(1, 11, size=(num_blocks, num_heads * head_size_v // 16, block_size, 16)).astype(np.float16)
        elif self.dtype == "bfloat16":
            self.key_cache = np.random.randint(1, 11, size=(num_blocks, num_heads * head_size_k // 16, block_size, 16)).astype(np.float32)
            self.value_cache = np.random.randint(1, 11, size=(num_blocks, num_heads * head_size_v // 16, block_size, 16)).astype(np.float32)
        else:
            self.key_cache = np.random.randint(1, 11, size=(num_blocks, num_heads * head_size_k // 32, block_size, 32)).astype(self.dtype)
            self.value_cache = np.random.randint(1, 11, size=(num_blocks, num_heads * head_size_v // 32, block_size, 32)).astype(self.dtype)

        context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_tokens)]
        max_context_len = max(context_lens)

        max_num_blocks_per_req = (max_context_len + block_size - 1) // block_size
        block_tables = []   # [num_tokens, max_num_blocks_per_seq]
        for _ in range(num_tokens):
            block_table = [
                random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_req)
            ]
            block_tables.append(block_table)

        self.context_lens = np.array(context_lens).astype(np.int32)
        self.block_tables = np.array(block_tables).astype(np.int32)

    def generate_test_data(self):

        num_heads = self.num_heads
        num_tokens = self.num_tokens
        head_size_k = self.head_size_k
        head_size_v = self.head_size_v
        block_size = self.block_size
        block_tables = self.block_tables
        context_lens = self.context_lens
        key_cache = self.key_cache
        value_cache = self.value_cache
        sum_context_lens = sum(context_lens)
        
        if self.dtype == "float16":
            key_expect = np.zeros((sum_context_lens, num_heads * head_size_k)).astype(np.float16)
            value_expect = np.zeros((sum_context_lens, num_heads * head_size_v)).astype(np.float16)
        elif self.dtype == "bfloat16":
            key_expect = np.zeros((sum_context_lens, num_heads * head_size_k)).astype(np.float32)
            value_expect = np.zeros((sum_context_lens, num_heads * head_size_v)).astype(np.float32)
        else:
            key_expect = np.zeros((sum_context_lens, num_heads * head_size_k)).astype(self.dtype)
            value_expect = np.zeros((sum_context_lens, num_heads * head_size_v)).astype(self.dtype)

        elenum_aligned = 16
        if self.dtype == "float16" or self.dtype == "bfloat16":
            elenum_aligned = 16
        elif self.dtype == "int8":
            elenum_aligned = 32
        else:
            elenum_aligned = 16

        kv_rslt_id = 0

        for i in range(num_tokens):
            block_table = block_tables[i]
            context_len = int(context_lens[i])

            for j in range(context_len):
                block_id = int(block_table[j // block_size])
                block_offset = j % block_size

                if block_id < 0:
                    continue

                temp_k = np.zeros((self.num_heads * self.head_size_k))
                temp_v = np.zeros((self.num_heads * self.head_size_v))

                for k in range(num_heads * head_size_k // elenum_aligned):
                    temp_k[k * elenum_aligned: k * elenum_aligned + elenum_aligned] = key_cache[block_id][k][block_offset][:]

                for k in range(num_heads * head_size_v // elenum_aligned):
                    temp_v[k * elenum_aligned: k * elenum_aligned + elenum_aligned] = value_cache[block_id][k][block_offset][:]

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
        head_size_k = 576
        head_size_v = 512
        block_size = 128
        num_blocks = 128
        dtype = "float16"

        num_tokens = batch * seq_len
        OP_NAME = "PagedCacheLoadOperation"
        self.set_paged_cache_load_param(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)

        OP_PARAM = {"type": 1, "cuSeqLens": False, "hasSeqStarts": False}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nz, self.format_nz, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd] * 2)

        key_cache = self.key_cache
        value_cache = self.value_cache
        block_tables = self.block_tables
        context_lens = self.context_lens

        sum_context_lens = int(sum(context_lens))
        key = np.zeros((sum_context_lens, num_heads * head_size_k)).astype(np.float16)
        value = np.zeros((sum_context_lens, num_heads * head_size_v)).astype(np.float16)

        return self.execute([torch.tensor(key_cache).half(), torch.tensor(value_cache).half(),
                             torch.tensor(block_tables).int(), torch.tensor(context_lens).int(),
                             torch.tensor(key).half(), torch.tensor(value).half(), torch.tensor(context_lens).int()], [4, 5])

    @op_test.only_910b
    def test_paged_cache_load_case1(self):
        batch = 16
        seq_len = 1
        num_heads = 16
        head_size_k = 576
        head_size_v = 512
        block_size = 128
        num_blocks = 128
        dtype = "bfloat16"

        num_tokens = batch * seq_len
        OP_NAME = "PagedCacheLoadOperation"
        self.set_paged_cache_load_param(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)

        OP_PARAM = {"type": 1, "cuSeqLens": False, "hasSeqStarts": False}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nz, self.format_nz, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd] * 2)

        key_cache = self.key_cache
        value_cache = self.value_cache
        block_tables = self.block_tables
        context_lens = self.context_lens

        sum_context_lens = sum(context_lens)
        key = np.zeros((sum_context_lens, num_heads * head_size_k)).astype(np.float32)
        value = np.zeros((sum_context_lens, num_heads * head_size_v)).astype(np.float32)

        return self.execute([torch.tensor(key_cache).bfloat16(), torch.tensor(value_cache).bfloat16(),
                             torch.tensor(block_tables).int(), torch.tensor(context_lens).int(),
                             torch.tensor(key).bfloat16(), torch.tensor(value).bfloat16(), torch.tensor(context_lens).int()], [4, 5])

    @op_test.only_910b
    def test_paged_cache_load_case2(self):
        batch = 16
        seq_len = 1
        num_heads = 16
        head_size_k = 576
        head_size_v = 512
        block_size = 128
        num_blocks = 128
        dtype = "int8"

        num_tokens = batch * seq_len
        OP_NAME = "PagedCacheLoadOperation"
        self.set_paged_cache_load_param(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)

        OP_PARAM = {"type": 1, "cuSeqLens": False, "hasSeqStarts": False}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nz, self.format_nz, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd] * 2)

        key_cache = self.key_cache
        value_cache = self.value_cache
        block_tables = self.block_tables
        context_lens = self.context_lens

        sum_context_lens = sum(context_lens)
        key = np.zeros((sum_context_lens, num_heads * head_size_k)).astype(self.dtype)
        value = np.zeros((sum_context_lens, num_heads * head_size_v)).astype(self.dtype)

        return self.execute([torch.tensor(key_cache), torch.tensor(value_cache),
                             torch.tensor(block_tables).int(), torch.tensor(context_lens).int(),
                             torch.tensor(key), torch.tensor(value), torch.tensor(context_lens).int()], [4, 5])

    @op_test.only_910b
    def test_paged_cache_load_case3(self):
        batch = 32
        seq_len = 1
        num_heads = 64
        head_size_k = 276
        head_size_v = 212
        block_size = 128
        num_blocks = 128
        dtype = "float16"

        num_tokens = batch * seq_len
        OP_NAME = "PagedCacheLoadOperation"
        self.set_paged_cache_load_param(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)
        OP_PARAM = {"type": 1, "cuSeqLens": False, "hasSeqStarts": False}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nz, self.format_nz, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd] * 2)

        key_cache = self.key_cache
        value_cache = self.value_cache
        block_tables = self.block_tables
        context_lens = self.context_lens

        sum_context_lens = int(sum(context_lens))
        key = np.zeros((sum_context_lens, num_heads * head_size_k)).astype(np.float16)
        value = np.zeros((sum_context_lens, num_heads * head_size_v)).astype(np.float16)

        return self.execute([torch.tensor(key_cache).half(), torch.tensor(value_cache).half(),
                             torch.tensor(block_tables).int(), torch.tensor(context_lens).int(),
                             torch.tensor(key).half(), torch.tensor(value).half(), torch.tensor(context_lens).int()], [4, 5])

    @op_test.only_910b
    def test_paged_cache_load_case4(self):
        batch = 32
        seq_len = 1
        num_heads = 64
        head_size_k = 276
        head_size_v = 212
        block_size = 128
        num_blocks = 128
        dtype = "bfloat16"

        num_tokens = batch * seq_len
        OP_NAME = "PagedCacheLoadOperation"
        self.set_paged_cache_load_param(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)

        OP_PARAM = {"type": 1, "cuSeqLens": False, "hasSeqStarts": False}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nz, self.format_nz, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd] * 2)

        key_cache = self.key_cache
        value_cache = self.value_cache
        block_tables = self.block_tables
        context_lens = self.context_lens

        sum_context_lens = sum(context_lens)
        key = np.zeros((sum_context_lens, num_heads * head_size_k)).astype(np.float32)
        value = np.zeros((sum_context_lens, num_heads * head_size_v)).astype(np.float32)

        return self.execute([torch.tensor(key_cache).bfloat16(), torch.tensor(value_cache).bfloat16(),
                             torch.tensor(block_tables).int(), torch.tensor(context_lens).int(),
                             torch.tensor(key).bfloat16(), torch.tensor(value).bfloat16(), torch.tensor(context_lens).int()], [4, 5])

    @op_test.only_910b
    def test_paged_cache_load_case5(self):
        batch = 32
        seq_len = 1
        num_heads = 64
        head_size_k = 276
        head_size_v = 212
        block_size = 128
        num_blocks = 128
        dtype = "int8"

        num_tokens = batch * seq_len
        OP_NAME = "PagedCacheLoadOperation"
        self.set_paged_cache_load_param(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)

        OP_PARAM = {"type": 1, "cuSeqLens": False, "hasSeqStarts": False}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nz, self.format_nz, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd] * 2)

        key_cache = self.key_cache
        value_cache = self.value_cache
        block_tables = self.block_tables
        context_lens = self.context_lens

        sum_context_lens = sum(context_lens)
        key = np.zeros((sum_context_lens, num_heads * head_size_k)).astype(self.dtype)
        value = np.zeros((sum_context_lens, num_heads * head_size_v)).astype(self.dtype)

        return self.execute([torch.tensor(key_cache), torch.tensor(value_cache),
                             torch.tensor(block_tables).int(), torch.tensor(context_lens).int(),
                             torch.tensor(key), torch.tensor(value), torch.tensor(context_lens).int()], [4, 5])

    @op_test.only_910b
    def test_paged_cache_load_case6(self):
        batch = 30
        seq_len = 1
        num_heads = 64
        head_size_k = 76
        head_size_v = 12
        block_size = 128
        num_blocks = 100
        dtype = "float16"

        num_tokens = batch * seq_len
        OP_NAME = "PagedCacheLoadOperation"
        self.set_paged_cache_load_param(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)

        OP_PARAM = {"type": 1, "cuSeqLens": False, "hasSeqStarts": False}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nz, self.format_nz, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd] * 2)

        key_cache = self.key_cache
        value_cache = self.value_cache
        block_tables = self.block_tables
        context_lens = self.context_lens

        sum_context_lens = int(sum(context_lens))
        key = np.zeros((sum_context_lens, num_heads * head_size_k)).astype(np.float16)
        value = np.zeros((sum_context_lens, num_heads * head_size_v)).astype(np.float16)

        return self.execute([torch.tensor(key_cache).half(), torch.tensor(value_cache).half(),
                             torch.tensor(block_tables).int(), torch.tensor(context_lens).int(),
                             torch.tensor(key).half(), torch.tensor(value).half(), torch.tensor(context_lens).int()], [4, 5])

    @op_test.only_910b
    def test_paged_cache_load_case7(self):
        batch = 30
        seq_len = 1
        num_heads = 64
        head_size_k = 76
        head_size_v = 12
        block_size = 128
        num_blocks = 100
        dtype = "bfloat16"

        num_tokens = batch * seq_len
        OP_NAME = "PagedCacheLoadOperation"
        self.set_paged_cache_load_param(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)

        OP_PARAM = {"type": 1, "cuSeqLens": False, "hasSeqStarts": False}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nz, self.format_nz, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd] * 2)

        key_cache = self.key_cache
        value_cache = self.value_cache
        block_tables = self.block_tables
        context_lens = self.context_lens

        sum_context_lens = sum(context_lens)
        key = np.zeros((sum_context_lens, num_heads * head_size_k)).astype(np.float32)
        value = np.zeros((sum_context_lens, num_heads * head_size_v)).astype(np.float32)

        return self.execute([torch.tensor(key_cache).bfloat16(), torch.tensor(value_cache).bfloat16(),
                             torch.tensor(block_tables).int(), torch.tensor(context_lens).int(),
                             torch.tensor(key).bfloat16(), torch.tensor(value).bfloat16(), torch.tensor(context_lens).int()], [4, 5])

    @op_test.only_910b
    def test_paged_cache_load_case8(self):
        batch = 30
        seq_len = 1
        num_heads = 64
        head_size_k = 76
        head_size_v = 12
        block_size = 128
        num_blocks = 100
        dtype = "int8"

        num_tokens = batch * seq_len
        OP_NAME = "PagedCacheLoadOperation"
        self.set_paged_cache_load_param(num_tokens, num_heads, head_size_k, head_size_v, block_size, num_blocks, dtype)

        OP_PARAM = {"type": 1, "cuSeqLens": False, "hasSeqStarts": False}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nz, self.format_nz, self.format_nd, self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd] * 2)

        key_cache = self.key_cache
        value_cache = self.value_cache
        block_tables = self.block_tables
        context_lens = self.context_lens

        sum_context_lens = sum(context_lens)
        key = np.zeros((sum_context_lens, num_heads * head_size_k)).astype(self.dtype)
        value = np.zeros((sum_context_lens, num_heads * head_size_v)).astype(self.dtype)

        return self.execute([torch.tensor(key_cache), torch.tensor(value_cache),
                             torch.tensor(block_tables).int(), torch.tensor(context_lens).int(),
                             torch.tensor(key), torch.tensor(value), torch.tensor(context_lens).int()], [4, 5])

if __name__ == '__main__':
    unittest.main()