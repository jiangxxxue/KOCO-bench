#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import json
import math
import os
import random
import sys
import unittest
import collections
import numpy as np
import torch
import torch_npu


sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402
not_support_device = ['Ascend910A','Ascend310B','Ascend310P']

MAX_SEQ_LEN = 1024
OP_NAME = "PagedCacheLoadOperation"
PARAM = json.dumps({"type": 1})

def shape_nd_to_nz(shape, dtype='float16'):
    assert len(shape) >= 2
    batch = shape[:-2]   
    a, b = shape[-2], shape[-1]
    if dtype == np.int8:
        a0, b0 = 16, 32
    else:
        a0, b0 = 16, 16
    return list(batch) + [math.ceil(b / b0), math.ceil(a / a0), a0, b0]

def gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]

def convert_nd_to_nz(x):
    array_trans = gen_axes_for_transpose(len(x.shape) - 2, [2, 0, 1, 3]) 
    x_shape = shape_nd_to_nz(x.shape, dtype=x.dtype)
    *_, n1, m1, m0, n0 = x_shape
    return x.reshape(x_shape[:-4] + [m1, m0, n1, n0]).transpose(*array_trans) 

def generate_data(
        batch=random.randint(1, 10),
        seq_len=random.randint(1, 10),
        num_heads= random.choice([32,64]),
        head_size_k=random.randint(1, 128),
        head_size_v=random.randint(1, 128),
        block_size=random.choice([16,32,64,128]),
        num_blocks=random.randint(1, 128),
):
    dtype =  random.choice(["float16", "bfloat16", np.int8])
    print(f"batch={batch}, seq_len={seq_len}, num_heads={num_heads}, head_size_k={head_size_k}, head_size_v={head_size_v}, block_size={block_size}, num_blocks={num_blocks}, dtype={dtype}")
    soc_version = operation_test.get_soc_version()
    num_tokens = batch * seq_len

    if dtype == "float16":
            key_cache = np.random.randint(1, 11, size=(num_blocks, block_size, num_heads, head_size_k)).astype(np.float16)
            value_cache = np.random.randint(1, 11, size=(num_blocks, block_size, num_heads, head_size_v)).astype(np.float16)
    elif dtype == "bfloat16":
            key_cache = np.random.randint(1, 11, size=(num_blocks, block_size, num_heads, head_size_k)).astype(np.float32)
            value_cache = np.random.randint(1, 11, size=(num_blocks, block_size, num_heads, head_size_v)).astype(np.float32)
    else:
            key_cache = np.random.randint(1, 11, size=(num_blocks, block_size, num_heads, head_size_k)).astype(dtype)
            value_cache = np.random.randint(1, 11, size=(num_blocks, block_size, num_heads, head_size_v)).astype(dtype)

    context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_tokens)]
    max_context_len = max(context_lens)

    max_num_blocks_per_req = (max_context_len + block_size - 1) // block_size
    block_tables = []  # [num_tokens, max_num_blocks_per_seq]
    for _ in range(num_tokens):
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_req)
        ]
        block_tables.append(block_table)

    context_lens = np.array(context_lens).astype(np.int32)
    block_tables = np.array(block_tables).astype(np.int32)
    sum_context_lens = sum(context_lens)

    if dtype == "float16":
            key_expect = np.zeros((sum_context_lens, num_heads, head_size_k)).astype(np.float16)
            value_expect = np.zeros((sum_context_lens, num_heads, head_size_v)).astype(np.float16)
            key = np.zeros((sum_context_lens, num_heads * head_size_k)).astype(np.float16)
            value = np.zeros((sum_context_lens, num_heads * head_size_v)).astype(np.float16)
    elif dtype == "bfloat16":
            key_expect = np.zeros((sum_context_lens, num_heads, head_size_k)).astype(np.float32)
            value_expect = np.zeros((sum_context_lens, num_heads, head_size_v)).astype(np.float32)
            key = np.zeros((sum_context_lens, num_heads * head_size_k)).astype(np.float32)
            value = np.zeros((sum_context_lens, num_heads * head_size_v)).astype(np.float32)
    else:
            key_expect = np.zeros((sum_context_lens, num_heads, head_size_k)).astype(dtype)
            value_expect = np.zeros((sum_context_lens, num_heads, head_size_v)).astype(dtype)
            key = np.zeros((sum_context_lens, num_heads * head_size_k)).astype(dtype)
            value = np.zeros((sum_context_lens, num_heads * head_size_v)).astype(dtype)
    
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

            temp_k = key_cache[block_id][block_offset]
            temp_v = value_cache[block_id][block_offset]

            key_expect[kv_rslt_id] = temp_k
            value_expect[kv_rslt_id] = temp_v
            kv_rslt_id += 1

    if dtype == "float16":
            key_cache = key_cache.reshape(num_blocks, block_size, -1)
            key_cache_nz = convert_nd_to_nz(key_cache)
            key_cache_nz = key_cache_nz.reshape(num_blocks, -1, block_size, 16).astype(np.float16)
            value_cache = value_cache.reshape(num_blocks, block_size, -1)
            value_cache_nz = convert_nd_to_nz(value_cache)
            value_cache_nz = value_cache_nz.reshape(num_blocks, -1, block_size, 16).astype(np.float16)
            key_cache_nz = np.ascontiguousarray(key_cache_nz)  
            value_cache_nz = np.ascontiguousarray(value_cache_nz)
    elif dtype == "bfloat16":
            key_cache = key_cache.reshape(num_blocks, block_size, -1)
            key_cache_nz = convert_nd_to_nz(key_cache)
            key_cache_nz = key_cache_nz.reshape(num_blocks, -1, block_size, 16).astype(np.float32)
            value_cache = value_cache.reshape(num_blocks, block_size, -1)
            value_cache_nz = convert_nd_to_nz(value_cache)
            value_cache_nz = value_cache_nz.reshape(num_blocks, -1, block_size, 16).astype(np.float32)
            key_cache_nz = np.ascontiguousarray(key_cache_nz)  
            value_cache_nz = np.ascontiguousarray(value_cache_nz)
    else:
            key_cache = key_cache.reshape(num_blocks, block_size, -1)
            key_cache_nz = convert_nd_to_nz(key_cache)
            key_cache_nz = key_cache_nz.reshape(num_blocks, -1, block_size, 32).astype(dtype)
            value_cache = value_cache.reshape(num_blocks, block_size, -1)
            value_cache_nz = convert_nd_to_nz(value_cache)
            value_cache_nz = value_cache_nz.reshape(num_blocks, -1, block_size, 32).astype(dtype)
            key_cache_nz = np.ascontiguousarray(key_cache_nz)  
            value_cache_nz = np.ascontiguousarray(value_cache_nz)
    return key_cache_nz, value_cache_nz, block_tables,context_lens, key_expect.reshape(sum_context_lens, num_heads * head_size_k), value_expect.reshape(sum_context_lens, num_heads * head_size_v), key, value

data = generate_data()
in_tensors = [torch.from_numpy(tensor) for tensor in data]
in_tensors = [tensor.npu() for tensor in in_tensors]
a = [print(tensor.dtype, tensor.device, tensor.shape) for tensor in in_tensors]


class ReshapeAndCacheGradOperation(operation_test.OperationTest):
    soc_version = operation_test.get_soc_version()
    def golden_calc(self, input_tensors):
        if in_tensors[4].dtype == torch.float32:
            in_tensors[4] = in_tensors[4].to(torch.bfloat16)
            in_tensors[5] = in_tensors[5].to(torch.bfloat16)
        return [in_tensors[4], in_tensors[5]]

    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.allclose(out_tensor, golden_out_tensor, rtol=0.001, atol=0.001)

    def test(self):
        if operation_test.get_soc_version() in not_support_device:
            print("These test cases only support A2/A3")
            return True

        in_tensors[0] = torch_npu.npu_format_cast(in_tensors[0], 29)
        in_tensors[1] = torch_npu.npu_format_cast(in_tensors[1] , 29)
        if in_tensors[0].dtype == torch.float32:
            in_tensors[0] = in_tensors[0].to(torch.bfloat16)
            in_tensors[1] = in_tensors[1].to(torch.bfloat16)
            in_tensors[6] = in_tensors[6].to(torch.bfloat16)
            in_tensors[7] = in_tensors[7].to(torch.bfloat16)
        self.execute_out(OP_NAME, PARAM, [in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[6], in_tensors[7]], [in_tensors[6], in_tensors[7]])


if __name__ == '__main__':
    unittest.main()
