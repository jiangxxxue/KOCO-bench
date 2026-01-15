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

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
import operation_test  # NOQA: E402

MAX_SEQ_LEN = 1024
kv_head_num = 16
OP_NAME = "PagedAttentionOperation"
SCALE_TYPE = 0
PARAM = json.dumps({"headNum": 32, "qkScale": (1 / float(math.sqrt(576))), "kvHeadNum": kv_head_num, "scaleType": SCALE_TYPE})
rand_list = np.random.uniform(0.0, 2.0, size=100).astype(np.float32)
rand_list = [x if x > 1.0 else 1.0 for x in rand_list]
print("rand_list: ", rand_list)


class PagedInputData:
    def __init__(self, query, key_cache, value_cache, block_tables, context_lens, logn_tensor):
        self.query = query
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.block_tables = block_tables
        self.context_lens = context_lens
        self.logn_tensor = logn_tensor


def group_matmul(head, kv_head, A, B):
    group_num = head // kv_head
    score = None
    for i in range(kv_head):
        group_score = np.matmul(A[i * group_num: (i + 1) * group_num, :, :].astype(np.float32),
                                B[i: (i + 1), :, :].astype(np.float32)).astype(np.float16)
        if score is None:
            score = group_score
        else:
            score = np.concatenate((score, group_score), 0)
    print(score.shape)
    return score


def ref_masked_attention(
        query,  # (1, num_heads, head_size)
        key,  # (context_len, kv_heads, head_size)
        value,
        scale: float
):
    # Q * K.T
    query = query * scale
    query = np.transpose(query, (1, 0, 2))
    key = np.transpose(key, (1, 2, 0))
    sim = group_matmul(query.shape[0], key.shape[0], query, key)
    # softmax
    row_max = np.max(sim, axis=-1, keepdims=True)
    sim -= row_max
    sim = sim.astype("float32")
    sim = np.exp(sim)
    row_sum = np.sum(sim, axis=-1, keepdims=True)
    p = sim / row_sum
    p = p.astype("float16")
    # P * V
    value = np.transpose(value, (1, 0, 2))
    out = group_matmul(query.shape[0], key.shape[0], p, value)
    out = np.transpose(out, (1, 0, 2))
    return out


def ref_single_query_cached_kv_attention(
        output,
        paged_input
) -> None:
    query = paged_input.query
    key_cache = paged_input.key_cache   # (num_blocks, block_size, num_heads, head_size)
    value_cache = paged_input.value_cache  # (num_blocks, block_size, num_heads, head_size)
    block_tables = paged_input.block_tables
    context_lens = paged_input.context_lens
    num_heads = query.shape[1]
    kv_heads = value_cache.shape[2]
    head_size = key_cache.shape[3]
    head_size_v = value_cache.shape[3]
    block_size = value_cache.shape[1]

    num_input_tokens = query.shape[0]
    for i in range(num_input_tokens):
        q = np.expand_dims(query[i], 0)
        block_table = block_tables[i]
        context_len = int(context_lens[i])

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, block_offset, :, :]
            k = k.reshape(kv_heads, head_size)
            keys.append(k)

            v = value_cache[block_number, block_offset, :, :]
            v = v.reshape(kv_heads, head_size_v)
            values.append(v)
        keys = np.stack(np.array(keys), axis=0)
        values = np.stack(np.array(values), axis=0)
        print(f"query.shape: {q.shape}, {q.dtype}, keys.shape: {keys.shape}, "
              f"context_len: {context_len}, keyblocknum: {(context_len + block_size - 1) // block_size}, "
              f"tail: {context_len % block_size}")
        if SCALE_TYPE == 1:
            scale = 1.0 * rand_list[i] / (head_size**0.5)
        else:
            scale = 1.0 / (head_size**0.5)
        out = ref_masked_attention(q, keys, values, scale)
        out = out.reshape(num_heads, head_size_v)
        output[i] = out


def generate_data(
        num_tokens=2,
        kv_heads=kv_head_num,
        block_size=16,
        num_blocks=64
):
    num_heads = 32
    head_size = 576
    # head_size_v = np.random.randint(1,head_size)
    head_size_v = 512
    dtype = "float16"
    query = np.random.uniform(-1.0, 1.0, size=(num_tokens, num_heads, head_size)).astype(dtype)

    # key value cache: (num_blocks, block_size, num_heads, head_size)
    key_cache = np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_size)).astype(dtype)
    value_cache = np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_size_v)).astype(dtype)

    context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_tokens)]  # 一个token处理多少个key
    context_lens = [1024] * num_tokens
    _ = [print(f"context_len: {x} % {block_size} == 1") for x in context_lens if x % block_size == 1]
    max_context_len = max(context_lens)

    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []   # （num_tokens, max_num_blocks_per_seq）
    for _ in range(num_tokens):
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)

    context_lens = np.array(context_lens).astype(np.int32)
    block_tables = np.array(block_tables).astype(np.int32)
    logn_tensor = np.array([rand_list[x] for x in range(query.shape[0])]).astype(np.float32)

    paged_input = PagedInputData(query, key_cache, value_cache, block_tables, context_lens, logn_tensor)
    ref_output = np.zeros((num_tokens, num_heads, head_size_v)).astype(dtype)
    ref_single_query_cached_kv_attention(
        ref_output,
        paged_input
    )

    print(f"==> block_tables: {block_tables.shape}, data generate finished!")
    return paged_input, ref_output


paged_attention_input, gt_output = generate_data()
data = paged_attention_input.query, paged_attention_input.key_cache, paged_attention_input.value_cache, \
    paged_attention_input.block_tables, paged_attention_input.context_lens, paged_attention_input.logn_tensor, gt_output

RUN_PARAM = json.dumps({"contextLens": paged_attention_input.context_lens.tolist(), "scaleType": SCALE_TYPE})

in_tensors = [torch.from_numpy(tensor) for tensor in data]
# print(f'------------in_tensors1:{in_tensors}')
# in_tensors = [tensor.npu().to(torch.bfloat16) for tensor in in_tensors]
in_tensors = [tensor.npu() for tensor in in_tensors]
# print(f'------------in_tensors2:{in_tensors}')
#
# print(f'------------in_tensors_final:{in_tensors}')
a = [print(tensor.dtype, tensor.device, tensor.shape) for tensor in in_tensors]


class TestPagedAttentionAttentionOperation(operation_test.OperationTest):
    def golden_calc(self, input_tensors):
        return [in_tensors[-1].to(torch.bfloat16)]
        # return [in_tensors[-1]]

    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.allclose(out_tensor, golden_out_tensor, rtol=0.001, atol=0.001)


    def test(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        in_tensors_final = [in_tensors[0].to(torch.bfloat16), in_tensors[1].to(torch.bfloat16), in_tensors[2].to(torch.bfloat16), in_tensors[3], in_tensors[4]]
        # in_tensors_final = [in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4]]
        if SCALE_TYPE == 1:
            in_tensors_final.append(in_tensors[5])
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, in_tensors_final)


if __name__ == '__main__':
    unittest.main()