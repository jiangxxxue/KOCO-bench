#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
 
import logging
import sys
import unittest
import math
import numpy as np
import torch
import random
import sys
import os
import math
import json
import torch_npu
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402
np.random.seed(1)
random.seed(1)
MAX_SEQ_LEN = 1024

MAX_SEQ_LEN = 1024
kv_head_num = 32
OP_NAME = "PagedAttentionOperation"
PARAM = json.dumps({"headNum": 32, "qkScale": (1 / float(math.sqrt(288))), "kvHeadNum": kv_head_num, "isSupportAlibi": True, "maskType":1})

class PagedInputData:
    def __init__(self, query, key_cache, value_cache, block_tables, context_lens, mask):
        self.query = query
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.block_tables = block_tables
        self.context_lens = context_lens
        self.mask = mask
 
def group_mm_torch(heads, group_num, A, B):
    group_head = heads // group_num
    score = None
    for i in range(group_num):
        group_score = torch.matmul(A[i * group_head: (i + 1) * group_head, :, :].to(torch.float32),
                                    B[i:(i + 1), :, :].to(torch.float32))
        if score is None:
            score = group_score
        else:
            score = torch.cat((score, group_score), 0)
    return score
 
def ref_masked_attention(
        query,  # (1, num_heads, head_size)
        key,  # (context_len, kv_heads, head_size)
        value,
        scale: float,
        alibi_bias,
        mask_data_type = torch.bfloat16
):
    # Q * K.T
    query = query
    query = torch.permute(query, (1, 0, 2))
    key = torch.permute(key, (1, 2, 0))  # 0 1 2
    sim = group_mm_torch(query.shape[0], key.shape[0], query, key).to(mask_data_type)  # (head_num, q_seqlen, k_seqlen)
    sim = sim.to(torch.float32) * scale
    sim = sim + alibi_bias.to(torch.float32)
    sim = sim.numpy()
    # softmax
    row_max = np.max(sim, axis=-1, keepdims=True)
    sim -= row_max
    sim = np.exp(sim)
    row_sum = np.sum(sim, axis=-1, keepdims=True)
    p = sim / row_sum
    p = torch.from_numpy(p).to(mask_data_type)
    # P * V
    value = torch.permute(value, (1, 0, 2))
    out = group_mm_torch(query.shape[0], key.shape[0], p, value)
    out = torch.permute(out, (1, 0, 2))
    return out
 
def ref_single_query_cached_kv_attention(
        output,
        query,
        key_cache,  # (num_blocks, block_size, num_heads, head_size)
        value_cache,  # (num_blocks, block_size, num_heads, head_size)
        block_tables,
        context_lens,
        mask,
        mask_dim = 4,
        mask_data_type = torch.bfloat16
) -> None:
    num_heads = query.shape[1]
    kv_heads = value_cache.shape[2]
    head_size = key_cache.shape[3]
    head_size_v = value_cache.shape[3]
    block_size = value_cache.shape[1]

    num_input_tokens = query.shape[0]
    for i in range(num_input_tokens):
        q = query[i].view(1, num_heads, head_size)
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
        keys = torch.stack(keys, axis=0)
        values = torch.stack(values, axis=0)
        scale = np.float32(1.0 / (head_size ** 0.5))
        if mask_dim == 4:
            print(f"query.shape: {q.shape}, {q.dtype}, keys.shape: {keys.shape}, "
                    f"context_len: {context_len}, keyblocknum: {(context_len + block_size - 1) // block_size}, "
                f"tail: {context_len % block_size}, alibi_bias.shape: {mask[i].shape}")
            out = ref_masked_attention(q, keys, values, scale, mask[i, :, :, :context_len], mask_data_type)
            out = out.reshape(num_heads, head_size_v)
        elif mask_dim == 3:
            print(f"query.shape: {q.shape}, {q.dtype}, keys.shape: {keys.shape}, "
                    f"context_len: {context_len}, keyblocknum: {(context_len + block_size - 1) // block_size}, "
                f"tail: {context_len % block_size}, alibi_bias.shape: {mask[i].shape}")
            out = ref_masked_attention(q, keys, values, scale, mask[i, :, :context_len], mask_data_type)
            out = out.reshape(num_heads, head_size_v)
        output[i] = out
 
def calc_data(num_tokens, num_heads, kv_heads, head_size, block_size, num_blocks, k_seqlen, dtype, mask_dim = 4, mask_data_type = torch.bfloat16):
 
    print("input info: ", num_tokens, num_heads, kv_heads, head_size, block_size, num_blocks, k_seqlen, dtype)

    head_size_v = np.random.randint(1,head_size)
    query = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_tokens, num_heads, head_size))).to(mask_data_type)
    # (num_blocks, block_size, num_heads, head_size)
    key_cache = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_size))).to(mask_data_type)
    # (num_blocks, block_size, num_heads, head_size)
    value_cache = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_size_v))).to(mask_data_type)

    context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_tokens)]  # 一个token处理多少个key
    
    context_lens = [k_seqlen] * num_tokens
    
    max_context_len = max(context_lens)

    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    
    block_tables = []   # （num_tokens, max_num_blocks_per_seq）
    for _ in range(num_tokens):
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)

    # alibi mask        
    if mask_data_type == torch.bfloat16 and mask_dim == 4:
        alibi_slopes = np.random.random(num_heads).astype(np.float16)
        mask = np.zeros((num_tokens, num_heads, 1, max_context_len), dtype=np.float16)
        for i, context_len in enumerate(context_lens):
            position_ids = np.arange(context_len).astype(np.int32)
            alibi_bias = (position_ids - context_len + 1).astype(np.float16)
            alibi_bias = alibi_slopes.reshape(-1, 1, 1) * alibi_bias.reshape(1, 1, -1)   # (head_num, 1, context)
            mask[i, :, :, :context_len] = alibi_bias
        print(f"mask.shape = {mask.shape}")
        mask = torch.from_numpy(mask).to(mask_data_type)
    # normal mask
    elif mask_data_type == torch.float16 and mask_dim == 3:
        mask = np.zeros((num_tokens, 1, max_context_len), dtype=np.float16)
        for i in range(num_tokens):
            mask[i, :, :i] = -10000
        print(f"norm_mask.shape = {mask.shape}")
        # print(mask)
        mask = torch.from_numpy(mask).to(mask_data_type)
    else:
        assert(False)
    ref_output = torch.zeros((num_tokens, num_heads, head_size_v)).to(mask_data_type)
    ref_single_query_cached_kv_attention(
        ref_output,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        mask,
        mask_dim,
        mask_data_type
    )

    q = query
    key_cache = key_cache
    value_cache = value_cache
    block_tables = np.array(block_tables).astype(np.int32)
    contex_lens = np.array(context_lens).astype(np.int32)
    alib_mask = mask
    golden_out = ref_output
    logging.debug("**********data gen shape***********")
    print(f"==> query shape: {query.shape}, key_cache shape: {key_cache.shape}, \
                alibi mask shape: {mask.shape}")
    print(f"==> block_tables: {block_tables.shape}, data generate finished!")
    return q, key_cache, value_cache, block_tables, contex_lens, alib_mask, golden_out


num_tokens = 1
num_heads = 32
kv_heads = 32
block_size = 128
head_size = 288
num_blocks = 64
k_seqlen = 128
tor = 1.0 / (head_size ** 0.5)
dtype = "float16"
mask_dim = 3
q, key_cache, value_cache, block_tables, contex_lens, alib_mask, golden_out = calc_data(num_tokens, num_heads, kv_heads, head_size, block_size, num_blocks, k_seqlen, dtype, mask_dim, torch.float16)
data = q, key_cache, value_cache, torch.from_numpy(block_tables), torch.from_numpy(contex_lens), alib_mask, golden_out
in_tensors = [tensor.npu() for tensor in data]

RUN_PARAM = json.dumps({"contextLens": contex_lens.tolist()})

a = [print(tensor.dtype, tensor.device, tensor.shape) for tensor in in_tensors]


class TestPagedAttentionAttentionOperation(operation_test.OperationTest):
    def golden_calc(self, input_tensors):
        return [in_tensors[6]]

    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.allclose(out_tensor, golden_out_tensor, rtol=0.001, atol=0.001)

    def test(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4], in_tensors[5]])
 
if __name__ == '__main__':
    unittest.main()