#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

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

def shape_nd_to_nz(shape, dtype='float16'):
    assert len(shape) >= 2
    batch = shape[:-2]   # 最后两维nd->nz
    a, b = shape[-2], shape[-1]
    a0, b0 = 16, 16
    return list(batch) + [math.ceil(b / b0), math.ceil(a / a0), a0, b0]


def gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]


def convert_nd_to_nz(x):
    array_trans = gen_axes_for_transpose(len(x.shape) - 2, [2, 0, 1, 3]) # (m1, m0, n1, n0) -> (n1, m1, m0, n0)
    x_shape = shape_nd_to_nz(x.shape, dtype=x.dtype)
    *_, n1, m1, m0, n0 = x_shape
    return x.reshape(x_shape[:-4] + [m1, m0, n1, n0]).transpose(*array_trans) # x原始需要对齐，才能reshape


def group_matmul(head, kv_head, A, B):
    group_num = head // kv_head
    score = None
    for i in range(kv_head):
        group_score = np.matmul(A[i*group_num: (i+1)*group_num, :, :].astype(np.float32),
                                B[i: (i+1), :, :].astype(np.float32)).astype(np.float16)
        if score is None:
            score = group_score
        else:
            score = np.concatenate((score, group_score), 0)
    print(score.shape)
    return score


def ref_masked_attention(
        query,# (1, num_heads, head_size)
        key,# (context_len, kv_heads, head_size)
        value,
        scale: float,
        alibi_bias
):
    # Q * K.T
    query = query * scale
    query = np.transpose(query, (1, 0, 2))
    key = np.transpose(key, (1, 2, 0))
    sim = group_matmul(query.shape[0], key.shape[0], query, key)
    sim = sim + alibi_bias
    # softmax
    row_max = np.max(sim, axis=-1, keepdims=True)
    sim -= row_max
    sim = sim.astype("float32")
    sim = np.exp(sim)
    row_sum = np.sum(sim, axis=-1, keepdims=True)
    p = sim / row_sum
    # P * V
    value = np.transpose(value, (1, 0, 2))
    out = group_matmul(query.shape[0], key.shape[0], p, value)
    out = np.transpose(out, (1, 0, 2))
    return out
 

def ref_single_query_cached_kv_attention(
        output,
        query,
        key_cache,   # (num_blocks, block_size, num_heads, head_size)
        value_cache,  # (num_blocks, block_size, num_heads, head_size)
        block_tables,
        context_lens,
        mask,
        mask_type
) -> None:
    num_heads = query.shape[1]
    kv_heads = value_cache.shape[2]
    head_size = value_cache.shape[3]
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
            v = v.reshape(kv_heads, head_size)
            values.append(v)

        keys = np.stack(np.array(keys), axis=0)
        values = np.stack(np.array(values), axis=0)
        scale = 1.0 / (head_size**0.5)
        if mask_type == 2:
            print(f"query.shape: {q.shape}, {q.dtype}, keys.shape: {keys.shape}, "
                    f"context_len: {context_len}, keyblocknum: {(context_len + block_size - 1) // block_size}, "
                f"tail: {context_len % block_size}, alibi_bias.shape: {mask[i].shape}")
            out = ref_masked_attention(q, keys, values, scale, mask[i, :, :, :context_len])
            out = out.reshape(num_heads, head_size)
        elif mask_type == 1:
            print(f"query.shape: {q.shape}, {q.dtype}, keys.shape: {keys.shape}, "
                    f"context_len: {context_len}, keyblocknum: {(context_len + block_size - 1) // block_size}, "
                f"tail: {context_len % block_size}, alibi_bias.shape: {mask[i].shape}")
            out = ref_masked_attention(q, keys, values, scale, mask[i, :, :context_len])
            out = out.reshape(num_heads, head_size)
        else:
            out = ref_masked_attention(q, keys, values, scale, mask[i, :, :, :context_len])
        output[i] = out


def generate_data(
        num_tokens = 1,
        num_heads = 28,
        kv_heads = 4,
        head_size = 64,
        block_size = 128,
        num_blocks = 4,
        k_seqlen = 500,
        dtype = "float16",
        mask_type = 0
):
    query = np.random.uniform(-1.0, 1.0, size=(num_tokens, num_heads, head_size)).astype(dtype)
    # kv cache shape: (num_blocks, block_size, num_heads, head_size)
    key_cache = np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_size)).astype(dtype)
    value_cache = np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_size)).astype(dtype)
    context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_tokens)]# 一个token处理多少个key
    context_lens = [k_seqlen] * num_tokens
    max_context_len = max(context_lens)

    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []   # （num_tokens, max_num_blocks_per_seq）
    for _ in range(num_tokens):
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    mask = np.zeros((num_tokens, num_heads, 1, max_context_len), dtype=np.float16)
    # alibi mask
    if mask_type == 2:
        alibi_slopes = np.random.random(num_heads).astype(np.float16)
        mask = np.zeros((num_tokens, num_heads, 1, max_context_len), dtype=np.float16)
        for i, context_len in enumerate(context_lens):
            position_ids = np.arange(context_len).astype(np.int32)
            alibi_bias = (position_ids - context_len + 1).astype(np.float16)
            alibi_bias = alibi_slopes.reshape(-1, 1, 1) * alibi_bias.reshape(1, 1, -1)   # (head_num, 1, context)
            if mask_type != 0:
                mask[i, :, :, :context_len] = alibi_bias
        print(f"mask.shape = {mask.shape}")
    # normal mask
    elif mask_type == 1:
        mask = np.zeros((num_tokens, 1, max_context_len), dtype=np.float16)
        for i in range(num_tokens):
            if mask_type != 0:
                mask[i, :, :i] = -10000
        print(f"norm_mask.shape = {mask.shape}")
 
    ref_output = np.zeros_like(query)
    ref_single_query_cached_kv_attention(
        ref_output,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        mask,
        mask_type
    )

    context_lens = np.array(context_lens).astype(np.int32)
    block_tables = np.array(block_tables).astype(np.int32)
    print(f"==> block_tables: {block_tables.shape}, data generate finished!")

    key_cache = key_cache.reshape(num_blocks, block_size, -1)
    key_cache_nz = convert_nd_to_nz(key_cache)
    key_cache_nz = key_cache_nz.reshape(num_blocks, -1, block_size, 16).astype(np.float16)
    value_cache = value_cache.reshape(num_blocks, block_size, -1)
    value_cache_nz = convert_nd_to_nz(value_cache)
    value_cache_nz = value_cache_nz.reshape(num_blocks, -1, block_size, 16).astype(np.float16)

    max_context_len_pad = (max_context_len + 15) // 16 * 16
    mask_nz = np.zeros((num_tokens, 16, max_context_len_pad))
    if mask_type == 1:
        mask_pad = np.zeros((num_tokens, 16, max_context_len_pad))
        mask_pad[:, :1, :max_context_len] = mask
        mask_nz = convert_nd_to_nz(mask_pad)
        mask_nz = mask_nz.reshape(num_tokens, -1, 16, 16).astype(np.float16)
        key_cache_nz = np.ascontiguousarray(key_cache_nz)  # fixbug
        value_cache_nz = np.ascontiguousarray(value_cache_nz)
        mask_nz = np.ascontiguousarray(mask_nz)
    elif mask_type == 2:
        max_context_len_pad = (max_context_len + 15) // 16 * 16
        mask_pad = np.zeros((num_tokens, num_heads, 16, max_context_len_pad))
        mask_pad[:, :, :1, :max_context_len] = mask
        mask_nz = convert_nd_to_nz(mask_pad)
        mask_nz = mask_nz.reshape(num_tokens * num_heads, -1, 16, 16).astype(np.float16)
        key_cache_nz = np.ascontiguousarray(key_cache_nz)  # fixbug
        value_cache_nz = np.ascontiguousarray(value_cache_nz)
        mask_nz = np.ascontiguousarray(mask_nz)

    return query, key_cache_nz, value_cache_nz, block_tables, context_lens, mask_nz, ref_output

class TestPagedAttentionAttentionOperation(operation_test.OperationTest):
    def golden_calc(self, input_tensors):
        return [self.in_tensors[6]]

    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.allclose(out_tensor, golden_out_tensor, rtol=0.001, atol=0.001)
    
    def test_No_Mask(self):
        if not operation_test.get_soc_version() == 'Ascend910A':
            print("this testcase only supports Ascend910A")
            return True
        OP_NAME = "PagedAttentionOperation"
        num_tokens = 2
        num_heads = 32
        kv_heads = 32
        head_size = 128
        block_size = 32
        num_blocks = 64
        k_seqlen = 500
        dtype = "float16"
        mask_type = 0
        qkScale = 1 / float(math.sqrt(head_size))
        data = generate_data(num_tokens,num_heads,kv_heads,head_size,block_size,num_blocks,k_seqlen,dtype,mask_type)
        self.in_tensors = [torch.from_numpy(tensor) for tensor in data]
        self.in_tensors = [tensor.npu() for tensor in self.in_tensors]
        a = [print(tensor.dtype, tensor.device, tensor.shape) for tensor in self.in_tensors]
        PARAM = json.dumps({"headNum":num_heads, "qkScale":qkScale,"kvHeadNum":kv_heads,"maskType":mask_type})
        self.in_tensors[1] = torch_npu.npu_format_cast(self.in_tensors[1], 29)
        self.in_tensors[2] = torch_npu.npu_format_cast(self.in_tensors[2], 29)

        self.execute(OP_NAME, PARAM, [self.in_tensors[0], self.in_tensors[1],
                                      self.in_tensors[2], self.in_tensors[3], self.in_tensors[4]])
    
    def test_Mask_norm(self):
        if not operation_test.get_soc_version() == 'Ascend910A':
            print("this testcase only supports Ascend910A")
            return True
        OP_NAME = "PagedAttentionOperation"
        num_tokens = 2
        num_heads = 32
        kv_heads = 32
        head_size = 128
        block_size = 32
        num_blocks = 64
        k_seqlen = 500
        dtype = "float16"
        mask_type = 1
        qkScale = 1 / float(math.sqrt(head_size))
        data = generate_data(num_tokens,num_heads,kv_heads,head_size,block_size,num_blocks,k_seqlen,dtype,mask_type)
        self.in_tensors = [torch.from_numpy(tensor) for tensor in data]
        self.in_tensors = [tensor.npu() for tensor in self.in_tensors]
        a = [print(tensor.dtype, tensor.device, tensor.shape) for tensor in self.in_tensors]
        PARAM = json.dumps({"headNum":num_heads, "qkScale":qkScale,"kvHeadNum":kv_heads,"maskType":mask_type})
        self.in_tensors[1] = torch_npu.npu_format_cast(self.in_tensors[1], 29)
        self.in_tensors[2] = torch_npu.npu_format_cast(self.in_tensors[2], 29)
        self.in_tensors[5] = torch_npu.npu_format_cast(self.in_tensors[5], 29)

        self.execute(OP_NAME, PARAM, [self.in_tensors[0], self.in_tensors[1], self.in_tensors[2],
                                      self.in_tensors[3], self.in_tensors[4], self.in_tensors[5]])
    
    def test_Mask_alibi(self):
        if not operation_test.get_soc_version() == 'Ascend910A':
            print("this testcase only supports Ascend910A")
            return True
        OP_NAME = "PagedAttentionOperation"
        num_tokens = 2
        num_heads = 32
        kv_heads = 32
        head_size = 128
        block_size = 32
        num_blocks = 64
        k_seqlen = 500
        dtype = "float16"
        mask_type = 2
        qkScale = 1 / float(math.sqrt(head_size))
        data = generate_data(num_tokens,num_heads,kv_heads,head_size,block_size,num_blocks,k_seqlen,dtype,mask_type)
        self.in_tensors = [torch.from_numpy(tensor) for tensor in data]
        self.in_tensors = [tensor.npu() for tensor in self.in_tensors]
        a = [print(tensor.dtype, tensor.device, tensor.shape) for tensor in self.in_tensors]
        PARAM = json.dumps({"headNum":num_heads, "qkScale":qkScale,"kvHeadNum":kv_heads,"maskType":mask_type})
        self.in_tensors[1] = torch_npu.npu_format_cast(self.in_tensors[1], 29)
        self.in_tensors[2] = torch_npu.npu_format_cast(self.in_tensors[2], 29)
        self.in_tensors[5] = torch_npu.npu_format_cast(self.in_tensors[5], 29)

        self.execute(OP_NAME, PARAM, [self.in_tensors[0], self.in_tensors[1], self.in_tensors[2],
                                      self.in_tensors[3], self.in_tensors[4], self.in_tensors[5]])

if __name__ == '__main__':
    unittest.main()