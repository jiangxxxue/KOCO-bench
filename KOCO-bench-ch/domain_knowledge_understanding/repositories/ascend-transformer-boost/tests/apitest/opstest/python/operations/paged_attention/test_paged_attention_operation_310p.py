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
from precision_calcu import *

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

MAX_SEQ_LEN = 1024
kv_head_num = 32
OP_NAME = "PagedAttentionOperation"
MASK_TYPE = 0
SCALE_TYPE = 1
PARAM = json.dumps({"headNum": 32, "qkScale": (1 / float(math.sqrt(128))), "kvHeadNum": kv_head_num, "maskType": MASK_TYPE, "scaleType": SCALE_TYPE})
rand_list = np.random.uniform(0.0, 2.0, size=100).astype(np.float16)
rand_list = [x if x > 1.0 else 1.0 for x in rand_list]


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

def qkMM1(
    query,
    key
):
    result = None
    qk_k = key.shape[1]
    for qk_k_split in range(0, qk_k, 128):
        sub_k = 128
        if qk_k_split == 512:
            sub_k = 64
        query_k = query[:, :, qk_k_split : qk_k_split + sub_k]
        key_k = key[:, qk_k_split : qk_k_split + sub_k, :]
        result_split = torch.matmul(query_k.to(torch.float32), key_k.to(torch.float32))
        if result is None:
            result = result_split
        else:
            result = result + result_split
    return result

def softmax1(
    qk_result,
    is_first,
    gm
):
    sim = qk_result.numpy()
    lm = np.max(sim, axis=-1, keepdims=True)
    if is_first:
        hm = lm
        dm = 0
    else:
        hm = np.maximum(gm, lm)
        dm = gm - hm
    gm = hm
    sim_sub = sim - hm
    sim_sub = np.exp(sim_sub.astype(np.float32))
    if qk_result.dtype == torch.float16:
        sim_sub = sim_sub.astype(np.float16)
    row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
    return torch.from_numpy(sim_sub), row_sum, dm, gm

def ref_flash_attention(
        query,
        key,
        value,
        scale: float,
        alibi_bias,
        mask_data_type = torch.bfloat16,
        query_rope = None,
        key_rope = None,
        context_len = 0,
):
    if not torch.is_tensor(query):
        query = torch.from_numpy(query)
    if not torch.is_tensor(key):
        key = torch.from_numpy(key)
    if not torch.is_tensor(value):
        value = torch.from_numpy(value)
    if not torch.is_tensor(alibi_bias):
        alibi_bias = torch.from_numpy(alibi_bias)
    query = torch.permute(query, (1, 0, 2))
    key = torch.permute(key, (1, 2, 0))
    value = torch.permute(value, (1, 0, 2))
    context_size = 128
    group_num = query.shape[0] // key.shape[0]
    if group_num != 1:
        key = key.repeat_interleave(group_num, dim=0)
        value = value.repeat_interleave(group_num, dim=0)
    gl = None
    gl_high = None
    go = None
    go_high = None
    for kv_start in range(0, context_len - 1, context_size):
        sub_len = context_size
        if kv_start + context_size > context_len:
            sub_len = context_len - kv_start
        sub_key = key[:, :, kv_start : kv_start + sub_len]
        sub_mask = alibi_bias[:, :, kv_start : kv_start + sub_len]
        sub_value = value[:, kv_start : kv_start + sub_len, :]
        qk_result = qkMM1(query, sub_key)
        qk_result_high = qkMM1(query.to(torch.float32), sub_key.to(torch.float32))
        qk_result = qk_result.to(torch.float16) * scale
        qk_result_high = qk_result_high * scale
        if MASK_TYPE != 0:
            qk_result += sub_mask
            qk_result_high += sub_mask.to(torch.float32)
        if kv_start == 0:
            gm = None
        p_result, row_sum, dm, gm = softmax1(qk_result, kv_start == 0, gm)
        if kv_start == 0:
            gm_high = None
        p_result_high, row_sum_high, dm_high, gm_high = softmax1(qk_result_high, kv_start == 0, gm_high)
        lo = torch.matmul(p_result.to(torch.float32), sub_value.to(torch.float32))
        lo = lo.to(torch.float16)
        lo_high = torch.matmul(p_result_high, sub_value.to(torch.float32))
        lo = lo.numpy()
        lo_high = lo_high.numpy()
        if kv_start == 0:
            gl = row_sum
            gl_high = row_sum_high
            go = lo
            go_high = lo_high
        else:
            dm = np.exp(dm)
            dm_high = np.exp(dm_high)
            gl = gl * dm
            gl = gl + row_sum

            go = go * dm
            go = go + lo

            gl_high = gl_high * dm_high
            gl_high = gl_high + row_sum_high

            go_high = go_high * dm_high
            go_high = go_high + lo_high
    go = go / gl
    go_high = go_high / gl_high
    go = np.transpose(go, (1, 0, 2))
    go_high = np.transpose(go_high, (1, 0, 2))
    return torch.from_numpy(go), torch.from_numpy(go_high)

def ref_single_query_cached_kv_attention(
        output,
        output_gt,
        query,
        key_cache,   # (num_blocks, block_size, num_heads, head_size)
        value_cache,  # (num_blocks, block_size, num_heads, head_size)
        block_tables,
        context_lens,
        alibi_mask
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
        scale = 1.0 * rand_list[i] / (head_size**0.5)

        print(f"query.shape: {q.shape}, {q.dtype}, keys.shape: {keys.shape}, "
              f"context_len: {context_len}, keyblocknum: {(context_len + block_size - 1) // block_size}, "
              f"tail: {context_len % block_size}, alibi_bias.shape: {alibi_mask[i].shape}")

        out, out_high = ref_flash_attention(q, keys, values, scale, alibi_bias = alibi_mask[i, :, :, :context_len], context_len = context_len)

        out_high = out_high.reshape(num_heads, head_size)
        out = out.reshape(num_heads, head_size)
        output_gt[i] = out_high
        output[i] = out


def generate_data(
        num_tokens=2,
        num_heads=32,
        kv_heads=32,
        head_size=128,
        block_size=32,
        num_blocks=64,
        k_seqlen=500,
        dtype="float16",
):
    query = np.random.uniform(-5.0, 5.0, size=(num_tokens, num_heads, head_size)).astype(dtype)
    # kv cache shape: (num_blocks, block_size, num_heads, head_size)
    key_cache = np.random.uniform(-5.0, 5.0, size=(num_blocks, block_size, kv_heads, head_size)).astype(dtype) 
    value_cache = np.random.uniform(-5.0, 5.0, size=(num_blocks, block_size, kv_heads, head_size)).astype(dtype)
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

    # alibi mask
    alibi_slopes = np.random.random(num_heads).astype(np.float16)
    alibi_mask = np.zeros((num_tokens, num_heads, 1, max_context_len), dtype=np.float16)
    for i, context_len in enumerate(context_lens):
        position_ids = np.arange(context_len).astype(np.int32)
        alibi_bias = (position_ids - context_len + 1).astype(np.float16)
        alibi_bias = alibi_slopes.reshape(-1, 1, 1) * alibi_bias.reshape(1, 1, -1)   # (head_num, 1, context)
        if MASK_TYPE != 0:
            alibi_mask[i, :, :, :context_len] = alibi_bias
    print(f"alibi_mask.shape = {alibi_mask.shape}")

 
    # alibi_slopes = np.zeros(num_heads)
    ref_output = np.zeros_like(query)
    output_gt = np.zeros_like(query).astype(np.float32)
    ref_single_query_cached_kv_attention(
        ref_output,
        output_gt,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        alibi_mask
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
    alibi_mask_pad = np.zeros((num_tokens, num_heads, 16, max_context_len_pad))
    alibi_mask_pad[:, :, :1, :max_context_len] = alibi_mask
    alibi_mask_nz = convert_nd_to_nz(alibi_mask_pad)
    alibi_mask_nz = alibi_mask_nz.reshape(num_tokens * num_heads, -1, 16, 16).astype(np.float16)
    key_cache_nz = np.ascontiguousarray(key_cache_nz)  # fixbug
    value_cache_nz = np.ascontiguousarray(value_cache_nz)
    alibi_mask_nz = np.ascontiguousarray(alibi_mask_nz)
    logn_tensor = np.array([rand_list[x] for x in range(query.shape[0])]).astype(np.float16)
    return query, key_cache_nz, value_cache_nz, block_tables, context_lens, alibi_mask_nz, logn_tensor, ref_output, output_gt


data = generate_data()
in_tensors = [torch.from_numpy(tensor) for tensor in data]
in_tensors = [tensor.npu() for tensor in in_tensors]
a = [print(tensor.dtype, tensor.device, tensor.shape) for tensor in in_tensors]


class TestPagedAttentionAttentionOperation(operation_test.OperationTest):
    def golden_calc(self, input_tensors):
        return [in_tensors[-2]]

    def golden_compare(self, out_tensor, golden_out_tensor):
        result_double = compare_cv(in_tensors[-1], golden_out_tensor.npu(), out_tensor.npu())
        return result_double

    def test(self):
        if not operation_test.get_soc_version() == 'Ascend310P':
            print("this testcase only supports Ascend310P")
            return True
        in_tensors[1] = torch_npu.npu_format_cast(in_tensors[1], 29)
        in_tensors[2] = torch_npu.npu_format_cast(in_tensors[2] , 29)
        in_tensors[5] = torch_npu.npu_format_cast(in_tensors[5] , 29)

        in_tensors_final = [in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4]]

        if MASK_TYPE != 0:
            in_tensors_final.append(in_tensors[5])

        if SCALE_TYPE != 0:
            in_tensors_final.append(in_tensors[6])

        self.execute(OP_NAME, PARAM, in_tensors_final)


if __name__ == '__main__':
    unittest.main()