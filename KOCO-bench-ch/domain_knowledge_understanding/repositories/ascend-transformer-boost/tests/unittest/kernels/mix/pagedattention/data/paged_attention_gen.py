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
import logging
import os
import random
import unittest

import numpy
import numpy as np
import torch
import torch_npu

np.random.seed(1)
random.seed(1)

MAX_SEQ_LEN = 1024


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
    logging.info(score.shape)
    return score


def ref_masked_attention(
        query,  # (1, num_heads, head_size)
        key,  # (context_len, kv_heads, head_size)
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
        alibi_mask
) -> None:
    num_heads = query.shape[1]
    kv_heads = value_cache.shape[2]
    head_size = value_cache.shape[3]
    block_size = value_cache.shape[1]
    alibi_dim = alibi_mask.ndim

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
        if alibi_dim == 4:
            logging.info(f"query.shape: {q.shape}, {q.dtype}, keys.shape: {keys.shape}, "
                  f"context_len: {context_len}, keyblocknum: {(context_len + block_size - 1) // block_size}, "
                  f"tail: {context_len % block_size}, alibi_bias.shape: {alibi_mask[i].shape}")            
            out = ref_masked_attention(q, keys, values, scale, alibi_mask[i, :, :, :context_len])
        else:
            logging.info(f"query.shape: {q.shape}, {q.dtype}, keys.shape: {keys.shape}, "
                  f"context_len: {context_len}, keyblocknum: {(context_len + block_size - 1) // block_size}, "
                  f"tail: {context_len % block_size}, alibi_bias.shape: {alibi_mask.shape}")            
            out = ref_masked_attention(q, keys, values, scale, alibi_mask[:, :, :context_len])
        out = out.reshape(num_heads, head_size)
        output[i] = out


def generate_data(
        num_tokens=2,
        num_heads=32,
        kv_heads=32,
        head_size=128,
        block_size=128,
        num_blocks=64,
	    k_seqlen=500,
        dtype="float16",
        alibi_dim=4,
):
    query = np.random.uniform(-1.0, 1.0, size=(num_tokens, num_heads, head_size)).astype(dtype)

    # kv cache shape: (num_blocks, block_size, num_heads, head_size)
    key_cache = np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_size)).astype(dtype)
    value_cache = np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_size)).astype(dtype)

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
    alibi_slopes = np.random.random(num_heads).astype(np.float16)
    if alibi_dim == 4:
        alibi_mask = np.zeros((num_tokens, num_heads, 1, max_context_len), dtype=np.float16)
        for i, context_len in enumerate(context_lens):
            position_ids = np.arange(context_len).astype(np.int32)
            alibi_bias = (position_ids - context_len + 1).astype(np.float16)
            alibi_bias = alibi_slopes.reshape(-1, 1, 1) * alibi_bias.reshape(1, 1, -1)   # (head_num, 1, context)
            alibi_mask[i, :, :, :context_len] = alibi_bias
    else:
        alibi_mask = np.zeros((num_heads, 1, max_context_len), dtype=np.float16)
        context_len = context_lens[0]
        position_ids = np.arange(context_len).astype(np.int32)
        alibi_bias = (position_ids - context_len + 1).astype(np.float16)
        alibi_bias = alibi_slopes.reshape(-1, 1, 1) * alibi_bias.reshape(1, 1, -1)   # (head_num, 1, context)
        alibi_mask[:, :, :context_len] = alibi_bias

    logging.info(f"alibi_mask.shape = {alibi_mask.shape}")

    # alibi_slopes = np.zeros(num_heads)
    ref_output = np.zeros_like(query)
    ref_single_query_cached_kv_attention(
        ref_output,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        alibi_mask
    )
    context_lens = np.array(context_lens).astype(np.int32)
    block_tables = np.array(block_tables).astype(np.int32)

    query.astype(np.float16).tofile("query.bin")
    key_cache.astype(np.float16).tofile("key_cache.bin")
    value_cache.astype(np.float16).tofile("value_cache.bin")
    block_tables.astype(np.int32).tofile("block_tables.bin")
    context_lens.astype(np.int32).tofile("context_lens.bin")
    alibi_mask.astype(np.float16).tofile("alibi_mask.bin")
    ref_output.astype(np.float16).tofile("expect.bin")
    logging.info(f"==> block_tables: {block_tables.shape}, data generate finished!")
    return query, key_cache, value_cache, block_tables, context_lens, ref_output


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        alibi_dim = 4
    else:
        alibi_dim = sys.argv[1]
    in_tensors = generate_data(kv_heads=32, alibi_dim=alibi_dim)
