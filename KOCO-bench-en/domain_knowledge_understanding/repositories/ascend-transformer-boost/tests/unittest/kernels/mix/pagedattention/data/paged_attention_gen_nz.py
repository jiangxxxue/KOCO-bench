# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import random
import sys
import numpy as np
import math
import logging
from enum import Enum

# np.random.seed(1)
# random.seed(1)
MAX_SEQ_LEN = 1024


class ScaleType(Enum):
    SCALE_TOR = 0
    SCALE_LOGN = 1

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
        group_score = np.matmul(A[i * group_num: (i + 1) * group_num, :, :].astype(np.float32),
                                B[i: (i + 1), :, :].astype(np.float32)).astype(np.float16)
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
    logging.info(f"query:{query.shape}, key:{key.shape}")
    sim = group_matmul(query.shape[0], key.shape[0], query, key)  # (head_num, q_seqlen, k_seqlen)
    logging.info(f"sim:{sim.shape}, alibi_bias:{alibi_bias.shape}")
    sim = sim + alibi_bias
    # softmax
    row_max = np.max(sim, axis=-1, keepdims=True)
    sim -= row_max
    sim = sim.astype("float32")
    sim = np.exp(sim)
    row_sum = np.sum(sim, axis=-1, keepdims=True)
    p = sim / row_sum
    # p = p.astype("float16")
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
        alibi_mask,
        logn_list=None
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
        if logn_list is not None and len(logn_list) > 0:
            scale *= logn_list[i]
        logging.info(f"query.shape: {q.shape}, {q.dtype}, keys.shape: {keys.shape}, "
              f"context_len: {context_len}, keyblocknum: {(context_len + block_size - 1) // block_size}, "
              f"tail: {context_len % block_size}, alibi_bias.shape: {alibi_mask[i].shape}")
        out = ref_masked_attention(q, keys, values, scale, alibi_mask[i, :, :, :context_len])
        out = out.reshape(num_heads, head_size)
        output[i] = out


def run_single_query_cached_kv_attention(
        num_tokens: int,
        num_heads: int,
        kv_heads: int,
        head_size: int,
        block_size: int,
        num_blocks: int,
        k_seqlen: int,
        is_support_alibi: bool,
        dtype: str,
        scale_type: ScaleType
) -> None:
    query = np.random.uniform(-1.0, 1.0, size=(num_tokens, num_heads, head_size)).astype(dtype)

    # (num_blocks, block_size, num_heads, head_size)
    key_cache = np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_size)).astype(dtype)

    # (num_blocks, block_size, num_heads, head_size)
    value_cache = np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_size)).astype(dtype)

    context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_tokens)]  # 一个token处理多少个key
    context_lens = [k_seqlen] * num_tokens
    a = [logging.info(f"context_len: {x} % {block_size} == 1") for x in context_lens if x % block_size == 1]
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
        if (is_support_alibi == True):
            alibi_mask[i, :, :, :context_len] = alibi_bias
    logging.info(f"alibi_mask.shape = {alibi_mask.shape}")

    # 构造logn
    m = 8192 # qwen-7b的seq length
    if scale_type == ScaleType.SCALE_LOGN:
        logn_list = np.array([
                        math.log(i, m) if i > m else 1
                        for i in range(3 * m, 3 * m + num_tokens)
                    ]).astype(np.float16)
    else:
        logn_list = np.array([]).astype(np.float16)

    ref_output = np.zeros_like(query)
    ref_single_query_cached_kv_attention(
        ref_output,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        alibi_mask,
        logn_list=logn_list
    )
    context_lens = np.array(context_lens)
    block_tables = np.array(block_tables)

    tokens_pad = (num_tokens + 15) // 16 * 16
    query = query.reshape(1, num_tokens, num_heads * head_size)
    query_pad = np.zeros((1, tokens_pad, num_heads * head_size))
    query_pad[:, :num_tokens, :] = query
    query_nz = convert_nd_to_nz(query_pad)
    query_nz.astype(np.float16).tofile("query_nz.bin")

    key_cache = key_cache.reshape(num_blocks, block_size, -1)
    key_cache_nz = convert_nd_to_nz(key_cache)
    key_cache_nz.astype(np.float16).tofile("key_cache_nz.bin")

    value_cache = value_cache.reshape(num_blocks, block_size, -1)
    value_cache_nz = convert_nd_to_nz(value_cache)
    value_cache_nz.astype(np.float16).tofile("value_cache_nz.bin")

    block_tables.astype(np.int32).tofile("block_tables.bin")
    context_lens.astype(np.int32).tofile("context_lens.bin")

    max_context_len_pad = (max_context_len + 15) // 16 * 16
    alibi_mask_pad = np.zeros((num_tokens, num_heads, 16, max_context_len_pad))
    alibi_mask_pad[:, :, :1, :max_context_len] = alibi_mask
    alibi_mask_nz = convert_nd_to_nz(alibi_mask_pad)
    alibi_mask_nz.astype(np.float16).tofile("alibi_mask_nz.bin")

    logn_list.tofile("logn_list.bin")

    ref_output = ref_output.reshape(1, num_tokens, num_heads * head_size)
    ref_output_pad = np.zeros((1, tokens_pad, num_heads * head_size))
    ref_output_pad[:, :num_tokens, :] = ref_output
    ref_output_nz = convert_nd_to_nz(ref_output_pad)
    ref_output_nz.astype(np.float16).tofile("expect_nz.bin")
    logging.info(f"==> query nz shape: {query_nz.shape}, key_cache nz shape: {key_cache_nz.shape}, \
          alibi mask nz shape: {alibi_mask_nz.shape}")
    logging.info(f"==> block_tables: {block_tables.shape}, data generate finished!")

    logging.info(f"output pad占比：{ref_output.shape} / {ref_output_pad.shape} = {np.size(ref_output) / np.size(ref_output_pad)}")
    return query, key_cache, value_cache, block_tables, context_lens, ref_output


def test_single_query_cached_kv_attention(num_tokens, block_size, num_blocks, kv_heads, k_seqlen, is_support_alibi, dtype="float16", scale_type=ScaleType.SCALE_LOGN) -> None:

    logging.info(f'Testing single_query_cached_kv_attention with '
          f'num_tokens={num_tokens}, dtype={dtype}, block_size={block_size} \n')
    run_single_query_cached_kv_attention(
        num_tokens=num_tokens,
        num_heads=32,
        kv_heads=kv_heads,
        head_size=128,
        block_size=block_size,
        num_blocks=num_blocks,
        k_seqlen=k_seqlen,
        is_support_alibi=is_support_alibi,
        dtype=dtype,
        scale_type=scale_type
    )


if __name__ == "__main__":
    mode = str(sys.argv[1])
    args = int(sys.argv[2])

    scale_type = ScaleType.SCALE_TOR
    if args == 1:
        num_tokens = 1
        block_size = 128
        num_blocks = 128
        kv_heads = 32
        k_seqlen = 128
    elif args == 2:
        num_tokens = 2
        block_size = 128
        num_blocks = 128
        kv_heads = 32
        k_seqlen = 128
    elif args == 3:
        num_tokens = 20
        block_size = 128
        num_blocks = 128
        kv_heads = 16
        k_seqlen = 128
    elif args == 4:
        num_tokens = 20
        block_size = 128
        num_blocks = 128
        kv_heads = 16
        k_seqlen = 513
    elif args == 5:
        num_tokens = 20
        block_size = 128
        num_blocks = 128
        kv_heads = 32
        k_seqlen = 513
        scale_type = ScaleType.SCALE_LOGN
    is_support_alibi = False
    if (mode == "MIX_PAGED_ATTENTION_NZ"):
        is_support_alibi = False
    else:
        is_support_alibi = True


    test_single_query_cached_kv_attention(num_tokens, block_size, num_blocks, kv_heads, k_seqlen, is_support_alibi, scale_type=scale_type)
