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

def group_mm_torch(heads, group_num, A, B, is_k, offset1, de_scale1_fp32, offset2, de_scale2_fp32, is_int8_flag=True, has_bias=True):
    group_head = heads // group_num
    score = None
    for i in range(group_num):
        if is_int8_flag:
            int8_B = B[i: (i+1), :, :, ]
            head_dim = int8_B.shape[2]
            int32_B = torch.matmul(torch.eye(int8_B.shape[1]).to(torch.int32), int8_B.to(torch.int32)).to(torch.int32)
            if is_k:
                if has_bias:
                    int32_B = int32_B + offset1[i * head_dim:(i + 1) * head_dim]
                fp32_B = int32_B.to(torch.float32) * de_scale1_fp32[i * head_dim:(i + 1) * head_dim] 
                fp32_B = torch.permute(fp32_B, (0, 2, 1))
            else:
                if has_bias:
                    int32_B = int32_B + offset2[i * head_dim:(i + 1) * head_dim]
                fp32_B = int32_B.to(torch.float32) * de_scale2_fp32[i * head_dim:(i + 1) * head_dim] 
            group_score = torch.matmul(A[i * group_head: (i + 1) * group_head, :, :].to(torch.float32),
                                        fp32_B).to(torch.float16)
        else:
            group_score = torch.matmul(A[i * group_head: (i + 1) * group_head, :, :].to(torch.float32),
                                        B[i:(i + 1), :, :].to(torch.float32))
        if score is None:
            score = group_score
        else:
            score = torch.cat((score, group_score), 0)
    return score


def process_deq_scale(deq_scale) -> np.ndarray:
    new_deq_scale = np.frombuffer(deq_scale.tobytes(), dtype=np.uint32)
    return new_deq_scale.astype(np.uint64)

def ref_masked_attention(
            query,  # (1, num_heads, head_size)
            key,  # (context_len, kv_heads, head_size)
            value,
            scale: float,
            alibi_bias,
            is_int8_flag,
            offset1,
            de_scale1_fp32,
            offset2,
            de_scale2_fp32,
            mask_data_type = torch.bfloat16
    ):
    # Q * K.T
    query = query
    query = torch.permute(query, (1, 0, 2))
    if not is_int8_flag:
        key = torch.permute(key, (1, 2, 0))  # 0 1 2
    else:
        key = torch.permute(key, (1, 0, 2))
    sim = group_mm_torch(query.shape[0], key.shape[0], query, key, 1, offset1, de_scale1_fp32, offset2, de_scale2_fp32).to(mask_data_type)  # (head_num, q_seqlen, k_seqlen)
    if alibi_bias is None:
        sim = sim.to(torch.float16) * np.float16(scale)
    else :
        if (mask_data_type == torch.bfloat16):
            sim = sim.to(torch.float32) * scale
            sim = sim + alibi_bias.to(torch.float32)
        else:
            sim = sim.to(torch.float16) * np.float16(scale)
            sim = sim + alibi_bias.to(torch.float16)
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
    out = group_mm_torch(query.shape[0], key.shape[0], p, value, 0, offset1, de_scale1_fp32, offset2, de_scale2_fp32)
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
            is_int8_flag,
            offset1,
            de_scale1_fp32,
            offset2,
            de_scale2_fp32,
            mask_dim = 4,
            mask_data_type = torch.bfloat16
    ) -> None:
    num_heads = query.shape[1]
    kv_heads = value_cache.shape[2]
    head_size = value_cache.shape[3]
    block_size = value_cache.shape[1]
 
    num_input_tokens = query.shape[0]
    index = 0
    for i in range(len(context_lens)):
        block_table = block_tables[i]
        context_len = int(context_lens[i])
        if context_len == 0:
            continue
            
        q = query[index].view(1, num_heads, head_size)
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
        keys = torch.stack(keys, axis=0)
        values = torch.stack(values, axis=0)
        scale = np.float32(1.0 / (head_size ** 0.5))
        if mask_dim == 4:
            logging.info(f"query.shape: {q.shape}, {q.dtype}, keys.shape: {keys.shape}, "
                    f"context_len: {context_len}, keyblocknum: {(context_len + block_size - 1) // block_size}, "
                f"tail: {context_len % block_size}, alibi_bias.shape: {mask[i].shape}")
            out = ref_masked_attention(q, keys, values, scale, mask[i, :, :, :context_len], is_int8_flag, offset1, de_scale1_fp32, offset2, de_scale2_fp32, mask_data_type)
            out = out.reshape(num_heads, head_size)
        elif mask_dim == 3:
            logging.info(f"query.shape: {q.shape}, {q.dtype}, keys.shape: {keys.shape}, "
                    f"context_len: {context_len}, keyblocknum: {(context_len + block_size - 1) // block_size}, "
                f"tail: {context_len % block_size}, alibi_bias.shape: {mask[i].shape}")
            out = ref_masked_attention(q, keys, values, scale, mask[i, :, :context_len], is_int8_flag, offset1, de_scale1_fp32, offset2, mask_data_type)
            out = out.reshape(num_heads, head_size)
        else:
            out = self.ref_masked_attention(q, keys, values, scale, mask, is_int8_flag, offset1, de_scale1_fp32, offset2, mask_data_type)
            out = out.reshape(num_heads, head_size)
 
        output[index] = out
        index = index + 1


def calc_data(num_tokens, num_heads, kv_heads, head_size, block_size, num_blocks, k_seqlen,\
                  dtype, mask_dim = 4, mask_data_type = torch.bfloat16,\
                  dynamic_batch = False, dynamic_seqlen = None,is_int8_flag = True, has_bias = True):
 
        logging.info(f'input info: {num_tokens}, {num_heads}, {kv_heads}, {head_size}, {block_size}, {num_blocks}, {k_seqlen}, {dtype}')
        
        query = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_tokens, num_heads, head_size))).to(mask_data_type)
        # (num_blocks, block_size, num_heads, head_size)
        if is_int8_flag:
            key_cache = torch.from_numpy(np.random.uniform(-4.0, 4.0, size=(num_blocks, block_size, kv_heads, head_size))).to(torch.int8)
            # (num_blocks, block_size, num_heads, head_size)
            value_cache = torch.from_numpy(np.random.uniform(-4.0, 4.0, size=(num_blocks, block_size, kv_heads, head_size))).to(torch.int8)
        else:
            key_cache = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_size))).to(mask_data_type)
            # (num_blocks, block_size, num_heads, head_size)
            value_cache = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_size))).to(mask_data_type)

 
        if dynamic_batch:
            context_lens = dynamic_seqlen
        else:
            context_lens = [k_seqlen] * num_tokens
        max_context_len = max(context_lens)
 
        max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
        block_tables = []   # （num_tokens, max_num_blocks_per_seq）
        batch = len(context_lens)
        for _ in range(batch):
            block_table = [
                random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)
 
        # alibi mask
        if mask_dim == 4:
            alibi_slopes = np.random.random(num_heads).astype(np.float16)
            mask = np.zeros((batch, num_heads, 1, max_context_len), dtype=np.float16)
            for i, context_len in enumerate(context_lens):
                if context_len == 0:
                    continue
                position_ids = np.arange(context_len).astype(np.int32)
                alibi_bias = (position_ids - context_len + 1).astype(np.float16)
                alibi_bias = alibi_slopes.reshape(-1, 1, 1) * alibi_bias.reshape(1, 1, -1)   # (head_num, 1, context)
                mask[i, :, :, :context_len] = alibi_bias
            logging.info(f"mask.shape = {mask.shape}")
            mask = torch.from_numpy(mask).to(mask_data_type)
        # normal mask
        elif mask_dim == 3:
            mask = np.zeros((batch, 1, max_context_len), dtype=np.float16)
            for i in range(batch):
                mask[i, :, :i] = -10000
            logging.info(f"norm_mask.shape = {mask.shape}")
            mask = torch.from_numpy(mask).to(mask_data_type)
        else: # no mask
            mask = None
 
        
        
        de_scale1_fp32 = np.random.randint(-1, 2, size=(kv_heads * head_size)).astype(np.float32)
        de_scale1_int64 = process_deq_scale(de_scale1_fp32)
 
        de_scale2_fp32 =  np.random.randint(-1, 2, size=(kv_heads * head_size)).astype(np.float32)
        de_scale2_int64 = process_deq_scale(de_scale2_fp32)
 
        offset1 = np.random.randint(-20, 20, size=(kv_heads * head_size)).astype(np.int32)
 
        offset2 = np.random.randint(-20, 20, size=(kv_heads * head_size)).astype(np.int32)
 
        de_scale1_int64 = torch.tensor(list(de_scale1_int64), dtype=torch.int64)
        de_scale2_int64 =  torch.tensor(list(de_scale2_int64), dtype=torch.int64)
        de_scale1_fp32 = torch.from_numpy(de_scale1_fp32)
        de_scale2_fp32 = torch.from_numpy(de_scale2_fp32)
        offset1 = torch.from_numpy(offset1)
        offset2 = torch.from_numpy(offset2)

 
        ref_output = torch.zeros_like(query)
        ref_single_query_cached_kv_attention(
            ref_output,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            mask,
            is_int8_flag,
            offset1,
            de_scale1_fp32,
            offset2,
            de_scale2_fp32,
            mask_dim,
            mask_data_type
        )


 

        block_tables = np.array(block_tables).astype(np.int32)
        contex_lens = np.array(context_lens).astype(np.int32)

        return query, key_cache, value_cache, block_tables, contex_lens, mask, de_scale1_fp32, offset1, de_scale2_fp32, offset2, ref_output

num_tokens = 4
num_heads = 8
kv_heads = 8
block_size = 128
head_size = 128
num_blocks = 64
k_seqlen = 34
tor = 1.0 / (head_size ** 0.5)
dtype = "float16"
mask_dim = 4
dynamic_batch = False
is_int8_flag = True
has_bias = True

q, key_cache, value_cache, block_tables, contex_lens, alib_mask, de_scale1_fp32, offset1, de_scale2_fp32, offset2, golden_out = calc_data(num_tokens, num_heads, kv_heads, head_size, block_size, num_blocks, k_seqlen, dtype, mask_dim, torch.float16)
data = q, key_cache, value_cache, torch.from_numpy(block_tables), torch.from_numpy(contex_lens), alib_mask, de_scale1_fp32, offset1, de_scale2_fp32, offset2, golden_out

RUN_PARAM = json.dumps({"contextLens": contex_lens.tolist()})

in_tensors = [tensor.npu() for tensor in data]
a = [print(tensor.dtype, tensor.device, tensor.shape) for tensor in in_tensors]

OP_NAME = "PagedAttentionOperation"
PARAM = json.dumps({"headNum": 8, "qkScale": (1 / float(math.sqrt(128))), "kvHeadNum": 8, "isSupportAlibi": True, "maskType":1, "quantType":1, "hasQuantOffset": True})


class TestPagedAttentionOperationDequant(operation_test.OperationTest):
    def golden_calc(self, input_tensors):
        return [in_tensors[10]]

    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.allclose(out_tensor, golden_out_tensor, rtol=1, atol=1)

    def test(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4], in_tensors[5], in_tensors[6], in_tensors[7], in_tensors[8], in_tensors[9]])
 
if __name__ == '__main__':
    unittest.main()