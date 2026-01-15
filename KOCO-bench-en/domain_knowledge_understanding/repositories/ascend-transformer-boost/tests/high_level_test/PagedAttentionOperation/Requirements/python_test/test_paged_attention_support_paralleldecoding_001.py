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
from paged_attention_test_data_generator import PagedAttentionDataGenerator

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

import operation_test  # NOQA: E402

np.random.seed(1)
random.seed(1)
MAX_SEQ_LEN = 1024


class UnpadPagedAttention():
    def group_matmul(self, head, kv_head, A, B):
        group_num = head // kv_head
        score = None
        for i in range(kv_head):
            group_score = np.matmul(A[i * group_num: (i + 1) * group_num, :, :].astype(np.float32),
                                    B[i: (i + 1), :, :].astype(np.float32)).astype(np.float16)
            if score is None:
                score = group_score
            else:
                score = np.concatenate((score, group_score), 0)
        return score

    def ref_masked_attention(self,
                             query,  # (q_seqlen, num_heads, head_size)
                             key,  # (k_seqlen, kv_heads, head_size)
                             value,
                             scale: float,
                             mask  # (q_seqlen, k_seqlen)
                             ):
        # Q * K.T
        query = query * scale
        query = np.transpose(query, (1, 0, 2))
        key = np.transpose(key, (1, 2, 0))
        sim = self.group_matmul(query.shape[0], key.shape[0], query, key)  # (head_num, q_seqlen, k_seqlen)
        sim = sim + mask[:sim.shape[-2], :sim.shape[-1]]
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
        out = self.group_matmul(query.shape[0], key.shape[0], p, value)
        out = np.transpose(out, (1, 0, 2))
        return out

    def ref_single_query_cached_kv_attention(self,
                                             output,
                                             query,
                                             key_cache,  # (num_blocks, block_size, num_heads, head_size)
                                             value_cache,  # (num_blocks, block_size, num_heads, head_size)
                                             block_tables,
                                             q_seqlen_list,
                                             k_seqlen_list,
                                             global_mask,
                                             mask_type
                                             ) -> None:
        num_heads = query.shape[1]
        kv_heads = value_cache.shape[2]
        head_size = value_cache.shape[3]
        block_size = value_cache.shape[1]

        batch = len(q_seqlen_list)
        cu_seqlen = 0
        for i in range(batch):
            q_seqlen = int(q_seqlen_list[i])
            k_seqlen = int(k_seqlen_list[i])
            q = query[cu_seqlen: cu_seqlen + q_seqlen, :, :]
            block_table = block_tables[i]
            keys = []
            values = []
            for j in range(k_seqlen):  # j 每个k token拼接
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
            scale = 1.0 / (head_size ** 0.5)
            if mask_type == 0:
                mask = global_mask[k_seqlen - q_seqlen: k_seqlen, :k_seqlen]  # prefill, decoder, prefill + decoder
            elif mask_type == 3:
                mask = global_mask[cu_seqlen: (cu_seqlen + q_seqlen), :]  # lookahead: cur_q: cur_q + q
            print(f"q_seqlen: {q_seqlen}, query.shape: {q.shape}, {q.dtype}, "
                  f"k_seqlen: {k_seqlen}, keys.shape: {keys.shape},"
                  f"key block_num: {(k_seqlen + block_size - 1) // block_size}, tail: {k_seqlen % block_size}")
            out = self.ref_masked_attention(q, keys, values, scale, mask)
            out = out.reshape(-1, num_heads, head_size)
            print(f"out.shape: {out.shape}")
            output[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out
            cu_seqlen += q_seqlen_list[i]

    def calc_data(self,
                  num_heads: int,
                  kv_heads: int,
                  num_blocks: int,
                  block_size: int,
                  head_size: int,
                  q_seqlen_list: int,
                  k_seqlen_list: int,
                  mask_type,
                  dtype: str
                  ):
        num_tokens = np.array(q_seqlen_list).sum()
        batch_size = len(q_seqlen_list)
        self.max_context_len = max(k_seqlen_list)
        query = np.random.uniform(-1.0, 1.0, size=(num_tokens, num_heads, head_size)).astype(dtype)

        # (num_blocks, block_size, num_heads, head_size)
        key_cache = np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_size)).astype(dtype)
        # (num_blocks, block_size, num_heads, head_size)
        value_cache = np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_size)).astype(dtype)

        max_k_seqlen = max(k_seqlen_list)
        max_num_blocks_per_seq = (max_k_seqlen + block_size - 1) // block_size
        block_tables = []  # (num_tokens, max_num_blocks_per_seq）
        for i in range(batch_size):
            block_table = [
                max_num_blocks_per_seq * i + j for j in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)

        if mask_type == 0:
            mask = np.ones(shape=(max_k_seqlen, max_k_seqlen)).astype(dtype)
            mask = np.triu(mask, 1)
            mask *= -10000.0
        elif mask_type == 3:
            mask = np.zeros(shape=(num_tokens, max_k_seqlen)).astype(dtype)
            pre_qseqlen = 0
            for i in range(batch_size):
                qseqlen = q_seqlen_list[i]
                tri = np.ones((qseqlen, qseqlen))
                tri = np.triu(tri, 1)
                tri *= -10000.0
                mask[pre_qseqlen:(pre_qseqlen + qseqlen), -qseqlen:] = tri
                pre_qseqlen += qseqlen

        ref_output = np.zeros_like(query)
        self.ref_single_query_cached_kv_attention(
            ref_output,
            query,
            key_cache,
            value_cache,
            block_tables,
            q_seqlen_list,
            k_seqlen_list,
            mask,
            mask_type
        )

        self.q = query
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.block_tables = np.array(block_tables).astype(np.int32)
        self.q_seqlen_list = np.array(q_seqlen_list).astype(np.int32)
        self.k_seqlen_list = np.array(k_seqlen_list).astype(np.int32)
        self.mask = mask
        self.golden_out = ref_output

        return self.q, self.key_cache, self.value_cache, self.block_tables, self.k_seqlen_list, \
            self.mask, self.q_seqlen_list, ref_output


q_seqlen_list = [1, 15, 30, 6]
k_seqlen_list = [10, 64, 64, 64]
num_heads = 32
kv_heads = 32
head_size = 128
num_blocks = 512
block_size = 128
tor = 1.0 / (head_size ** 0.5)
dtype = "float16"

pa = UnpadPagedAttention()
datas = pa.calc_data(num_heads, kv_heads, num_blocks, block_size, head_size, q_seqlen_list, k_seqlen_list, 3, dtype)
data_generator = PagedAttentionDataGenerator()
data_generator.data_type = torch.float16
data_generator.is_int8_flag = False
data_generator.max_context_len = 64

OP_NAME = "PagedAttentionOperation"
PARAM = json.dumps({"headNum": num_heads, "qkScale": tor, "kvHeadNum": kv_heads, "maskType": 3, "calcType": 1})
RUN_PARAM = json.dumps({"contextLens": datas[4].tolist(), "qLens": datas[6].tolist(), "maskType": 3})

tensors = [torch.from_numpy(data) for data in datas]
in_tensors = [tensor.npu().contiguous() for tensor in tensors]


class TestPagedAttentionAttentionOperation(operation_test.OperationTest):
    def golden_calc(self, input_tensors):
        return [in_tensors[-1]]

    def golden_compare(self, out_tensor, golden_out_tensor):
        ratios = [0.001, 0.001, 0.005, 0.005]
        return data_generator.compare_output_data(out_tensor.cpu(), golden_out_tensor.cpu(), ratios)

    def test(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM,
                                [in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4],
                                 in_tensors[5], in_tensors[6]])


if __name__ == '__main__':
    unittest.main()