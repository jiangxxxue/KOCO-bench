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
from paged_attention.paged_attention_test_data_generator import PagedAttentionDataGenerator
 
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402
 
np.random.seed(1)
random.seed(1)
MAX_SEQ_LEN = 1024
 
def shape_nd_to_nz(shape, dtype='float16'):
    assert len(shape) >= 2
    batch = shape[:-2]   # nd->nz
    a, b = shape[-2], shape[-1]
    a0, b0 = 16, 16
    return list(batch) + [math.ceil(b / b0), math.ceil(a / a0), a0, b0]
 
 
def gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]
 
 
def convert_nd_to_nz(x):
    array_trans = gen_axes_for_transpose(len(x.shape) - 2, [2, 0, 1, 3]) # (m1, m0, n1, n0) -> (n1, m1, m0, n0)
    x_shape = shape_nd_to_nz(x.shape, dtype=x.dtype)
    *_, n1, m1, m0, n0 = x_shape
    return x.reshape(x_shape[:-4] + [m1, m0, n1, n0]).transpose(*array_trans)
 
class UnpadPagedAttention():
    def group_matmul(self, head, kv_head, A, B):
        group_num = head // kv_head
        score = None
        for i in range(kv_head):
            group_score = torch.matmul(A[i * group_num: (i + 1) * group_num, :, :].to(torch.float32),
                                    B[i : (i + 1), :, :].to(torch.float32))
            if score is None:
                score = group_score
            else:
                score = torch.cat((score, group_score), 0)
        return score
 
    def ref_masked_attention(self,
            query,  # (q_seqlen, num_heads, head_size)
            key,    # (k_seqlen, kv_heads, head_size)
            value,
            scale: float,
            mask    # (q_seqlen, k_seqlen)
    ):
        # Q * K.T
        query = query * scale
        query = query.permute(1, 0, 2)
        key = key.permute(1, 2, 0)
        sim = self.group_matmul(query.shape[0], key.shape[0], query, key)  # (head_num, q_seqlen, k_seqlen)
        sim = sim + mask[:sim.shape[-2], :sim.shape[-1]]
        # softmax
        row_max = torch.max(sim,-1, keepdim=True)[0]
        sim -= row_max
        sim = sim.to(torch.float32)
        sim = torch.exp(sim)
        row_sum = torch.sum(sim, -1, keepdim=True)
        p = sim / row_sum
        # p = p.astype("float16")
        # P * V
        value = value.permute(1, 0, 2)
        out = self.group_matmul(query.shape[0], key.shape[0], p, value)
        out = out.permute(1, 0, 2)
        return out.reshape((1, out.shape[0], out.shape[1] * out.shape[2]))
 
    def ref_single_query_cached_kv_attention(self,
        output,
        query,
        key_cache,    # (num_blocks, block_size, num_heads, head_size)
        value_cache,  # (num_blocks, block_size, num_heads, head_size)
        block_tables,
        q_seqlen_list,
        k_seqlen_list,
        global_mask,
        masktype,
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
            q = query[cu_seqlen : cu_seqlen + q_seqlen, :, :]
            block_table = block_tables[i]
 
            keys = []
            values = []
            for j in range(k_seqlen): # j 每个k token拼接
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
            scale = 1.0 / (head_size ** 0.5)
            mask = global_mask # prefill, decoder, prefill + decoder
            print(f"q_seqlen: {q_seqlen}, query.shape: {q.shape}, {q.dtype}, "
                f"k_seqlen: {k_seqlen}, keys.shape: {keys.shape},"
                f"key block_num: {(k_seqlen + block_size - 1) // block_size}, tail: {k_seqlen % block_size},")
                # f"mask.shape: {mask.shape}, {mask}")
            if masktype == "norm":
                out = self.ref_masked_attention(q, keys, values, scale, mask[i])
            else:
                out = self.ref_masked_attention(q, keys, values, scale, mask[cu_seqlen:(cu_seqlen + q_seqlen),:])
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
                  masktype: str,
    ):
        num_tokens = np.array(q_seqlen_list).sum()
        max_q = max(q_seqlen_list)
        batch_size = len(q_seqlen_list)
        query = torch.zeros((num_tokens, num_heads, head_size)).half()
        query.uniform_(-1, 1)
        # (num_blocks, block_size, num_heads, head_size)
        key_cache = torch.zeros((num_blocks, block_size, kv_heads, head_size)).half()
        key_cache.uniform_(-1, 1)
        # (num_blocks, block_size, num_heads, head_size)
        value_cache = torch.zeros((num_blocks, block_size, kv_heads, head_size)).half()
        value_cache.uniform_(-1, 1)
 
        max_k_seqlen = max(k_seqlen_list)
        max_num_blocks_per_seq = (max_k_seqlen + block_size - 1) // block_size
        block_tables = []   # （num_tokens, max_num_blocks_per_seq）
        for i in range(batch_size):
            block_table = [
            max_num_blocks_per_seq * i + j for j in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)
        if masktype == "norm":
            mask = torch.zeros(size=(batch_size, max_q, max_k_seqlen))
            for i in range(batch_size):
                qseq = q_seqlen_list[i]
                tri = torch.ones((qseq, qseq)).half()
                tri = torch.triu(tri, 1)
                tri *= -60000
                mask[i][-qseq:, -qseq:] = tri
        else:
            mask = torch.zeros(size=(num_tokens, max_k_seqlen))
            prev_qseq = 0
            for i in range(batch_size):
                qseq = q_seqlen_list[i]
                tri = torch.ones((qseq, qseq)).half()
                tri = torch.triu(tri, 1)
                tri *= -60000
                mask[prev_qseq:(prev_qseq + qseq), -qseq:] = tri
                prev_qseq += qseq
        ref_output = torch.zeros_like(query)
        self.ref_single_query_cached_kv_attention(
            ref_output,
            query,
            key_cache,
            value_cache,
            block_tables,
            q_seqlen_list,
            k_seqlen_list,
            mask,
            masktype
        )
        query = query.reshape(num_tokens, num_heads * head_size)
        ref_output = ref_output.reshape(num_tokens, num_heads * head_size)
        tokens_pad = (num_tokens + 15) // 16 * 16
        max_k_seq_pad = (max_k_seqlen + 15) // 16 * 16
        max_q_pad = (max_q + 15) // 16 * 16
        query_pad = torch.zeros((1, tokens_pad, num_heads * head_size)).half()
        query_pad[:, :num_tokens, :] = query
        o_pad = torch.zeros((1, tokens_pad, num_heads * head_size)).half()
        o_pad[:, :num_tokens, :] = ref_output
        self.golden_out = o_pad.view(1, tokens_pad, num_heads * head_size // 16, 16).permute(0, 2, 1, 3)
        self.q = query_pad.view(1, tokens_pad, num_heads * head_size // 16, 16).permute(0, 2, 1, 3)
        self.key_cache = key_cache.view(num_blocks, block_size, kv_heads * head_size // 16, 16).permute(0, 2, 1, 3)
        self.value_cache = value_cache.view(num_blocks, block_size, kv_heads * head_size // 16, 16).permute(0, 2, 1, 3)
        if masktype == "norm":
            mask_pad = torch.zeros((batch_size, max_q_pad, max_k_seq_pad))
            mask_pad[:,:max_q, :max_k_seqlen] = mask
            self.mask = mask_pad.reshape((batch_size, max_q_pad, max_k_seq_pad // 16, 16)).permute(0, 2, 1, 3)
        else:
            mask_pad = torch.zeros((1, tokens_pad, max_k_seq_pad))
            mask_pad[0][:num_tokens, :max_k_seqlen] = mask
            self.mask = mask_pad.reshape((1, tokens_pad, max_k_seq_pad // 16, 16)).permute(0, 2, 1, 3)
        self.block_tables = torch.from_numpy(np.array(block_tables).astype(np.int32))
        self.q_seqlen_list = torch.from_numpy(np.array(q_seqlen_list).astype(np.int32))
        self.k_seqlen_list = torch.from_numpy(np.array(k_seqlen_list).astype(np.int32))
        return query.view(num_tokens, num_heads, head_size), self.key_cache, self.value_cache, self.block_tables, self.k_seqlen_list, \
            self.mask.half(), self.q_seqlen_list, ref_output.view(num_tokens, num_heads, head_size)
 
 
batch = 4
block_size = 128
num_blocks = 1024
kv_heads = 8
num_heads = 8
head_size = 128
k_seqlen = 256
q_seqlen_list=[random.randint(1, 128) for _ in range(batch)]
k_seqlen_list=[random.randint(128, k_seqlen) for _ in range(batch)]

mask_type = "la"
pa = UnpadPagedAttention() 
data = pa.calc_data(num_heads, kv_heads, num_blocks, block_size, head_size, q_seqlen_list, k_seqlen_list, mask_type)
data_generator = PagedAttentionDataGenerator()
data_generator.data_type = torch.float16
data_generator.is_int8_flag = False
data_generator.max_context_len = k_seqlen

OP_NAME = "PagedAttentionOperation"
PARAM = json.dumps({"headNum": num_heads, "qkScale": (1 / float(math.sqrt(head_size))), "kvHeadNum": kv_heads, "maskType": 3, "calcType": 1})
RUN_PARAM = json.dumps({"contextLens": data[4].tolist(), "qLens": data[6].tolist(), "maskType": 3})
 
 
in_tensors = [tensor.npu().contiguous() for tensor in data]
class TestPagedAttentionAttentionOperation(operation_test.OperationTest):
    def golden_calc(self, input_tensors):
        return [in_tensors[-1]]
 
    def golden_compare(self, out_tensor, golden_out_tensor):
        ratios = [0.001, 0.001, 0.005, 0.005]
        return data_generator.compare_output_data(out_tensor.cpu(), golden_out_tensor.cpu(), ratios)
 
 
    def test(self):
        if not operation_test.get_soc_version() == 'Ascend310P':
            print("this testcase only supports Ascend310P")
            return
        in_tensors[1] = torch_npu.npu_format_cast(in_tensors[1], 29)
        in_tensors[2] = torch_npu.npu_format_cast(in_tensors[2] , 29)
        in_tensors[5] = torch_npu.npu_format_cast(in_tensors[5] , 29)
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM,
            [in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4], in_tensors[5], in_tensors[6]])
 
 
 
if __name__ == '__main__':
    unittest.main()