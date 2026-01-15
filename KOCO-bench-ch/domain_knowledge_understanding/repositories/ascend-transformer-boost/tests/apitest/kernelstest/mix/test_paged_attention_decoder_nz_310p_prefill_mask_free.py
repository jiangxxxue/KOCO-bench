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
import sys, os
import unittest
import math
import numpy as np
import op_test
import torch
import random
import sys
import numpy as np
import math
import itertools
np.random.seed(1)
random.seed(1)
MAX_SEQ_LEN = 1024
rand_list = np.random.uniform(0.0, 2.0, size=100).astype(np.float16)
rand_list = [x if x > 1.0 else 1.0 for x in rand_list]

class TestPagedAttentionNz(op_test.OpTest):

    def compare_output_data(self, out, golden, ratios,data_type, is_int8_flag):
        error_count = 0
        strict_error_count = 0
        fp16_min_normal = 1.0 / (1 << 14)
        golden = golden.to(torch.float32)
        out = out.to(torch.float32)
        len = out.shape[0] * out.shape[1] * out.shape[2]
        diff = torch.abs(golden - out)
        max_diff = diff.max().item()
        limit_error = torch.maximum(torch.abs(golden * ratios[0]), torch.tensor(ratios[1]))
        strict_limit_error = torch.maximum(torch.abs(golden * ratios[2]), torch.tensor(ratios[3]))
        error_count = torch.gt(diff, limit_error).sum().item()
        strict_error_count = torch.gt(diff, strict_limit_error).sum().item()
        print(f"maxDiff {max_diff}")
        print("1/1000 Accuracy is %f",  1 - float(error_count) / len)
        print("5/1000 Accuracy is %f",  1 - float(strict_error_count) / len)
        # 旧精度标准：双千分之五
        if data_type == torch.bfloat16 or is_int8_flag:
            print("accuracy is correct: %r", (float(strict_error_count) / len) <= ratios[2])
        else:
            print("accuracy is correct: %r", (float(strict_error_count) / len) <= ratios[0])
        # 新精度标准 参考精度标准v0.3浮点计算单标杆
        # 计算次数 两个matmul + 一个softmax
        error = 2**(-8)
        error_threshold = torch.clamp(torch.abs(golden), min = 1) * error
        return (diff <= error_threshold).all()

    def shape_nd_to_nz(self, shape, dtype='float16'):
        assert len(shape) >= 2
        batch = shape[:-2]  # 最后两维nd->nz
        a, b = shape[-2], shape[-1]
        a0, b0 = 16, 16
        return list(batch) + [math.ceil(b / b0), math.ceil(a / a0), a0, b0]

    def gen_axes_for_transpose(self,offset, base):
        return [x for x in range(offset)] + [x + offset for x in base]

    def convert_nd_to_nz(self, x):
        array_trans = self.gen_axes_for_transpose(len(x.shape) - 2, [2, 0, 1, 3])  # (m1, m0, n1, n0) -> (n1, m1, m0, n0)
        x_shape = self.shape_nd_to_nz(x.shape, dtype=x.dtype)
        *_, n1, m1, m0, n0 = x_shape
        return x.reshape(x_shape[:-4] + [m1, m0, n1, n0]).transpose(*array_trans)  # x原始需要对齐，才能reshape

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
                kseq = k_seqlen_list[i]
                start = kseq - qseq
                tri = torch.ones((qseq, qseq)).half()
                tri = torch.triu(tri, 1)
                tri *= -60000
                mask[prev_qseq:(prev_qseq + qseq), start:kseq] = tri
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
        tokens_pad = (num_tokens + 15) // 16 * 16
        query = query.reshape(num_tokens, num_heads * head_size)
        query_pad = torch.zeros((1, tokens_pad, num_heads * head_size)).half()
        query_pad[:, :num_tokens, :] = query
        
        query_nz = self.convert_nd_to_nz(query_pad.numpy())
        query_nz = query_nz.reshape(1, num_heads * head_size // 16, tokens_pad, 16).astype(np.float16)
        query_nz = np.ascontiguousarray(query_nz)
        self.q = torch.from_numpy(query_nz)

        key_cache = key_cache.reshape(num_blocks, block_size, -1)
        key_cache_nz = self.convert_nd_to_nz(key_cache.numpy())
        key_cache_nz = key_cache_nz.reshape(num_blocks, kv_heads * head_size // 16,block_size, 16)
        key_cache_nz = np.ascontiguousarray(key_cache_nz)
        self.key_cache = torch.from_numpy(key_cache_nz)

        value_cache = value_cache.reshape(num_blocks, block_size, -1)
        value_cache_nz = self.convert_nd_to_nz(value_cache.numpy())
        value_cache_nz = value_cache_nz.reshape(num_blocks, kv_heads * head_size // 16, block_size, 16)
        value_cache_nz = np.ascontiguousarray(value_cache_nz)
        self.value_cache = torch.from_numpy(value_cache_nz)

        max_k_seq_pad = (max_k_seqlen + 15) // 16 * 16
        max_q_pad = (max_q + 15) // 16 * 16
        
        if masktype == "norm":
            mask_pad = torch.zeros((batch_size, max_q_pad, max_k_seq_pad))
            mask_pad[:,:max_q, :max_k_seqlen] = mask
            mask_pad_nz = self.convert_nd_to_nz(mask_pad.numpy())
            mask_pad_nz = mask_pad_nz.reshape((batch_size, max_k_seq_pad // 16, max_q_pad,  16))
            mask_pad_nz = np.ascontiguousarray(mask_pad_nz)
            self.mask = torch.from_numpy(mask_pad_nz)
        else:
            seq_len = 128
            # 创建一个矩阵，初始为全 0
            mask = np.zeros((seq_len, seq_len), dtype=np.float16)
            # 构造严格上三角（不包括对角线）的 mask，赋值为 -inf
            mask[np.triu_indices(seq_len, k=1)] = 1
            mask = torch.from_numpy(mask)
            mask *= -60000

            mask = mask.reshape(1, seq_len, seq_len)
            
            mask_pad_nz = self.convert_nd_to_nz(mask.numpy())
            mask_pad_nz = mask_pad_nz.reshape((1, seq_len // 16, seq_len, 16))
            mask_pad_nz = np.ascontiguousarray(mask_pad_nz)
            self.mask = torch.from_numpy(mask_pad_nz)
        
        self.block_tables = torch.from_numpy(np.array(block_tables).astype(np.int32))
        self.q_seqlen_list = torch.from_numpy(np.array(q_seqlen_list).astype(np.int32))
        self.k_seqlen_list = torch.from_numpy(np.array(k_seqlen_list).astype(np.int32))

        ref_output = ref_output.reshape(num_tokens, num_heads * head_size)
        o_pad = torch.zeros((1, tokens_pad, num_heads * head_size)).half()
        o_pad[:, :num_tokens, :] = ref_output
        ref_output_nz = self.convert_nd_to_nz(o_pad.numpy())
        ref_output_nz = ref_output_nz.reshape(tokens_pad, num_heads, head_size) 
        ref_output_nz = np.ascontiguousarray(ref_output_nz)
        self.ref_output = torch.from_numpy(ref_output_nz)   


        return self.q, self.key_cache, self.value_cache, self.block_tables, self.k_seqlen_list, \
            self.mask.half(), self.q_seqlen_list, self.ref_output
 

    def golden_calc(self, in_tensors):
        return [self.ref_output]

    def golden_compare(self, out_tensors, golden_tensors):
        return self.compare_output_data(out_tensors[0].half(), golden_tensors[0].half(), [0.0004, 0.0004, 0.005, 0.005],torch.float16, False)

    def get_case_from_json(self):  
        json_str = {  
        "batch": [1, 2],  
        "head_comb":[[8,8],[7,1]],  
        "headdim":[64, 128],
        "kvSeqlen":[1, 11]    
        }  
        case_list = list(itertools.product(json_str["batch"], json_str["head_comb"], json_str["headdim"], json_str["kvSeqlen"]))  
        return case_list

    @op_test.only_310p
    def test_paged_attention_nz_case0(self):

        cases = self.get_case_from_json()
        for case in cases:
            batch=case[0]
            block_size = 128
            num_blocks = 1024
            kv_heads=case[1][1]

            num_heads =case[1][0]
            head_size = case[2]

            q_seqlen_list=[128*1 + case[3]] * batch
            k_seqlen_list=[128*2 + case[3]] * batch
            mask_type = "la"
            dtype = "float16"
            self.data_type = torch.float16
            data = self.calc_data(num_heads, kv_heads, num_blocks, block_size, head_size, q_seqlen_list, k_seqlen_list, mask_type)
            in_tensors = [tensor.npu().contiguous() for tensor in data]

            OP_NAME = "PagedAttentionOperation"
            OP_PARAM = {"tor": (1.0 / (head_size ** 0.5)), "headSize": num_heads, "kvHead": kv_heads, "maskType": 4, "scaleType": 0, "type": 2003, "qSeqLen":q_seqlen_list}

            self.set_param(OP_NAME, OP_PARAM)
            self.set_input_formats([self.format_nz, self.format_nz,self.format_nz, self.format_nd, self.format_nd, self.format_nz, self.format_nd])
            self.set_output_formats([self.format_nz])

            attention_out = np.zeros_like(self.ref_output)
            attention_out[:] = 0.0
            try:
                self.execute(
                            [
                                in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4], in_tensors[5], in_tensors[6]
                            ],
                            [
                                torch.tensor(attention_out).half()
                            ]
                            )
            except Exception as e:
                logging.error(f"process case (batch: {batch} embeddim:{head_size} head_num: {num_heads} kv_head:{kv_heads} q-seqlen:{q_seqlen_list} kv_seqlen:{k_seqlen_list} occur error!")

if __name__ == '__main__':
    unittest.main()
