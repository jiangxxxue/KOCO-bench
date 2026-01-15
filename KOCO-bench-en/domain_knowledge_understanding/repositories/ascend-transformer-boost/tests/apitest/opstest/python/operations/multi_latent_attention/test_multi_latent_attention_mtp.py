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
import os
import unittest
import math
import numpy as np
import torch
import random
import json
import torch.nn.functional as F
import torch_npu
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test
from precision_calcu import *

class TestPagedAttentionMLA(operation_test.OperationTest):

    def compare_output_data(self, out, golden, ratios):
        error_count = 0
        strict_error_count = 0
        fp16_min_normal = 1.0 / (1 << 14)
        golden = golden.flatten().to(torch.float32)
        out = out.flatten().to(torch.float32)
        len = out.shape[0]
        diff = torch.abs(golden - out)
        max_diff = diff.max().item()
        limit_error = torch.maximum(torch.abs(golden * ratios[0]), torch.tensor(ratios[1]))
        strict_limit_error = torch.maximum(torch.abs(golden * ratios[2]), torch.tensor(ratios[3]))
        error_count = torch.gt(diff, limit_error).sum().item()
        strict_error_count = torch.gt(diff, strict_limit_error).sum().item()
        print(f"maxDiff {max_diff}")
        print("1/1000 Accuracy is %f",  1 - float(error_count) / len)
        print("5/1000 Accuracy is %f",  1 - float(strict_error_count) / len)
        if self.data_type == torch.bfloat16:
            print("accuracy is correct in old standard: %r", (float(strict_error_count) / len) <= ratios[2])
        else:
            print("accuracy is correct in old standard: %r", (float(strict_error_count) / len) <= ratios[0])
        calc_times = self.head_size_qk * self.max_context_len + 4
        if self.data_type == torch.bfloat16:
            if calc_times < 2048:
                error = 2**(-7)
            else :
                error = 2**(-6)
            error_threshold = torch.clamp(torch.abs(golden), min = 1) * error
            res = (diff <= error_threshold).all().item()
            print("accuracy is correct in new standard: %r", res)
            return res
        else:
            if calc_times < 2048:
                error = 2**(-8)
            else :
                error = 2**(-7)
            error_threshold = torch.clamp(torch.abs(golden), min = 1) * error
            res = (diff <= error_threshold).all().item()
            print("accuracy is correct in new standard: %r", res)
            return res

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

    def softmax_numpy(self, sim):
        sim = sim.cpu().numpy()
        row_max = np.max(sim, axis=-1, keepdims=True)
        sim_sub = sim - row_max
        sim_sub = np.exp(sim_sub)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        soft_res = sim_sub / row_sum
        return soft_res, row_max + np.log(row_sum)

    def ref_masked_attention(self,
            query,  # (q_seqlen, num_heads, head_size)
            key,    # (k_seqlen, kv_heads, head_size)
            value,
            scale: float,
            mask    # (q_seqlen, k_seqlen)
    ):
        # Q * K.T
        query = query
        query = torch.permute(query, (1, 0, 2))
        key = torch.permute(key, (1, 2, 0))
        sim_high = self.group_matmul(query.shape[0], key.shape[0], query, key)  # (head_num, q_seqlen, k_seqlen)
        sim_high = sim_high * scale
        if mask is not None:
            sim_high = sim_high + (mask[:sim_high.shape[-2], :sim_high.shape[-1]] * self.post_mask_factor).to(torch.float32)
        
        # softmax
        p_high, lse_high = self.softmax_numpy(sim_high)
        p = torch.from_numpy(p_high).to(query.dtype)
        p_high = torch.from_numpy(p_high).to(torch.float32)
        lse_high = torch.permute(torch.from_numpy(lse_high).to(torch.float32), (1, 0, 2))  # (q_seqlen, head_num, 1)
        value = torch.permute(value, (1, 0, 2))
        out_high = self.group_matmul(query.shape[0], key.shape[0], p_high, value)
        out = self.group_matmul(query.shape[0], key.shape[0], p, value)
        out_high = torch.permute(out_high, (1, 0, 2))
        out = torch.permute(out, (1, 0, 2))
        out = out.to(query.dtype)
        return out, out_high, lse_high.to(query.dtype), lse_high

    def ref_single_query_cached_kv_attention(self,
        output,
        true_out,
        lse,
        true_lse,
        query,
        key_cache,    # (num_blocks, block_size, num_heads, head_size)
        value_cache,  # (num_blocks, block_size, num_heads, head_size)
        block_tables,
        q_seqlen_list,
        k_seqlen_list,
        global_mask,
        mask_type
    ) -> None:
        num_heads = query.shape[1]
        kv_heads = value_cache.shape[2]
        head_size_qk = key_cache.shape[3]
        head_size_vo = value_cache.shape[3]
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
                k = k.reshape(kv_heads, head_size_qk)
                keys.append(k)

                v = value_cache[block_number, block_offset, :, :]
                v = v.reshape(kv_heads, head_size_vo)
                values.append(v)
            keys = torch.stack(keys, axis=0)
            values = torch.stack(values, axis=0)
            scale = 1.0 / (head_size_qk ** 0.5)
            if mask_type == 1:
                mask = global_mask[k_seqlen - q_seqlen : k_seqlen, :k_seqlen]  # prefill, decoder, prefill + decoder
            elif mask_type == 3:
                mask = global_mask[cu_seqlen : (cu_seqlen + q_seqlen), :]  # lookahead: cur_q: cur_q + q
            else:
                mask = None
            out, out_high, lse_i, lse_i_high = self.ref_masked_attention(q, keys, values, scale, mask)
            lse_i = lse_i.reshape(-1, num_heads, 1)
            lse[cu_seqlen: cu_seqlen + q_seqlen, :, :] = lse_i
            lse_i_high = lse_i_high.reshape(-1, num_heads, 1)
            true_lse[cu_seqlen: cu_seqlen + q_seqlen, :, :] = lse_i_high
            out = out.reshape(-1, num_heads, head_size_vo)
            out_high = out_high.reshape(-1, num_heads, head_size_vo)
            output[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out
            true_out[cu_seqlen: cu_seqlen + q_seqlen, :, :] = out_high
            cu_seqlen += q_seqlen_list[i]

    def calc_data(self,
                  num_heads: int,
                  kv_heads: int,
                  num_blocks: int,
                  block_size: int,
                  head_size_qk: int,
                  head_size_vo: int,
                  q_seqlen_list: int,
                  k_seqlen_list: int,
                  mask_type,
                  dtype = torch.float16,
                  calcType=0
    ):
        self.data_type = dtype
        self.head_size_qk = head_size_qk
        self.calcType = calcType
        q_min_range = -1.0
        q_max_range = 1.0
        kv_min_range = -1.0
        kv_max_range = 1.0
        num_tokens = np.array(q_seqlen_list).sum()
        batch_size = len(q_seqlen_list)
        self.max_context_len = max(k_seqlen_list)
        query = torch.from_numpy(np.random.uniform(q_min_range, q_max_range, size=(num_tokens, num_heads, head_size_qk))).to(dtype)
    
        key_cache = torch.from_numpy(np.random.uniform(kv_min_range, kv_max_range, size=(num_blocks, block_size, kv_heads, head_size_qk))).to(dtype)

        # (num_blocks, block_size, num_heads, head_size)
        value_cache = key_cache[:, :, :, :head_size_vo]

        max_k_seqlen = max(k_seqlen_list)
        max_num_blocks_per_seq = (max_k_seqlen + block_size - 1) // block_size
        block_tables = []   # (num_tokens, max_num_blocks_per_seq）
        for i in range(batch_size):
            block_table = [
                # max_num_blocks_per_seq * i + j for j in range(max_num_blocks_per_seq)
                random.randint(0, num_blocks - 1) for j in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)

        self.pre_mask_factor = -10000.0
        self.post_mask_factor = 1.0

        if mask_type == 1:
            mask = np.ones(shape=(max_k_seqlen, max_k_seqlen)).astype(np.float16)
            mask = np.triu(mask, 1)
            mask *= -10000.0
            mask = torch.from_numpy(mask).to(dtype)
        elif mask_type == 3:
            mask = np.zeros(shape=(num_tokens, max_k_seqlen)).astype(np.float16)
            pre_qseqlen = 0
            for i in range(batch_size):
                qseqlen = q_seqlen_list[i]
                kseqlen = k_seqlen_list[i]
                tri = np.ones((qseqlen, qseqlen))
                tri = np.triu(tri, 1)
                tri *= self.pre_mask_factor
                mask[pre_qseqlen:(pre_qseqlen + qseqlen), kseqlen-qseqlen:kseqlen] = tri
                pre_qseqlen += qseqlen
            
            mask = torch.from_numpy(mask).to(dtype)
        elif mask_type == 0:
            mask = None

        logging.info(f'input info: {num_tokens}, {num_heads}, {kv_heads}, {head_size_qk}, {head_size_vo}, {block_size}, {num_blocks}')
        shape_out = (num_tokens, num_heads, head_size_vo)
        ref_output = torch.zeros(shape_out, dtype=dtype)
        true_out = torch.zeros(shape_out, dtype=torch.float32)
        lse = torch.zeros((num_tokens, num_heads, 1), dtype=dtype)
        true_lse = torch.zeros((num_tokens, num_heads, 1), dtype=torch.float32)
        self.ref_single_query_cached_kv_attention(
            ref_output,
            true_out,
            lse,
            true_lse,
            query,
            key_cache,
            value_cache,
            block_tables,
            q_seqlen_list,
            k_seqlen_list,
            mask,
            mask_type
        )

        self.q_split1, self.q_split2 = torch.split(query, [512, 64], dim=2)
        self.key_cache_split1, self.key_cache_split2 = torch.split(key_cache, [512, 64], dim=3)

        self.q = query
        self.num_tokens = num_tokens
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.block_tables = np.array(block_tables).astype(np.int32)
        self.q_seqlen_list = np.array(q_seqlen_list).astype(np.int32)
        self.k_seqlen_list = np.array(k_seqlen_list).astype(np.int32)
        self.mask = mask
        self.golden_out = ref_output
        self.true_out = true_out
        self.lse = lse
        self.true_lse = true_lse

    def gen_mask(self, q_len, dtype):
        """
        generate mask used in mask free feature
        :param q_len: q_len
        :param dtype: mask data type
        :return: constructed mask, e.g. when q_len=2, returned as
        [[-10000.0 -10000.0 -10000.0 ... -10000.0],
         [0        -10000.0 -10000.0 ... -10000.0],
         [0               0 -10000.0 ... -10000.0],
         ...
         [0               0        0 ... -10000.0],
         [0               0        0 ...        0]]
 
        """
        mask_free = np.full((125 + 2*q_len, 128), -10000.0).astype(np.float16)
        mask_free = np.triu(mask_free, 2 - q_len)
        return torch.from_numpy(mask_free).to(dtype)

    def golden_calc(self, in_tensors):
        golden_out = torch.tensor(self.golden_out)
        result = [golden_out]
        if self.calcType == 3:
            result.append(self.lse)
        return result


    def golden_compare(self, out_tensors, golden_tensors):
        if self.compare_index == 0:
            result_double = compare_cv(self.true_out.npu(), golden_tensors.npu(), out_tensors.npu())
            result_old = self.compare_output_data(out_tensors.npu(), golden_tensors.npu(), [0.001, 0.001, 0.005, 0.005])
        else:
            result_double = compare_cv(self.true_lse.npu(), golden_tensors.npu(), out_tensors.npu())
            result_old = self.compare_output_data(out_tensors.npu(), golden_tensors.npu(), [0.001, 0.001, 0.005, 0.005])
        self.compare_index += 1
        return (result_double or result_old)

    def test_mla_split_mtp_head32_fp16(self):
        self.compare_index = 0
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        batch = 10
        q_seqlen = [2] * batch
        num_tokens = np.array(q_seqlen).sum()
        num_heads = 32
        kv_heads = 1
        block_size = 128
        head_size_qk = 576
        head_size_vo = 512
        num_blocks = 64
        k_seqlen = [64] * batch
        tor = 1.0 / (head_size_qk ** 0.5)
        dtype = torch.float16
        mask_type = 0

        self.calc_data(num_heads, kv_heads, num_blocks, block_size, head_size_qk, head_size_vo,
                       q_seqlen, k_seqlen, mask_type, dtype)

        OP_NAME = "MultiLatentAttentionOperation"
        PARAM = json.dumps({"headNum": num_heads, "qkScale":tor, "kvHeadNum":kv_heads, "maskType": 0, "calcType": 1, "cacheMode": 1})
        RUN_PARAM = json.dumps({"contextLens": self.k_seqlen_list.tolist(), "qSeqlen": self.q_seqlen_list.tolist()})
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM,
                                [self.q_split1.npu(),
                                 self.q_split2.npu(),
                                 self.key_cache_split1.npu(),
                                 self.key_cache_split2.npu(),
                                 torch.tensor(self.block_tables).int().npu(),
                                 torch.tensor(self.k_seqlen_list).npu(),
                                 torch.tensor(self.q_seqlen_list).npu()])

    def test_mla_split_mtp_head128_fp16(self):
        self.compare_index = 0
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        batch = 10
        q_seqlen = [2] * batch
        num_tokens = np.array(q_seqlen).sum()
        num_heads = 32
        kv_heads = 1
        block_size = 128
        head_size_qk = 576
        head_size_vo = 512
        num_blocks = 64
        k_seqlen = [256] * batch
        tor = 1.0 / (head_size_qk ** 0.5)
        dtype = torch.float16
        mask_type = 0

        self.calc_data(num_heads, kv_heads, num_blocks, block_size, head_size_qk, head_size_vo,
                       q_seqlen, k_seqlen, mask_type, dtype)

        OP_NAME = "MultiLatentAttentionOperation"
        PARAM = json.dumps({"headNum": num_heads, "qkScale":tor, "kvHeadNum":kv_heads, "maskType": 0, "calcType": 1, "cacheMode": 1})
        RUN_PARAM = json.dumps({"contextLens": self.k_seqlen_list.tolist(), "qSeqlen": self.q_seqlen_list.tolist()})
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM,
                                [self.q_split1.npu(),
                                 self.q_split2.npu(),
                                 self.key_cache_split1.npu(),
                                 self.key_cache_split2.npu(),
                                 torch.tensor(self.block_tables).int().npu(),
                                 torch.tensor(self.k_seqlen_list).npu(),
                                 torch.tensor(self.q_seqlen_list).npu()])

    def test_mla_split_mtp_mask_head32_fp16(self):
        self.compare_index = 0
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        batch = 2
        q_seqlen = [2] * batch
        num_tokens = np.array(q_seqlen).sum()
        num_heads = 128
        kv_heads = 1
        block_size = 128
        head_size_qk = 576
        head_size_vo = 512
        num_blocks = 64
        k_seqlen = [257] * batch
        tor = 1.0 / (head_size_qk ** 0.5)
        dtype = torch.float16
        mask_type = 3

        self.calc_data(num_heads, kv_heads, num_blocks, block_size, head_size_qk, head_size_vo,
                       q_seqlen, k_seqlen, mask_type, dtype)

        OP_NAME = "MultiLatentAttentionOperation"
        PARAM = json.dumps({"headNum": num_heads, "qkScale":tor, "kvHeadNum":kv_heads, "calcType": 1, "maskType": 1, "cacheMode": 1})
        RUN_PARAM = json.dumps({"contextLens": self.k_seqlen_list.tolist(), "qSeqlen": self.q_seqlen_list.tolist(), "maskType": 1})
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM,
                                [self.q_split1.npu(),
                                 self.q_split2.npu(),
                                 self.key_cache_split1.npu(),
                                 self.key_cache_split2.npu(),
                                 torch.tensor(self.block_tables).int().npu(),
                                 torch.tensor(self.k_seqlen_list).npu(),
                                 self.mask.npu(),
                                 torch.tensor(self.q_seqlen_list).npu()])

    def test_mla_split_mtp_mask_head32_fp16_isRing(self):
        self.compare_index = 0
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        batch = 2
        q_seqlen = [2] * batch
        num_tokens = np.array(q_seqlen).sum()
        num_heads = 128
        kv_heads = 1
        block_size = 128
        head_size_qk = 576
        head_size_vo = 512
        num_blocks = 64
        k_seqlen = [257] * batch
        tor = 1.0 / (head_size_qk ** 0.5)
        dtype = torch.float16
        mask_type = 3

        self.calc_data(num_heads, kv_heads, num_blocks, block_size, head_size_qk, head_size_vo,
                       q_seqlen, k_seqlen, mask_type, dtype, 3)

        OP_NAME = "MultiLatentAttentionOperation"
        PARAM = json.dumps({"headNum": num_heads, "qkScale":tor, "kvHeadNum":kv_heads, "calcType": 3, "maskType": 1, "cacheMode": 1})
        RUN_PARAM = json.dumps({"contextLens": self.k_seqlen_list.tolist(), "qSeqlen": self.q_seqlen_list.tolist(), "maskType": 1})
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM,
                                [self.q_split1.npu(),
                                 self.q_split2.npu(),
                                 self.key_cache_split1.npu(),
                                 self.key_cache_split2.npu(),
                                 torch.tensor(self.block_tables).int().npu(),
                                 torch.tensor(self.k_seqlen_list).npu(),
                                 self.mask.npu(),
                                 torch.tensor(self.q_seqlen_list).npu()])

    def test_mla_split_mtp_mask_head128_fp16(self):
        self.compare_index = 0
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        batch = 2
        q_seqlen = [2] * batch
        num_tokens = np.array(q_seqlen).sum()
        num_heads = 128
        kv_heads = 1
        block_size = 128
        head_size_qk = 576
        head_size_vo = 512
        num_blocks = 64
        k_seqlen = [256] * batch
        tor = 1.0 / (head_size_qk ** 0.5)
        dtype = torch.float16
        mask_type = 3

        self.calc_data(num_heads, kv_heads, num_blocks, block_size, head_size_qk, head_size_vo,
                       q_seqlen, k_seqlen, mask_type, dtype)

        OP_NAME = "MultiLatentAttentionOperation"
        PARAM = json.dumps({"headNum": num_heads, "qkScale":tor, "kvHeadNum":kv_heads, "calcType": 1, "maskType": 1, "cacheMode": 1})
        RUN_PARAM = json.dumps({"contextLens": self.k_seqlen_list.tolist(), "qSeqlen": self.q_seqlen_list.tolist(), "maskType": 1})
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM,
                                [self.q_split1.npu(),
                                 self.q_split2.npu(),
                                 self.key_cache_split1.npu(),
                                 self.key_cache_split2.npu(),
                                 torch.tensor(self.block_tables).int().npu(),
                                 torch.tensor(self.k_seqlen_list).npu(),
                                 self.mask.npu(),
                                 torch.tensor(self.q_seqlen_list).npu()])



    def test_mla_split_mtp_mask_free(self):
        self.compare_index = 0
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        batch = 2
        q_seqlen = [2] * batch
        num_tokens = np.array(q_seqlen).sum()
        num_heads = 128
        kv_heads = 1
        block_size = 128
        head_size_qk = 576
        head_size_vo = 512
        num_blocks = 64
        k_seqlen = [127] * batch
        tor = 1.0 / (head_size_qk ** 0.5)
        dtype = torch.float16
        mask_type = 3

        self.calc_data(num_heads, kv_heads, num_blocks, block_size, head_size_qk, head_size_vo,
                       q_seqlen, k_seqlen, mask_type, dtype)
        mask_free = self.gen_mask(q_seqlen[0], dtype)
        OP_NAME = "MultiLatentAttentionOperation"
        PARAM = json.dumps({"headNum": num_heads, "qkScale":tor, "kvHeadNum":kv_heads, "calcType": 1, "maskType": 2, "cacheMode": 1})
        RUN_PARAM = json.dumps({"contextLens": self.k_seqlen_list.tolist(), "qSeqlen": self.q_seqlen_list.tolist(), "maskType": 2})
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM,
                                [self.q_split1.npu(),
                                 self.q_split2.npu(),
                                 self.key_cache_split1.npu(),
                                 self.key_cache_split2.npu(),
                                 torch.tensor(self.block_tables).int().npu(),
                                 torch.tensor(self.k_seqlen_list).npu(),
                                 mask_free.npu(),
                                 torch.tensor(self.q_seqlen_list).npu()])

if __name__ == '__main__':
    unittest.main()
