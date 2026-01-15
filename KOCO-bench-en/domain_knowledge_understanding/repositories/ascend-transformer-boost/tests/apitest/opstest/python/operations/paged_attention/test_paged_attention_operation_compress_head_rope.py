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
sys.path.append('../')
import torch
import random
import sys
import numpy as np
import math
np.random.seed(1)
random.seed(1)
MAX_SEQ_LEN = 1024

import os
import json
import torch_npu
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

class PagedAttentionDataGenerator():

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
        logging.info(f"maxDiff {max_diff}")
        logging.info("1/1000 Accuracy is %f",  1 - float(error_count) / len)
        logging.info("5/1000 Accuracy is %f",  1 - float(strict_error_count) / len)
        if self.data_type == torch.bfloat16 or self.is_int8_flag:
            logging.info("accuracy is correct in old standard: %r", (float(strict_error_count) / len) <= ratios[2])
        else:
            logging.info("accuracy is correct in old standard: %r", (float(strict_error_count) / len) <= ratios[0])
        calc_times = self.head_size * self.max_context_len + 4
        if self.data_type == torch.bfloat16:
            if calc_times < 2048:
                error = 2**(-7)
            else :
                error = 2**(-6)
            error_threshold = torch.clamp(torch.abs(golden), min = 1) * error
            res = (diff <= error_threshold).all().item()
            logging.info("accuracy is correct in new standard: %r", res)
            return res
        else:
            if calc_times < 2048:
                error = 2**(-8)
            else :
                error = 2**(-7)
            error_threshold = torch.clamp(torch.abs(golden), min = 1) * error
            res = (diff <= error_threshold).all().item()
            logging.info("accuracy is correct in new standard: %r", res)
            return res

    def get_alibi_slopes(self, n_heads):
        n = 2 ** math.floor(math.log2(n_heads))
        m0 = 2.0 ** (-8.0 / n)
        slopes = torch.pow(m0, torch.arange(1, n + 1))
        if n < n_heads:
            m1 = 2.0 ** ( -4.0 / n)
            mm = torch.pow(m1, torch.arange(1, 1 + 2 * (n_heads - n), 2))
            slopes = torch.cat([slopes, mm])
        # slopes = torch.ones(n_heads)
        return slopes

    def group_mm_torch(self, heads, group_num, A, B, razor_mod, is_k):
        group_head = heads // group_num
        score_high = None
        for i in range(group_num):
            if self.is_int8_flag:
                int8_B = B[i: (i+1), :, :, ]
                head_dim = int8_B.shape[2]
                int32_B = torch.matmul(torch.eye(int8_B.shape[1]).to(torch.float32), int8_B.to(torch.float32)).to(torch.int32)
                if is_k:
                    if self.has_bias:
                        int32_B = int32_B + self.offset1[(i + razor_mod) * head_dim : (i + razor_mod + 1) * head_dim]
                    fp32_B = int32_B.to(torch.float32) * self.de_scale1_fp32[(i + razor_mod) * head_dim : (i + razor_mod + 1) * head_dim]
                    fp32_B = torch.permute(fp32_B, (0, 2, 1))
                else:
                    if self.has_bias:
                        int32_B = int32_B + self.offset2[(i + razor_mod) * head_dim : (i + razor_mod + 1) * head_dim]
                    fp32_B = int32_B.to(torch.float32) * self.de_scale2_fp32[(i + razor_mod) * head_dim : (i + razor_mod + 1) * head_dim]
                group_score_high = torch.matmul(A[i * group_head: (i + 1) * group_head, :, :].to(torch.float32),
                                            fp32_B)
            else:
                group_score_high = torch.matmul(A[i * group_head: (i + 1) * group_head, :, :].to(torch.float32),
                                           B[i:(i + 1), :, :].to(torch.float32))
            if score_high is None:
                score_high = group_score_high
            else:
                score_high = torch.cat((score_high, group_score_high), 0)
        return score_high


    def process_deq_scale(self, deq_scale) -> np.ndarray:
        new_deq_scale = np.frombuffer(deq_scale.tobytes(), dtype=np.uint32)
        return new_deq_scale.astype(np.uint64)

    def softmax(self, sim):
        row_max = torch.max(sim, axis=-1, keepdims=True)[0]
        sim_sub = sim - row_max
        sim_sub = torch.exp(sim_sub)
        row_sum = torch.sum(sim_sub, axis=-1, keepdims=True)
        soft_res = sim_sub / row_sum
        return soft_res

    def softmax_numpy(self, sim):
        sim = sim.cpu().numpy()
        row_max = np.max(sim, axis=-1, keepdims=True)
        sim_sub = sim - row_max
        sim_sub = np.exp(sim_sub)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        soft_res = sim_sub / row_sum
        return soft_res

    def ref_masked_attention(self,
            query,  # (1, num_heads, head_size)
            key,  # (context_len, kv_heads, head_size)
            value,
            scale: float,
            alibi_bias,
            razor_rope,
            razor_offset_list,
            razor_mod,
            mask_data_type = torch.bfloat16
    ):
        # Q * K.T
        query = query
        query = torch.permute(query, (1, 0, 2))
        if not self.is_int8_flag:
            key = torch.permute(key, (1, 2, 0))  # 0 1 2
        else:
            key = torch.permute(key, (1, 0, 2))
        sim_high = self.group_mm_torch(query.shape[0], key.shape[0], query, key, razor_mod, 1)  # (head_num, q_seqlen, k_seqlen)

        if razor_rope:
            razor_offset_list = razor_offset_list.view(1, 1, razor_offset_list.shape[0])
            sim_high = sim_high.to(torch.float32) + razor_offset_list

        sim_high = sim_high.to(torch.float32) * scale
        if alibi_bias is not None:
            sim_high = sim_high + alibi_bias.to(torch.float32)
        # softmax
        p_high = self.softmax_numpy(sim_high)
        p = torch.from_numpy(p_high).to(mask_data_type)
        p_high = torch.from_numpy(p_high)

        # P * V
        value = torch.permute(value, (1, 0, 2))
        out = self.group_mm_torch(query.shape[0], key.shape[0], p, value, razor_mod, 0)
        out_high = self.group_mm_torch(query.shape[0], key.shape[0], p_high, value, razor_mod, 0)
        out = torch.permute(out, (1, 0, 2))
        out_high = torch.permute(out_high, (1, 0, 2))
        return out, out_high

    def ref_single_query_cached_kv_attention(self,
            output,
            true_out,
            query,
            key_cache,  # (num_blocks, block_size, num_heads, head_size)
            value_cache,  # (num_blocks, block_size, num_heads, head_size)
            block_tables,
            context_lens,
            mask,
            razor_offset,
            razor_rope,
            mask_dim = 4,
            mask_data_type = torch.bfloat16    
    ) -> None:
        mask_index_coff = 1
        head_size = key_cache.shape[3]
        head_size_v = value_cache.shape[3]
        if self.compressHead:
            query = query.view(self.num_tokens * self.kv_heads, self.num_heads // self.kv_heads, self.head_size)
            output = output.view(self.num_tokens * self.kv_heads, self.num_heads // self.kv_heads, head_size_v)
            true_out = true_out.view(self.num_tokens * self.kv_heads, self.num_heads // self.kv_heads, head_size_v)
            if mask_dim == 4:
                mask_shape = mask.shape
                mask = mask.view(mask_shape[0] * self.kv_heads, self.num_heads // self.kv_heads, 1, self.max_context_len)
            else:
                mask_index_coff = self.kv_heads
        num_heads = query.shape[1]
        kv_heads = value_cache.shape[2]
        block_size = value_cache.shape[1]

        num_input_tokens = query.shape[0]
        index = 0
        razor_mod = 0
        for i in range(len(context_lens)):
            block_table = block_tables[i]
            context_len = int(context_lens[i])
            if context_len == 0:
                continue

            q = query[index].view(1, num_heads, head_size)
            keys = []
            values = []
            razor_offset_list = []
            for j in range(context_len):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size

                k = key_cache[block_number, block_offset, :, :]
                k = k.reshape(kv_heads, head_size)
                keys.append(k)

                v = value_cache[block_number, block_offset, :, :]
                v = v.reshape(kv_heads, head_size_v)
                values.append(v)

                if razor_rope:
                    offset = razor_offset[block_number, block_offset]
                    razor_offset_list.append(offset)

            keys = torch.stack(keys, axis=0)
            values = torch.stack(values, axis=0)
            if razor_rope:
                razor_mod = i % self.kv_heads
                razor_offset_list = torch.stack(razor_offset_list, axis=0)

            if self.compressHead:
                razor_mod = i % self.kv_heads
                
            scale = np.float32(1.0 / (head_size ** 0.5))
            if mask_dim == 4:
                out, out_high = self.ref_masked_attention(q, keys, values, scale, mask[i, :, :, :context_len], razor_rope, razor_offset_list, razor_mod, mask_data_type)
                out = out.reshape(num_heads, head_size_v)
            elif mask_dim == 3:
                out,out_high = self.ref_masked_attention(q, keys, values, scale, mask[i // mask_index_coff, :, :context_len], razor_rope, razor_offset_list, razor_mod, mask_data_type)
                out = out.reshape(num_heads, head_size_v)
            else:
                out,out_high = self.ref_masked_attention(q, keys, values, scale, mask, razor_rope, razor_offset_list, razor_mod, mask_data_type)
                out = out.reshape(num_heads, head_size_v)
            out_high = out_high.reshape(num_heads, head_size_v)
            output[index] = out.to(mask_data_type)
            true_out[index] = out_high
            index = index + 1

    def calc_data(self, num_tokens, num_heads, kv_heads, head_size, block_size, num_blocks, k_seqlen,\
                  dtype, mask_dim = 4, mask_data_type = torch.bfloat16,\
                  dynamic_batch = False, dynamic_seqlen = None, is_int8_flag = False, has_bias = False,
                  compressHead = False, razor_rope = False):
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.num_tokens = num_tokens
        self.compressHead = compressHead
        self.head_size = head_size
        head_size_v = head_size
        logging.debug(f'input info: {num_tokens}, {num_heads}, {kv_heads}, {head_size}, {block_size}, {num_blocks}, {k_seqlen}, {dtype}')

        query = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_tokens, num_heads, head_size))).to(dtype)
        # (num_blocks, block_size, num_heads, head_size)
        kv_range = 1.0
        kv_type = dtype
        if is_int8_flag:
            kv_range = 4.0
            kv_type = torch.int8
        if not compressHead:
            key_cache = torch.from_numpy(np.random.uniform(-kv_range, kv_range, size=(num_blocks, block_size, kv_heads, head_size))).to(kv_type)
            # (num_blocks, block_size, num_heads, head_size)
            value_cache = torch.from_numpy(np.random.uniform(-kv_range, kv_range, size=(num_blocks, block_size, kv_heads, head_size_v))).to(kv_type)
        else:
            key_cache = torch.from_numpy(np.random.uniform(-kv_range, kv_range, size=(num_blocks * kv_heads, block_size, 1, head_size))).to(kv_type)
            # (num_blocks, block_size, num_heads, head_size)
            value_cache = torch.from_numpy(np.random.uniform(-kv_range, kv_range, size=(num_blocks * kv_heads, block_size, 1, head_size_v))).to(kv_type)
        self.data_type = dtype

        razor_offset = torch.tensor([], dtype=torch.float32)
        if razor_rope:
            razor_offset = torch.zeros(num_blocks * kv_heads, block_size)
            mask = np.random.choice([False, True], size=num_blocks * kv_heads, p=[0.2, 0.8])

            random_indices = np.random.randint(0, block_size, size=np.sum(mask))
            random_values = np.random.uniform(0, 20, size=np.sum(mask))

            active_rows = np.where(mask)[0]
            razor_offset[active_rows, random_indices] = torch.from_numpy(random_values).to(torch.float32)
        
        if dynamic_batch:
            context_lens = dynamic_seqlen
        else:
            context_lens = [k_seqlen] * num_tokens
        max_context_len = max(context_lens)
        self.max_context_len = max_context_len
        batch = len(context_lens)

        # alibi mask
        if mask_dim == 4:
            mask = np.zeros((batch, num_heads, 1, self.max_context_len), dtype=np.float32)
            alibi_slopes = self.get_alibi_slopes(num_heads)
            for i, context_len in enumerate(context_lens):
                if context_len == 0:
                    continue
                position_ids = np.arange(context_len).astype(np.int32)
                alibi_bias = (position_ids - context_len + 1).astype(np.float32)
                alibi_bias = alibi_slopes.reshape(-1, 1, 1) * alibi_bias.reshape(1, 1, -1)   # (head_num, 1, context)
                mask[i, :, :, :context_len] = alibi_bias
            mask = torch.from_numpy(mask).to(mask_data_type)
        # normal mask
        elif mask_dim == 3:
            mask = np.zeros((batch, 1, max_context_len), dtype=np.float16)
            for i in range(batch):
                mask[i, :, :i] = -10000
            mask = torch.from_numpy(mask).to(mask_data_type)
        else: # no mask
            mask = None

        if compressHead:
            context_lens = [val for val in context_lens for _ in range(kv_heads)]
        batch = len(context_lens)
        max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
        block_tables = []   # （num_tokens, max_num_blocks_per_seq）
        for _ in range(batch):
            block_table = [
                random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)

        self.is_int8_flag = is_int8_flag
        if is_int8_flag:
            de_scale1_fp32 = np.random.randint(-1, 2, size=(kv_heads * head_size)).astype(np.float32)
            de_scale1_int64 = self.process_deq_scale(de_scale1_fp32)

            de_scale2_fp32 =  np.random.randint(-1, 2, size=(kv_heads * head_size)).astype(np.float32)
            de_scale2_int64 = self.process_deq_scale(de_scale2_fp32)

            offset1 = np.random.randint(-20, 20, size=(kv_heads * head_size)).astype(np.int32)

            offset2 = np.random.randint(-20, 20, size=(kv_heads * head_size)).astype(np.int32)

            self.de_scale1_int64 = torch.tensor(list(de_scale1_int64), dtype=torch.int64)
            self.de_scale2_int64 =  torch.tensor(list(de_scale2_int64), dtype=torch.int64)
            self.de_scale1_fp32 = torch.from_numpy(de_scale1_fp32)
            self.de_scale2_fp32 = torch.from_numpy(de_scale2_fp32)
            self.offset1 = torch.from_numpy(offset1)
            self.offset2 = torch.from_numpy(offset2)
            self.has_bias = has_bias


        ref_output = torch.zeros((num_tokens, num_heads, head_size_v))
        # true_out = torch.zeros_like(query, dtype=torch.float32)
        true_out = torch.zeros((num_tokens, num_heads, head_size_v), dtype=torch.float32)
        self.ref_single_query_cached_kv_attention(
            ref_output,
            true_out,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            mask,
            razor_offset,
            razor_rope,
            mask_dim,
            mask_data_type 
        )

        self.q = query
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.block_tables = np.array(block_tables).astype(np.int32)
        self.contex_lens = np.array(context_lens).astype(np.int32)
        self.alib_mask = mask
        self.golden_out = ref_output
        self.true_out = true_out
        self.razor_offset = razor_offset
    
    
    def test_pa_fp16_case_no_mask_multi_batch_compress_head_razor_rope(self):
        # self.set_support_910b_only()
        compressHead = True
        razor_rope = True
        num_tokens = 7
        num_heads = 36
        kv_heads = 4
        head_size = 128
        block_size = 128
        num_blocks = 128
        seqlen = 4097
        k_seqlen = seqlen if compressHead else math.ceil(seqlen / 0.3)
        tor = 1.0 / (head_size ** 0.5)
        dtype = torch.float16
        mask_dim = 0
        is_int8_flag = False
        has_bias = False

        self.calc_data(num_tokens, num_heads, kv_heads, head_size, block_size, num_blocks, k_seqlen, dtype, mask_dim, mask_data_type=dtype, is_int8_flag=is_int8_flag, has_bias=has_bias, compressHead = compressHead, razor_rope=razor_rope)


data_generator = PagedAttentionDataGenerator()
data_generator.test_pa_fp16_case_no_mask_multi_batch_compress_head_razor_rope()

(q, k, v, block_table, context_lens, razor_offset) = (
    data_generator.q,
    data_generator.key_cache,
    data_generator.value_cache,
    data_generator.block_tables,
    data_generator.contex_lens,
    data_generator.razor_offset
)

data = [q, k, v, torch.from_numpy(block_table), torch.from_numpy(context_lens), razor_offset]
in_tensors = [tensor.npu() for tensor in data]
_ = [print(tensor.dtype, tensor.device, tensor.shape) for tensor in in_tensors]

OP_NAME = "PagedAttentionOperation"
PARAM = json.dumps({"headNum":36, "qkScale": (1 / 128 ** 0.5), "kvHeadNum": 4, "maskType": 0, "compressType": 2})

RUN_PARAM = json.dumps({"contextLens": data_generator.contex_lens.tolist()})

class TestPagedAttentionAttentionOperationCompressHead(operation_test.OperationTest):
    def golden_calc(self, input_tensors):
        return [data_generator.golden_out]

    def golden_compare(self, out_tensor, golden_out_tensor):
        ratios = [0.001, 0.001, 0.005, 0.005]
        return data_generator.compare_output_data(out_tensor.cpu(), golden_out_tensor.cpu(), ratios)

    def test(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4], in_tensors[5]])


if __name__ == '__main__':
    unittest.main()