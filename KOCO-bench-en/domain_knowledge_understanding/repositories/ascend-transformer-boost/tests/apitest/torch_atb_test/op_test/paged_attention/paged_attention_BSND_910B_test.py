#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import torch
import torch_npu
import torch_atb
import math
import random
import numpy as np
import unittest
import logging
import re

def is910B():
    device_name = torch.npu.get_device_name()
    return (re.search("Ascend910B", device_name, re.I) and len(device_name) > 10)

class TestPagedAttentionBNSDDataGenerator():

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
        logging.info("1/1000 Accuracy is %f", 1 - float(error_count) / len)
        logging.info("5/1000 Accuracy is %f", 1 - float(strict_error_count) / len)
        if self.data_type == torch.bfloat16 or self.is_int8_flag:
            logging.info("accuracy is correct in old standard: %r", (float(strict_error_count) / len) <= ratios[2])
        else:
            logging.info("accuracy is correct in old standard: %r", (float(strict_error_count) / len) <= ratios[0])
        calc_times = self.head_size * self.max_context_len + 4
        if self.data_type == torch.bfloat16:
            if calc_times < 2048:
                error = 2 ** (-7)
            else:
                error = 2 ** (-6)
            error_threshold = torch.clamp(torch.abs(golden), min=1) * error
            res = (diff <= error_threshold).all().item()
            logging.info("accuracy is correct in new standard: %r", res)
            return res
        else:
            if calc_times < 2048:
                error = 2 ** (-8)
            else:
                error = 2 ** (-7)
            error_threshold = torch.clamp(torch.abs(golden), min=1) * error
            res = (diff <= error_threshold).all().item()
            logging.info("accuracy is correct in new standard: %r", res)
            return res

    def get_alibi_slopes(self, n_heads):
        n = 2 ** math.floor(math.log2(n_heads))
        m0 = 2.0 ** (-8.0 / n)
        slopes = torch.pow(m0, torch.arange(1, n + 1))
        if n < n_heads:
            m1 = 2.0 ** (-4.0 / n)
            mm = torch.pow(m1, torch.arange(1, 1 + 2 * (n_heads - n), 2))
            slopes = torch.cat([slopes, mm])
        # slopes = torch.ones(n_heads)
        return slopes

    def group_mm_torch(self, heads, group_num, A, B, razor_mod, is_k):
        group_head = heads // group_num
        score_high = None
        for i in range(group_num):
            if self.is_int8_flag:
                int8_B = B[i: (i + 1), :, :, ]
                head_dim = int8_B.shape[2]
                int32_B = torch.matmul(torch.eye(int8_B.shape[1]).to(torch.float32), int8_B.to(torch.float32)).to(
                    torch.int32)
                if is_k:
                    if self.has_bias:
                        int32_B = int32_B + self.offset1[(i + razor_mod) * head_dim: (i + razor_mod + 1) * head_dim]
                    fp32_B = int32_B.to(torch.float32) * self.de_scale1_fp32[
                                                         (i + razor_mod) * head_dim: (i + razor_mod + 1) * head_dim]
                    fp32_B = torch.permute(fp32_B, (0, 2, 1))
                else:
                    if self.has_bias:
                        int32_B = int32_B + self.offset2[(i + razor_mod) * head_dim: (i + razor_mod + 1) * head_dim]
                    fp32_B = int32_B.to(torch.float32) * self.de_scale2_fp32[
                                                         (i + razor_mod) * head_dim: (i + razor_mod + 1) * head_dim]
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
                             mask_data_type=torch.bfloat16
                             ):
        # Q * K.T
        query = query
        query = torch.permute(query, (1, 0, 2))
        if not self.is_int8_flag:
            key = torch.permute(key, (1, 2, 0))  # 0 1 2
        else:
            key = torch.permute(key, (1, 0, 2))
        sim_high = self.group_mm_torch(query.shape[0], key.shape[0], query, key, razor_mod,
                                       1)  # (head_num, q_seqlen, k_seqlen)

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
                                             mask_dim=4,
                                             mask_data_type=torch.bfloat16
                                             ) -> None:
        mask_index_coff = 1
        if self.compressHead:
            query = query.view(self.num_tokens * self.kv_heads, self.num_heads // self.kv_heads, self.head_size)
            output = output.view(self.num_tokens * self.kv_heads, self.num_heads // self.kv_heads, self.head_size)
            true_out = true_out.view(self.num_tokens * self.kv_heads, self.num_heads // self.kv_heads, self.head_size)
            if mask_dim == 4:
                mask_shape = mask.shape
                mask = mask.view(mask_shape[0] * self.kv_heads, self.num_heads // self.kv_heads, 1,
                                 self.max_context_len)
            else:
                mask_index_coff = self.kv_heads
        num_heads = query.shape[1]
        kv_heads = value_cache.shape[2]
        head_size = value_cache.shape[3]
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
                v = v.reshape(kv_heads, head_size)
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
                out, out_high = self.ref_masked_attention(q, keys, values, scale, mask[i, :, :, :context_len],
                                                          razor_rope, razor_offset_list, razor_mod, mask_data_type)
                out = out.reshape(num_heads, head_size)
            elif mask_dim == 3:
                out, out_high = self.ref_masked_attention(q, keys, values, scale,
                                                          mask[i // mask_index_coff, :, :context_len], razor_rope,
                                                          razor_offset_list, razor_mod, mask_data_type)
                out = out.reshape(num_heads, head_size)
            else:
                out, out_high = self.ref_masked_attention(q, keys, values, scale, mask, razor_rope, razor_offset_list,
                                                          razor_mod, mask_data_type)
                out = out.reshape(num_heads, head_size)
            out_high = out_high.reshape(num_heads, head_size)
            output[index] = out.to(mask_data_type)
            true_out[index] = out_high
            index = index + 1

    def calc_data(self, num_tokens, num_heads, kv_heads, head_size, block_size, num_blocks, k_seqlen, \
                  dtype, mask_dim=4, mask_data_type=torch.bfloat16, \
                  dynamic_batch=False, dynamic_seqlen=None, is_int8_flag=False, has_bias=False,
                  compressHead=False, razor_rope=False, blocknum=20, is_quant_flag=0, is_quant_offiline=0, scaleType=0):
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.num_tokens = num_tokens
        self.compressHead = compressHead
        self.head_size = head_size
        self.scaleType = scaleType
        self.group_num = num_heads / kv_heads
        self.scaleType = scaleType
        logging.info(
            f'input info: {num_tokens}, {num_heads}, {kv_heads}, {head_size}, {block_size}, {num_blocks}, {k_seqlen}, {dtype}')

        q_min_range = -1.0
        q_max_range = 1.0
        kv_min_range = -1.0
        kv_max_range = 1.0
        kv_type = dtype
        self.is_quant_flag = is_quant_flag
        self.is_quant_offiline = is_quant_offiline
        if self.is_quant_flag:
            q_min_range = -5
            q_max_range = 5
            kv_min_range = -5
            kv_max_range = 5
            dtype = torch.int8
            kv_type = torch.int8
        if is_int8_flag:
            kv_min_range = -4
            kv_max_range = 4
            kv_type = torch.int8
        query = torch.from_numpy(
            np.random.uniform(q_min_range, q_max_range, size=(num_tokens, num_heads, head_size))).to(dtype)
        # (num_blocks, block_size, num_heads, head_size)
        if not compressHead:
            key_cache = torch.from_numpy(
                np.random.uniform(kv_min_range, kv_max_range, size=(num_blocks, block_size, kv_heads, head_size))).to(
                kv_type)
            # # (num_blocks, block_size, num_heads, head_size)
            value_cache = torch.from_numpy(
                np.random.uniform(kv_min_range, kv_max_range, size=(num_blocks, block_size, kv_heads, head_size))).to(
                kv_type)
            # (num_blocks, block_size, num_heads, head_size)
        else:
            key_cache = torch.from_numpy(np.random.uniform(kv_min_range, kv_max_range,
                                                           size=(num_blocks * kv_heads, block_size, 1, head_size))).to(
                kv_type)
            # # (num_blocks, block_size, num_heads, head_size)
            value_cache = torch.from_numpy(np.random.uniform(kv_min_range, kv_max_range, size=(
                num_blocks * kv_heads, block_size, 1, head_size))).to(kv_type)
            # (num_blocks, block_size, num_heads, head_size)
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
                alibi_bias = alibi_slopes.reshape(-1, 1, 1) * alibi_bias.reshape(1, 1, -1)  # (head_num, 1, context)
                mask[i, :, :, :context_len] = alibi_bias
            mask = torch.from_numpy(mask).to(mask_data_type)
        # normal mask headnum, 1, maxS
        elif mask_dim == 3:
            mask = np.zeros((batch, 1, max_context_len), dtype=np.float16)
            for i in range(batch):
                mask[i, :, :i] = -10000
            mask = torch.from_numpy(mask).to(mask_data_type)
        else:  # no mask
            mask = None

        if compressHead:
            context_lens = [val for val in context_lens for _ in range(kv_heads)]
        batch = len(context_lens)
        max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
        block_tables = []  # （num_tokens, max_num_blocks_per_seq）
        for _ in range(batch):
            block_table = [
                random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)
        self.is_int8_flag = is_int8_flag

        if is_int8_flag:
            de_scale1_fp32 = np.random.randint(-1, 2, size=(kv_heads * head_size)).astype(np.float32)
            de_scale1_int64 = self.process_deq_scale(de_scale1_fp32)

            de_scale2_fp32 = np.random.randint(-1, 2, size=(kv_heads * head_size)).astype(np.float32)
            de_scale2_int64 = self.process_deq_scale(de_scale2_fp32)

            offset1 = np.random.randint(-20, 20, size=(kv_heads * head_size)).astype(np.int32)

            offset2 = np.random.randint(-20, 20, size=(kv_heads * head_size)).astype(np.int32)

            self.de_scale1_int64 = torch.tensor(list(de_scale1_int64), dtype=torch.int64)
            self.de_scale2_int64 = torch.tensor(list(de_scale2_int64), dtype=torch.int64)
            self.de_scale1_fp32 = torch.from_numpy(de_scale1_fp32)
            self.de_scale2_fp32 = torch.from_numpy(de_scale2_fp32)
            self.offset1 = torch.from_numpy(offset1)
            self.offset2 = torch.from_numpy(offset2)
            self.has_bias = has_bias

        if self.is_quant_flag:
            self.de_scale1_fp32 = torch.from_numpy(
                np.random.uniform(-5 / 127, 5 / 127, size=(num_heads)).astype(np.float32)).to(torch.float32)
            self.de_scale2_fp32 = torch.from_numpy(
                np.random.uniform(-5 / 127, 5 / 127, size=(num_heads)).astype(np.float32)).to(torch.float32)
            self.scale = torch.from_numpy(np.random.uniform(0, 127, size=(num_heads)).astype(np.float32)).to(
                torch.float32)
            isLongSeq = max_context_len > blocknum * 128 * 2 and num_tokens < blocknum * 0.8
            if num_tokens * num_heads < 0.8 * blocknum or isLongSeq:
                self.kvsplit, self.kv_split_per_core = self.getkvsplit(num_tokens, num_heads, max_context_len,
                                                                       block_size, blocknum, isLongSeq)
            else:
                self.kvsplit = 1
                self.kv_split_per_core = max_context_len
            self.head_num_move = self.get_head_num_move(num_heads, kv_heads, head_size, head_size)
            self.block_size_calc = self.get_blockszie_calc(max_context_len, block_size, head_size, head_size)
            self.block_size = block_size

        ref_output = torch.zeros_like(query).to(torch.float32)
        true_out = torch.zeros_like(query, dtype=torch.float32)

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

    def test_pa_bf16_case_norm_mask_not_bnsd(self):
        # mte2bound shape
        num_tokens = 16
        num_heads = 40
        kv_heads = 10
        block_size = 256
        head_size = 256
        num_blocks = 64
        k_seqlen = 3000
        tor = 1.0 / (head_size ** 0.5)
        mask_dim = 3
        dtype = torch.bfloat16

        self.calc_data(num_tokens, num_heads, kv_heads, head_size, block_size, num_blocks, k_seqlen, dtype, mask_dim)
        # return q, key_cache, value_cache, block_tables, contex_lens, alib_mask, golden_out

        return [self.q.npu(), self.key_cache.npu(), self.value_cache.npu(), torch.from_numpy(self.block_tables).npu(),
                torch.from_numpy(self.contex_lens), self.alib_mask.npu(), self.golden_out.npu()]


def main():
    if not is910B():
        print("This test case only supports 910B")
        return True
    print("----------- paged_attention test begin ------------")
    page_attention_param = torch_atb.PagedAttentionParam()
    page_attention_param.head_num = 40
    page_attention_param.qk_scale = (1 / float(math.sqrt(256)))
    page_attention_param.kv_head_num = 10
    page_attention_param.mask_type = torch_atb.PagedAttentionParam.MaskType.MASK_TYPE_NORM
    page_attention_param.input_layout = torch_atb.InputLayout.TYPE_BSND
    logging.info("PagedAttentionParam: ", page_attention_param)
    page_attention = torch_atb.Operation(page_attention_param)
    pagedAttentionBNSDDataGenerator = TestPagedAttentionBNSDDataGenerator()
    data = pagedAttentionBNSDDataGenerator.test_pa_bf16_case_norm_mask_not_bnsd()
    in_tensors = [tensor for tensor in data]
    outputs = page_attention.forward([in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4],
                                      in_tensors[5]])
    logging.info("calc_outputs", outputs)
    logging.info("golden_output", in_tensors[6])
    print("----------- paged_attention test success ------------")

class TestSelfAttentionPAEncoder(unittest.TestCase):
    def test(self):
        torch.npu.synchronize()
        main()

if __name__ == "__main__":
    unittest.main()