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
np.random.seed(1)
random.seed(1)
MAX_SEQ_LEN = 1024
torch.set_printoptions(sci_mode=False)

class TestUnpadPagedAttention(op_test.OpTest):
    def compare_output_data(self, out, golden, ratios):
        error_count = 0
        strict_error_count = 0
        max_diff = -1.0
        fp16_min_normal = 1.0 / (1 << 14)
        out, golden = out.numpy().flatten(), golden.numpy().flatten()
        # logging.info(f"out shape {out.shape}")
        total = len(out)
        for i in range(len(out)):
            actual_item = out[i]
            golden_item = golden[i]
            limit_error = abs(float(golden_item)) * ratios[0]
            limit_error = max(limit_error, ratios[0])
            strict_limit_error = max(ratios[2], abs(float((golden_item)) * ratios[2]))
            diff = abs(float(actual_item) - float(golden_item))

            if (diff > max_diff):
                max_diff = diff
            if (diff > strict_limit_error):
                strict_error_count += 1
            if (diff > limit_error):
                error_count += 1
        logging.info(f"rate: {float(strict_error_count)}")
        logging.info(f"rate: {float(total)}")
        logging.info(f"rate: {float(strict_error_count) / total}")
        return (float(strict_error_count) / total) <= ratios[2]

    def group_matmul(self, head, kv_head, A, B):
        group_num = head // kv_head
        score = None
        for i in range(kv_head):
            group_score = np.matmul(A[i * group_num: (i + 1) * group_num, :, :].to(torch.float32),
                                    B[i : (i + 1), :, :].to(torch.float32)).to(torch.float32)
            if score is None:
                score = group_score
            else:
                score = torch.cat((score, group_score), 0)
        # print(f"sim score: {score.shape}")
        return score

    def ref_masked_attention(self,
            query,  # (q_seqlen, num_heads, head_size)
            key,    # (k_seqlen, kv_heads, head_size)
            value,
            scale: float,
            mask,   # (q_seqlen, k_seqlen)
            mask_data_type:torch.bfloat16
    ):
        # Q * K.T
        query = query * scale
        query = torch.permute(query, (1, 0, 2))
        key = torch.permute(key, (1, 2, 0))
        sim = self.group_matmul(query.shape[0], key.shape[0], query, key).to(mask_data_type)  # (head_num, q_seqlen, k_seqlen)
        sim = sim.to(torch.float32) + mask[:sim.shape[-2], :sim.shape[-1]].to(torch.float32)
        # softmax
        sim = sim.numpy()
        row_max = np.max(sim, axis=-1, keepdims=True)
        sim -= row_max
        sim = np.exp(sim)
        row_sum = np.sum(sim, axis=-1, keepdims=True)
        p = sim / row_sum
        p = torch.from_numpy(p).to(mask_data_type)
        # p = p.astype("float16")
        # P * V
        value = torch.permute(value, (1, 0, 2))
        out = self.group_matmul(query.shape[0], key.shape[0], p, value)
        out = torch.permute(out, (1, 0, 2))
        return out

    def ref_single_query_cached_kv_attention(self,
        output,
        query,
        key_cache,    # (num_blocks, block_size, num_heads, head_size)
        value_cache,  # (num_blocks, block_size, num_heads, head_size)
        block_tables,
        q_seqlen_list,
        k_seqlen_list,
        global_mask,
        mask_data_type=torch.bfloat16
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
            mask = global_mask[cu_seqlen : (cu_seqlen + q_seqlen), :]  # lookahead: cur_q: cur_q + q
            logging.debug(f"q_seqlen: {q_seqlen}, query.shape: {q.shape}, {q.dtype}, "
                f"k_seqlen: {k_seqlen}, keys.shape: {keys.shape},"
                f"key block_num: {(k_seqlen + block_size - 1) // block_size}, tail: {k_seqlen % block_size},")
                # f"mask.shape: {mask.shape}, {mask}")
            out = self.ref_masked_attention(q, keys, values, scale, mask, mask_data_type)
            out = out.reshape(-1, num_heads, head_size)
            logging.debug(f"out.shape: {out.shape}")
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
                  dtype: str,
                  mask_data_type = torch.bfloat16):

        num_tokens = np.array(q_seqlen_list).sum()
        batch_size = len(q_seqlen_list)
        max_k_seqlen = max(k_seqlen_list)
        query = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_tokens, num_heads, head_size))).to(mask_data_type)
        # (num_blocks, block_size, num_heads, head_size)
        kv_range = 1.0
        kv_type = mask_data_type
        key_cache = torch.from_numpy(np.random.uniform(-kv_range, kv_range, size=(num_blocks, block_size, kv_heads, head_size))).to(kv_type)
        # (num_blocks, block_size, num_heads, head_size)
        value_cache = torch.from_numpy(np.random.uniform(-kv_range, kv_range, size=(num_blocks, block_size, kv_heads, head_size))).to(kv_type)

        max_k_seqlen = max(k_seqlen_list)
        max_num_blocks_per_seq = (max_k_seqlen + block_size - 1) // block_size
        block_tables = []   # (num_tokens, max_num_blocks_per_seq）
        for i in range(batch_size):
            block_table = [
                max_num_blocks_per_seq * i + j for j in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)

        self.data_type = dtype
        mask = np.zeros((num_tokens, max_k_seqlen), dtype=np.float16)
        logging.debug(f"norm_mask.shape = {mask.shape}")
        pre_qseqlen = 0
        for i in range(batch_size):
            qseqlen = q_seqlen_list[i]
            tri = torch.ones((qseqlen, qseqlen))
            tri = torch.triu(tri, 1)
            tri *= -10000.0
            mask[pre_qseqlen:(pre_qseqlen + qseqlen), -qseqlen:] = tri
            pre_qseqlen += qseqlen

        mask = torch.from_numpy(mask).to(mask_data_type)
        logging.debug(f'input info: {num_tokens}, {num_heads}, {kv_heads}, {head_size}, {block_size}, {num_blocks}, {dtype}, mask.shape = {mask.shape}, {mask}')
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
            mask_data_type
        )

        self.q = query
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.block_tables = np.array(block_tables).astype(np.int32)
        self.q_seqlen_list = np.array(q_seqlen_list).astype(np.int32)
        self.k_seqlen_list = np.array(k_seqlen_list).astype(np.int32)
        self.mask = mask
        self.golden_out = ref_output


    def golden_calc(self, in_tensors):
        golden_out = torch.tensor(self.golden_out)
        return [golden_out]

    def golden_compare(self, out_tensors, golden_tensors):
        logging.debug(f"out_tensors: {out_tensors}")
        logging.debug(f"golden_tensors:{golden_tensors}")
        return self.compare_output_data(out_tensors[0].float(), golden_tensors[0].float(), [0.001, 0.001, 0.005, 0.005])


    @op_test.only_910b
    def test_pa_bf16_case_encoder_decoder1(self):
        np.random.seed(0)
        q_seqlen_list = [1, 4, 9]  # (24)
        k_seqlen_list = [15, 14, 29]  # (74)
        # q_seqlen_list = [3]  # (24)
        # k_seqlen_list = [150]
        num_heads = 32
        kv_heads = 32
        head_size = 128

        num_blocks = 512
        block_size = 128
        tor = 1.0 / (head_size ** 0.5)
        dtype = "float16"

        self.calc_data(num_heads, kv_heads, num_blocks, block_size, head_size, q_seqlen_list, k_seqlen_list, dtype, torch.bfloat16)

        OP_NAME = "PagedAttentionOperation"
        OP_PARAM = {"type": 2002, "kvHead":kv_heads, "headSize":num_heads, "maskType": 3, "tor": tor, "qSeqLen": q_seqlen_list, "kvSeqLen": k_seqlen_list}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.key_cache.shape}")
        logging.debug(f"v shape: {self.value_cache.shape}")
        logging.debug(f"block_tables shape: {self.block_tables}")
        logging.debug(f"k_seqlen_list shape: {self.k_seqlen_list}")
        logging.debug(f"mask: {self.mask.shape}")
        logging.debug(f"q_seqlen_list shape: {self.q_seqlen_list}")
        logging.debug(f"numHeads: {num_heads}, kvHead: {kv_heads}"
              f", headSize: {head_size}, numBlocks: {num_blocks}, blockSize: {block_size}")
        attention_out = torch.zeros_like(self.q)
        attention_out[:] = 0.1

        return self.execute(
                [
                    torch.tensor(self.q), torch.tensor(self.key_cache),
                    torch.tensor(self.value_cache), torch.tensor(self.block_tables).int(),
                    torch.tensor(self.mask),
                    torch.tensor([], dtype=torch.float16),
                    torch.tensor([], dtype=torch.float16),
                    torch.tensor([], dtype=torch.float16),
                    torch.tensor([], dtype=torch.float16),
                    torch.tensor([], dtype=torch.float16),
                    torch.tensor([], dtype=torch.float16),
                    torch.tensor([], dtype=torch.float32)
                ],
                [
                    torch.tensor(attention_out)
                ]
                )


    @op_test.only_910b
    def test_pa_bf16_case_encoder_decoder1_blockSize16(self):
        np.random.seed(0)
        q_seqlen_list = [1, 4, 9]  # (24)
        k_seqlen_list = [15, 14, 29]  # (74)
        num_heads = 32
        kv_heads = 32
        head_size = 128

        num_blocks = 512
        block_size = 16
        tor = 1.0 / (head_size ** 0.5)
        dtype = "float16"

        self.calc_data(num_heads, kv_heads, num_blocks, block_size, head_size, q_seqlen_list, k_seqlen_list, dtype, torch.bfloat16)

        OP_NAME = "PagedAttentionOperation"
        OP_PARAM = {"type": 2002, "kvHead":kv_heads, "headSize":num_heads, "maskType": 3, "tor": tor, "qSeqLen": q_seqlen_list, "kvSeqLen": k_seqlen_list}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.key_cache.shape}")
        logging.debug(f"v shape: {self.value_cache.shape}")
        logging.debug(f"block_tables shape: {self.block_tables}")
        logging.debug(f"k_seqlen_list shape: {self.k_seqlen_list}")
        logging.debug(f"mask: {self.mask.shape}")
        logging.debug(f"q_seqlen_list shape: {self.q_seqlen_list}")
        logging.debug(f"numHeads: {num_heads}, kvHead: {kv_heads}"
              f", headSize: {head_size}, numBlocks: {num_blocks}, blockSize: {block_size}")
        attention_out = torch.zeros_like(self.q)
        attention_out[:] = 0.1

        return self.execute(
                [
                    torch.tensor(self.q), torch.tensor(self.key_cache),
                    torch.tensor(self.value_cache), torch.tensor(self.block_tables).int(),
                    torch.tensor(self.mask),
                    torch.tensor([], dtype=torch.float16),
                    torch.tensor([], dtype=torch.float16),
                    torch.tensor([], dtype=torch.float16),
                    torch.tensor([], dtype=torch.float16),
                    torch.tensor([], dtype=torch.float16),
                    torch.tensor([], dtype=torch.float16),
                    torch.tensor([], dtype=torch.float32)
                ],
                [
                    torch.tensor(attention_out)
                ]
                )

    @op_test.only_910b
    def test_pa_bf16_case_la_mask(self):
        q_seqlen_list = [256,256,15] * 1
        k_seqlen_list = [512,512,2048] *1

        num_heads = 1
        kv_heads = 1
        head_size = 128

        num_blocks = 256
        block_size = 128
        tor = 1.0 / (head_size ** 0.5)
        dtype = "float16"

        self.calc_data(num_heads, kv_heads, num_blocks, block_size, head_size, q_seqlen_list, k_seqlen_list, dtype, torch.bfloat16)

        OP_NAME = "PagedAttentionOperation"
        OP_PARAM = {"type": 2002, "kvHead":kv_heads, "headSize":num_heads, "tor": tor, "maskType": 3, "qSeqLen": q_seqlen_list, "kvSeqLen": k_seqlen_list}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])

        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.key_cache.shape}")
        logging.debug(f"v shape: {self.value_cache.shape}")
        logging.debug(f"block_tables shape: {self.block_tables}")
        logging.debug(f"k_seqlen_list shape: {self.k_seqlen_list}")
        logging.debug(f"mask: {self.mask.shape}")
        logging.debug(f"q_seqlen_list shape: {self.q_seqlen_list}")
        logging.debug(f"numHeads: {num_heads}, kvHead: {kv_heads}"
            f", headSize: {head_size}, numBlocks: {num_blocks}, blockSize: {block_size}")
        attention_out = torch.zeros_like(self.q)
        attention_out[:] = 0.1

        return self.execute(
                [
                    torch.tensor(self.q), torch.tensor(self.key_cache),
                    torch.tensor(self.value_cache), torch.tensor(self.block_tables).int(),
                    torch.tensor(self.mask),
                    torch.tensor([], dtype=torch.float16),
                    torch.tensor([], dtype=torch.float16),
                    torch.tensor([], dtype=torch.float16),
                    torch.tensor([], dtype=torch.float16),
                    torch.tensor([], dtype=torch.float16),
                    torch.tensor([], dtype=torch.float16),
                    torch.tensor([], dtype=torch.float32)
                ],
                [
                    torch.tensor(attention_out)
                ]
                )

if __name__ == '__main__':
    unittest.main()