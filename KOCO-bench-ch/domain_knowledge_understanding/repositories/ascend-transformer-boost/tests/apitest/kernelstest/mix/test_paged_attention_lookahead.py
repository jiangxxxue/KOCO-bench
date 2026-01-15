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


class TestUnpadPagedAttention(op_test.OpTest):
    def compare_output_data(self, out, golden, ratios):
        error_count = 0
        strict_error_count = 0
        max_diff = -1.0
        golden = golden.to(torch.float32)
        out = out.to(torch.float32)

        diff = torch.abs(golden - out)
        len = out.shape[0] * out.shape[1] * out.shape[2]
        max_diff = diff.max().item()
        limit_error = torch.maximum(torch.abs(golden * ratios[0]), torch.tensor(ratios[1]))
        strict_limit_error = torch.maximum(torch.abs(golden * ratios[2]), torch.tensor(ratios[3]))
        error_count = torch.gt(diff, limit_error).sum().item()
        strict_error_count = torch.gt(diff, strict_limit_error).sum().item()
        logging.info(f"maxDiff {max_diff}")
        logging.info(f"rate: {float(strict_error_count)}")
        logging.info(f"rate: {float(len)}")
        logging.info(f"rate: {float(strict_error_count) / len}")
        logging.info("1/1000 Accuracy is %f",  1 - float(error_count) / len)
        logging.info("5/1000 Accuracy is %f",  1 - float(strict_error_count) / len)
        logging.info("accuracy is correct: %r", (float(strict_error_count) / len) <= ratios[0])
        calc_times = out.shape[2] * self.max_context_len + 4
        if calc_times < 2048:
            error = 2**(-8)
        else :
            error = 2**(-7)
        error_threshold = torch.clamp(torch.abs(golden), min = 1) * error
        return (diff <= error_threshold).all()

    def group_matmul(self, head, kv_head, A, B):
        group_num = head // kv_head
        score = None
        for i in range(kv_head):
            group_score = np.matmul(A[i * group_num: (i + 1) * group_num, :, :].astype(np.float32),
                                    B[i : (i + 1), :, :].astype(np.float32)).astype(np.float16)
            if score is None:
                score = group_score
            else:
                score = np.concatenate((score, group_score), 0)
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
            keys = np.stack(np.array(keys), axis=0)
            values = np.stack(np.array(values), axis=0)
            scale = 1.0 / (head_size ** 0.5)
            if mask_type == 0:
                mask = global_mask[k_seqlen - q_seqlen : k_seqlen, :k_seqlen]  # prefill, decoder, prefill + decoder
            elif mask_type == 3:
                mask = global_mask[cu_seqlen : (cu_seqlen + q_seqlen), :]  # lookahead: cur_q: cur_q + q
            out = self.ref_masked_attention(q, keys, values, scale, mask)
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
        block_tables = []   # (num_tokens, max_num_blocks_per_seq）
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

        logging.debug(f'input info: {num_tokens}, {num_heads}, {kv_heads}, {head_size}, {block_size}, {num_blocks}, {dtype}, mask.shape = {mask.shape}, {mask}')
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

    def golden_calc(self, in_tensors):
        golden_out = torch.tensor(self.golden_out)
        return [golden_out]

    def golden_compare(self, out_tensors, golden_tensors):
        logging.debug(f"out_tensors: {out_tensors}")
        logging.debug(f"golden_tensors:{golden_tensors}")
        return self.compare_output_data(out_tensors[0].half(), golden_tensors[0].half(), [0.001, 0.001, 0.005, 0.005])

    @op_test.only_910b
    def test_pa_fp16_case_encoder_decoder1(self):
        np.random.seed(0)
        # q_seqlen_list = [1000, 400, 100, 50, 16, 1]
        # k_seqlen_list = [1000, 800, 400, 100, 50, 300]
        # q_seqlen_list = [1]
        # k_seqlen_list = [300]
        # q_seqlen_list = [1000, 400]
        # k_seqlen_list = [1000, 800]
        q_seqlen_list = [1, 15, 30, 6]
        k_seqlen_list = [10, 64, 64, 64]

        num_heads = 32
        kv_heads = 32
        head_size = 128

        num_blocks = 512
        block_size = 128
        tor = 1.0 / (head_size ** 0.5)
        dtype = "float16"

        self.calc_data(num_heads, kv_heads, num_blocks, block_size, head_size, q_seqlen_list, k_seqlen_list, 3, dtype)

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
        attention_out = np.zeros_like(self.q)
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
    def test_pa_fp16_case_decoder_split_kv_large_headnum(self):
        np.random.seed(0)
        q_seqlen_list = [1, 1, 1]
        k_seqlen_list = [13333, 23333, 33331]

        num_heads = 40
        kv_heads = 40
        head_size = 128

        num_blocks = 1234
        block_size = 128
        tor = 1.0 / (head_size ** 0.5)
        dtype = "float16"

        self.calc_data(num_heads, kv_heads, num_blocks, block_size, head_size, q_seqlen_list, k_seqlen_list, 3, dtype)

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
        attention_out = np.zeros_like(self.q)
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
    def test_pa_fp16_case_encoder_decoder1_blocksize16(self):
        np.random.seed(0)
        q_seqlen_list = [1, 15, 30, 6]
        k_seqlen_list = [10, 64, 64, 64]

        num_heads = 32
        kv_heads = 32
        head_size = 128

        num_blocks = 512
        block_size = 16
        tor = 1.0 / (head_size ** 0.5)
        dtype = "float16"

        self.calc_data(num_heads, kv_heads, num_blocks, block_size, head_size, q_seqlen_list, k_seqlen_list, 3, dtype)

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
        attention_out = np.zeros_like(self.q)
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
    def test_pa_fp16_case_square_mask_blk32(self):
        np.random.seed(0)
        q_seqlen_list = [15, 103, 1024]*1
        k_seqlen_list = [64 ,103, 1025]*1

        num_heads = 32
        kv_heads = 32
        head_size = 128

        num_blocks = 256
        block_size = 32
        tor = 1.0 / (head_size ** 0.5)
        dtype = "float16"

        self.calc_data(num_heads, kv_heads, num_blocks, block_size, head_size, q_seqlen_list, k_seqlen_list, 0, dtype)

        OP_NAME = "PagedAttentionOperation"
        OP_PARAM = {"type": 2002, "kvHead":kv_heads, "headSize":num_heads, "tor": tor, "maskType": 1, "qSeqLen": q_seqlen_list, "kvSeqLen": k_seqlen_list}
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
        attention_out = np.zeros_like(self.q)
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
    def test_pa_fp16_case_la_mask(self):
        q_seqlen_list = [256,256,15] * 1
        k_seqlen_list = [512,512,2048] *1

        num_heads = 1
        kv_heads = 1
        head_size = 128

        num_blocks = 256
        block_size = 128
        tor = 1.0 / (head_size ** 0.5)
        dtype = "float16"

        self.calc_data(num_heads, kv_heads, num_blocks, block_size, head_size, q_seqlen_list, k_seqlen_list, 3, dtype)

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
        attention_out = np.zeros_like(self.q)
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
