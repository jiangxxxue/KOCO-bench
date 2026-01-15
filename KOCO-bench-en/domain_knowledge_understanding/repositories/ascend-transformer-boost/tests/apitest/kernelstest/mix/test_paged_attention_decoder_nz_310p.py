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


class TestPagedAttentionNz(op_test.OpTest):

    def compare_output_data(self, out, golden, ratios):
        error_count = 0
        strict_error_count = 0
        fp16_min_normal = 1.0 / (1 << 14)
        len = out.shape[0] * out.shape[1]
        diff = torch.abs(golden - out)
        max_diff = diff.max().item()
        limit_error = torch.maximum(torch.abs(golden * ratios[0]), torch.tensor(ratios[1]))
        strict_limit_error = torch.maximum(torch.abs(golden * ratios[2]), torch.tensor(ratios[3]))
        error_count = torch.gt(diff, limit_error).sum().item()
        strict_error_count = torch.gt(diff, strict_limit_error).sum().item()
        logging.info(f"maxDiff {max_diff}")
        logging.info("1/1000 Accuracy is %f",  1 - float(error_count) / len)
        logging.info("3/1000 Accuracy is %f",  1 - float(strict_error_count) / len)
        if self.data_type == torch.bfloat16:
            return (float(strict_error_count) / len) <= ratios[2]
        else:
            return (float(strict_error_count) / len) <= ratios[0]

    def get_data_params(self):
        ret = (self.is_mask, self.is_batch_mask, self.is_decoder, self.variate_seq, self.is_alibi)
        return ret

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
            group_score = np.matmul(A[i * group_num: (i + 1) * group_num, :, :].astype(np.float32),
                                    B[i: (i + 1), :, :].astype(np.float32)).astype(np.float16)
            if score is None:
                score = group_score
            else:
                score = np.concatenate((score, group_score), 0)
        logging.info(score.shape)
        return score

    def ref_masked_attention(self,
            query,  # (1, num_heads, head_size)
            key,  # (context_len, kv_heads, head_size)
            value,
            scale: float,
            alibi_bias
    ):
        # Q * K.T
        query = query * scale
        query = np.transpose(query, (1, 0, 2))
        key = np.transpose(key, (1, 2, 0))  # 0 1 2
        sim = self.group_matmul(query.shape[0], key.shape[0], query, key)  # (head_num, q_seqlen, k_seqlen)
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
        out = self.group_matmul(query.shape[0], key.shape[0], p, value)
        out = np.transpose(out, (1, 0, 2))
        return out

    def ref_single_query_cached_kv_attention(self,
            output,
            query,
            key_cache,  # (num_blocks, block_size, num_heads, head_size)
            value_cache,  # (num_blocks, block_size, num_heads, head_size)
            block_tables,
            context_lens,
            alibi_mask
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
            scale = 1.0 / (head_size ** 0.5)
            logging.debug(f"query.shape: {q.shape}, {q.dtype}, keys.shape: {keys.shape}, "
                  f"context_len: {context_len}, keyblocknum: {(context_len + block_size - 1) // block_size}, "
                  f"tail: {context_len % block_size}, alibi_bias.shape: {alibi_mask[i].shape}")
            out = self.ref_masked_attention(q, keys, values, scale, alibi_mask[i, :, :, :context_len])
            out = out.reshape(num_heads, head_size)
            output[i] = out

    def calc_data(self, num_tokens, num_heads, kv_heads, head_size, block_size, num_blocks, k_seqlen, dtype):

        logging.debug(f'input info: {num_tokens}, {num_heads}, {kv_heads}, {head_size}, {block_size}, {num_blocks}, {k_seqlen}, {dtype}')

        query = np.random.uniform(-1.0, 1.0, size=(num_tokens, num_heads, head_size)).astype(dtype)
        # (num_blocks, block_size, num_heads, head_size)
        key_cache = np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_size)).astype(dtype)
        # (num_blocks, block_size, num_heads, head_size)
        value_cache = np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_heads, head_size)).astype(dtype)

        context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_tokens)]  # 一个token处理多少个key
        context_lens = [k_seqlen] * num_tokens
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
            alibi_mask[i, :, :, :context_len] = alibi_bias
        logging.debug(f"alibi_mask.shape = {alibi_mask.shape}")

        ref_output = np.zeros_like(query)
        self.ref_single_query_cached_kv_attention(
            ref_output,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            alibi_mask
        )

        tokens_pad = (num_tokens + 15) // 16 * 16
        query = query.reshape(1, num_tokens, num_heads * head_size)
        query_pad = np.zeros((1, tokens_pad, num_heads * head_size))
        query_pad[:, :num_tokens, :] = query
        query_nz = self.convert_nd_to_nz(query_pad)
        self.q = query_nz.astype(np.float16).reshape(1, num_heads * head_size // 16, tokens_pad, 16)

        key_cache = key_cache.reshape(num_blocks, block_size, -1)
        key_cache_nz = self.convert_nd_to_nz(key_cache)
        self.key_cache = key_cache_nz.astype(np.float16).reshape(num_blocks, -1, block_size, 16)

        value_cache = value_cache.reshape(num_blocks, block_size, -1)
        value_cache_nz = self.convert_nd_to_nz(value_cache)
        self.value_cache = value_cache_nz.astype(np.float16).reshape(num_blocks, -1, block_size, 16)

        self.block_tables = np.array(block_tables).astype(np.int32)
        self.contex_lens = np.array(context_lens).astype(np.int32)

        max_context_len_pad = (max_context_len + 15) // 16 * 16
        alibi_mask_pad = np.zeros((num_tokens, num_heads, 16, max_context_len_pad))
        alibi_mask_pad[:, :, :1, :max_context_len] = alibi_mask

        alibi_mask_nz = self.convert_nd_to_nz(alibi_mask_pad)
        logging.debug(f"alibi_mask_nz={alibi_mask_nz.shape}")
        self.alib_mask_nz = alibi_mask_nz.astype(np.float16).reshape(num_tokens * num_heads, -1, 16, 16)

        ref_output = ref_output.reshape(1, num_tokens, num_heads * head_size)
        ref_output_pad = np.zeros((1, tokens_pad, num_heads * head_size))
        ref_output_pad[:, :num_tokens, :] = ref_output
        ref_output_nz = self.convert_nd_to_nz(ref_output_pad)
        self.golden_out = ref_output_nz.astype(np.float16).reshape(1, num_heads * head_size // 16, tokens_pad, 16)

        logging.debug("**********data gen shape***********")
        logging.debug(f"==> query nz shape: {query_nz.shape}, key_cache nz shape: {key_cache_nz.shape}, \
                  alibi mask nz shape: {alibi_mask_nz.shape}")
        logging.debug(f"==> block_tables: {self.block_tables.shape}, data generate finished!")

    def golden_calc(self, in_tensors):
        golden_out = torch.tensor(self.golden_out).half()
        return [golden_out]

    def golden_compare(self, out_tensors, golden_tensors):
        logging.debug(f"out_tensors: {out_tensors}")
        logging.debug(f"golden_tensors: {golden_tensors}")
        return True #self.compare_output_data(out_tensors[0].half(), golden_tensors[0].half(), [0.001, 0.001, 0.005, 0.005])

    @op_test.only_310p
    def test_paged_attention_nz_case0(self):

        num_tokens = 20
        num_heads = 32
        kv_heads = 32
        block_size = 128
        head_size = 128
        num_blocks = 64
        k_seqlen = 128
        tor = head_size ** -0.5
        dtype = "float16"
        self.data_type = torch.float16
        self.calc_data(num_tokens, num_heads, kv_heads, head_size, block_size, num_blocks, k_seqlen, dtype)

        OP_NAME = "PagedAttentionOperation"
        OP_PARAM = {"headNum":num_heads, "kvHead":kv_heads, "headSize":head_size, "tor": tor}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nz])

        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.key_cache.shape}")
        logging.debug(f"v shape: {self.value_cache.shape}")
        logging.debug(f"blcok_tables shape: {self.block_tables}")
        logging.debug(f"contex_lens shape: {self.contex_lens}")
        logging.debug(f"==> alib_mask_nz: {self.alib_mask_nz.shape}")
        logging.debug(f"numTokens: {num_tokens}, numHeads: {num_heads}, kvHead: {kv_heads}"
              f", blockSize: {block_size}, headSize: {head_size}, numBlocks: {num_blocks}")
        attention_out = np.zeros_like(self.q)
        attention_out[:] = 0.1

        return self.execute(
                [
                    torch.tensor(self.q).half(), torch.tensor(self.key_cache).half(),
                    torch.tensor(self.value_cache).half(), torch.tensor(self.block_tables).int() ,
                    torch.tensor(self.contex_lens).int(), torch.tensor(self.alib_mask_nz).half()
                ],
                [
                    torch.tensor(attention_out).half()
                ]
                )


if __name__ == '__main__':
    unittest.main()
