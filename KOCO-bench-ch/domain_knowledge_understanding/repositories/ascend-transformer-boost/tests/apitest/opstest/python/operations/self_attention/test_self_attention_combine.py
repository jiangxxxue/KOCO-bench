#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import time
import json
from enum import Enum
import torch
import logging
import unittest
import math
import numpy as np
import sys
import os
import random
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from self_attention.self_attention_test_data_generator import SelfAttentionTestDataGenerator

import operation_test  # NOQA: E402
torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)
sys.path.append("./tests/pythontest")
save_path = "./"
from golden_compare_cv import compare_cv

class ScaleType(Enum):
    SCALE_TOR = 0
    SCALE_LOGN = 1
    SCALE_LOGN_FP32 = 2
np.random.seed(123)
MASK_TYPE_NO_MASK = 0
MASK_TYPE_NO_HEAD = 1
MASK_TYPE_NO_BATCH = 2
MASK_TYPE_ALIBI_WITH_BATCH = 3
MASK_TYPE_ALIBI_NO_BATCH = 4
MASK_TYPE_NO_HEAD_DECODER = 5
MASK_TYPE_SWA = 6
MASK_TYPE_SWA_DECODER = 7
MASK_TYPE_ALIBI_WITH_PREFIX_BATCH = 8
MASK_TYPE_NO_BATCH_WITH_PREFIX = 9
MASK_TYPE_ALIBI_NO_BATCH_WITH_PREFIX = 10
MASK_TYPE_RAZOR_FUSION = 11
UNPAD_FLASH_ATTENTION_ENCODER_PREFIX_CACHE_ND = 2012
CAL_TYPE_PREFIX_ENCODER = 4
MASK_TYPE_ALIBI_COMPRESS = 4
MASK_TYPE_CAUSAL_MASK = 9
MASK_TYPE_ALIBI_COMPRESS_SQRT = 5
KERNELTYPE_HIGH_PRECISION = 1

def gen_seq_len(batch, max_seq, variate_seq=False):
    if variate_seq:
        num = max_seq // 16
        seqlen_aligned_arange = np.arange(1, num) * 16
        if batch > num:
            seqlen_aligned_remain = np.random.randint(1, max_seq, size=(batch - num))
            seqlen_aligned_remain[:] = ((seqlen_aligned_remain[:] + 15) // 16) * 16
            seqlen_aligned = np.concatenate((seqlen_aligned_arange, seqlen_aligned_remain), 0)
        else:
            seqlen_aligned = seqlen_aligned_arange
        sp_list = np.random.randint(0, 15, size=(num - 1))
        seqlen = seqlen_aligned - sp_list
        seqlen = seqlen[-batch:]
        seqlen_aligned = seqlen_aligned[-batch:]
        print(seqlen)
    else:
        max_seq_aligned = (max_seq + 15) // 16 * 16
        sp_list = np.ones((batch,)) * (max_seq_aligned - max_seq)
        sp_list = sp_list.astype(np.int32)
        seqlen = np.ones((batch,)) * max_seq
        seqlen = seqlen.astype(np.int32)
        print(seqlen)
        seqlen_aligned = np.ones((batch,)) * max_seq_aligned
        seqlen_aligned = seqlen_aligned.astype(np.int32)

    ntokens = seqlen.sum()
    print("ntokens:", ntokens)
    return seqlen, seqlen_aligned, ntokens

def group_matmul(heads, group_num, A, B):
    group_head = heads // group_num
    score = None
    for i in range(group_num):
        group_score = np.matmul(A[i * group_head: (i + 1) * group_head, :, :].astype(np.float32),
                                B[i:(i + 1), :, :].astype(np.float32)).astype(np.float16)
        if score is None:
            score = group_score
        else:
            score = np.concatenate((score, group_score), 0)
    print(score.shape)
    return score

def gen_swa_cmp(window_size, embeddim):
    swa_mask = np.ones(shape=(1, 512, 512)) * -10000.0
    pp_n = 128 if embeddim <= 128 else 64
    # pp_n = 128
    if window_size <= pp_n * 3:
        true_size = window_size
    else:
        if window_size % pp_n == 0:
            true_size = pp_n * 3
        else:
            true_size = pp_n * 2 + window_size % pp_n
    triu_mask = np.triu(swa_mask, 1)
    tril_mask = np.tril(swa_mask, -true_size)
    swa_mask = triu_mask + tril_mask
    swa_mask = swa_mask.reshape(512,512)
    return swa_mask

class TestFlashAttention(operation_test.OperationTest):

    def close_pack(self, in_data, seq_len):
        kv = in_data.numpy()
        dim1len = np.size(kv, -2)
        if max(seq_len) > dim1len:
            return None
        kv = kv.reshape(np.prod(kv.shape[0:-1]), kv.shape[-1])
        c_offset = 0
        s_offset = 0
        for i, len in enumerate(seq_len):
            kv[c_offset:c_offset + seq_len[i]][:] = kv[s_offset:s_offset + seq_len[i]][:]
            c_offset += seq_len[i]
            s_offset += dim1len
        return torch.from_numpy(kv[0:sum(seq_len)][:])

    def calc_expect_func(self, batch, seqlen, heads, embed, window_size, mask_type, group_num=32):
        is_mask = True
        self.is_mask = is_mask
        variate_seq = False
        is_decoder = False
        self.is_decoder = is_decoder
        max_seq = 2048
        self.max_seq = max_seq
        src_type = 'float16'
        fp32 = True
        print(f"group_num: {group_num}")
        print("q_seq is:")
        if is_decoder:
            q_seqlen, q_seqlen_aligned, q_ntokens = gen_seq_len(batch, 1, variate_seq)
            kv_seqlen, kv_seqlen_aligned, kv_ntokens = gen_seq_len(batch, seqlen, variate_seq)
        else:
            q_seqlen, q_seqlen_aligned, q_ntokens = gen_seq_len(batch, seqlen, variate_seq)
            kv_seqlen, kv_seqlen_aligned, kv_ntokens = q_seqlen, q_seqlen_aligned, q_ntokens   # crossattention时，q_seqlen != k_seqlen
        
        self.q_seqlen, self.q_seqlen_aligned, self.q_ntokens, self.kv_seqLen = q_seqlen, q_seqlen_aligned, q_ntokens, kv_seqlen
        print("qseqlen is ", self.q_seqlen)
        self.kv_seqlen_aligned, self.kv_ntokens = kv_seqlen_aligned, kv_ntokens
        max_s = np.max(q_seqlen)
        ntokens2 = (q_seqlen * kv_seqlen).sum()
        embed_v = embed

        q = np.random.uniform(-1.0, 1.0, size=(q_ntokens, heads * embed)).astype(np.float16)
        k = np.random.uniform(-1.0, 1.0, size=(kv_ntokens, group_num * embed)).astype(np.float16)
        v = np.random.uniform(-1.0, 1.0, size=(kv_ntokens, group_num * embed_v)).astype(np.float16)
        self.heads, self.embeddim, self.embeddimv = heads, embed, embed_v
        mask = np.ones(shape=(1, max_s, max_s)).astype(np.float16)  # 使用当前最大seqlen生成mask
        mask_u = np.triu(mask, 1)
        mask_l = np.tril(mask, -window_size)
        mask = mask_u + mask_l
        mask *= -10000.0

        q_offset = 0
        k_offset = 0
        v_offset = 0

        s = None
        _p = None
        out = None

        for idx in range(batch):
            q_s = q_seqlen[idx]
            kv_s = kv_seqlen[idx]
            q_slice = q[q_offset:q_offset + q_s][:]
            q_slice = q_slice.reshape(q_s, heads, embed)
            q_slice = np.transpose(q_slice, (1, 0, 2))  # (heads, q_seq, embed)
            k_slice = k[k_offset:k_offset + kv_s][:]
            k_slice = k_slice.reshape(kv_s, group_num, embed)
            k_slice = np.transpose(k_slice, (1, 0, 2))
            k_slice_t = np.transpose(k_slice, (0, 2, 1))   # get K^T (kv_heads, embed, k_seq)
            v_slice = v[v_offset:v_offset + kv_s][:]
            v_slice = v_slice.reshape(kv_s, group_num, embed_v)
            v_slice = np.transpose(v_slice, (1, 0, 2))
            score = group_matmul(heads, group_num, q_slice, k_slice_t)
            if s is None:
                s = score.reshape([-1, ])
            else:
                s = np.concatenate((s, score.reshape([-1, ])), 0)

            tor = np.float16(1.0 / math.sqrt(1.0 * embed))
            score = score * tor
            if is_mask:
                score = score + mask[:, :q_s, :kv_s]
            score_max = np.max(score, axis=-1)
            score = score - score_max.reshape((heads, q_s, 1))
            score_exp = np.exp(score.astype(np.float32))
            if not fp32:
                score_sum = np.sum(score_exp.astype(np.float16), axis=-1)
                if _p is None:
                    _p = score_exp.astype(np.float16).reshape([-1, ])
                else:
                    _p = np.concatenate((_p, score_exp.astype(np.float16).reshape([-1, ])), 0)
                p = score_exp.astype(np.float16) / score_sum.reshape((heads, q_s, 1)).astype(np.float16)
                out_sub = group_matmul(heads, group_num, p, v_slice)
            else:
                score_sum = np.sum(score_exp, axis=-1)
                if _p is None:
                    _p = score_exp.astype(np.float16).reshape([-1, ])
                else:
                    _p = np.concatenate((_p, score_exp.astype(np.float16).reshape([-1, ])), 0)
                p = score_exp.astype(np.float16)
                out_sub = group_matmul(heads, group_num, p, v_slice)
                out_sub = out_sub / score_sum.reshape((heads, q_s, 1)).astype(np.float16)

            out_sub = out_sub.reshape(heads, q_s, embed_v)
            out_sub = np.transpose(out_sub, (1, 0, 2))
            out_sub = np.ascontiguousarray(out_sub)
            if out is None:
                out = out_sub
            else:
                out = np.concatenate((out, out_sub), 0)

            q_offset += q_s
            k_offset += kv_s
            v_offset += kv_s

        print("==> data generate finished!")

        q = q.astype(src_type).reshape(-1, heads, embed)
        k = k.astype(src_type).reshape(-1, group_num, embed)
        v = v.astype(src_type).reshape(-1, group_num, embed_v)
        # mask = mask.astype(src_type).reshape(max_s, max_s)
        mask = gen_swa_cmp(window_size, embed).astype(src_type)
        q_len = q_seqlen.astype(np.int32)
        out = out.astype(src_type).reshape(-1, heads, embed_v)
        ret_data = q, k, v, mask, q_len, out
        return ret_data

    def set_data_params(self, dynamic_batch=False, batch_state=None, window_size=0, cache_type=0,
                       is_mask=True, is_decoder=False, is_alibi=False, is_razor_fusion = False, alibi_dim=4,
                       batch = 1, kv_head = 1, heads = 1, embeddim = 128, embeddimv = 0, max_seq = 2048,
                       kv_seqLen = [], is_clamp = 0, clamp_min = 0, preTokens = 0, nextTokens = 0,
                       tileQ = 0, tileKv = 0, razorLen = 0, baseM = 0, textQLen = 0, textKvLen = 0,
                       is_splitm = False,
                       clamp_max = 0, data_type = torch.float16, op_type = 0, mask_type = 0,
                       no_cache = False, long_seq = False, is_triu_mask = False, is_multi_layer = False,
                       is_sqrt = False, left_align = False, scaleType = ScaleType.SCALE_TOR.value, fav3 = False,
                       tor = 1, bnsd = False, is_compress = False, q_seqlens=None, num_blocks=None,
                       block_size=None):
        self.dynamic_batch = dynamic_batch
        self.batch_state = batch_state
        self.is_mask = is_mask
        self.is_decoder = is_decoder
        self.is_alibi = is_alibi
        self.preTokens = preTokens
        self.nextTokens = nextTokens
        self.tileQ = tileQ
        self.tileKv = tileKv
        self.razorLen = razorLen
        self.baseM = baseM
        self.textQLen = textQLen
        self.textKvLen = textKvLen
        self.is_razor_fusion = is_razor_fusion
        self.alibi_dim = alibi_dim
        self.batch = batch
        self.kv_head = kv_head
        self.heads = heads
        self.embeddim = embeddim
        self.embeddimv = embeddimv
        self.max_seq = max_seq
        self.kv_seqLen = kv_seqLen
        self.dynamic_batch = dynamic_batch
        self.is_clamp = is_clamp
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.data_type = data_type
        self.no_cache = no_cache
        self.long_seq = long_seq
        self.mask_type = mask_type
        self.is_triu_mask = is_triu_mask
        self.is_multi_layer = is_multi_layer
        self.is_sqrt = is_sqrt
        self.left_align = left_align
        self.fav3 = fav3
        self.scaleType = scaleType
        self.tor = tor
        self.is_int8_flag = False
        self.online = False
        self.bnsd = bnsd
        self.window_size = window_size
        self.is_compress = is_compress
        self.cache_type = cache_type
        self.q_seqlens = q_seqlens if q_seqlens is not None else kv_seqLen

        if self.embeddimv == 0:
            self.embeddimv = self.embeddim
        if is_decoder:
            self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, [1] * batch)
        else:
            self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, self.q_seqlens)
        self.kv_seqlen, self.kv_ntokens = self.gen_seq_len(batch, kv_seqLen)
        # gen intensor for fa kernel
        if is_multi_layer:
            self.layer_id = torch.from_numpy(np.array([1], dtype=np.int32)).to(torch.int32)
        else:
            self.layer_id = torch.from_numpy(np.array([0], dtype=np.int32)).to(torch.int32)
        print("here is ", self.q_seqlen)
        self.q_max_seq = np.max(self.q_seqlen)
        self.kv_max_seq = np.max(self.kv_seqlen)
        q = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(self.q_ntokens, heads * self.embeddim)))

        self.q = q.to(data_type)
        if num_blocks is None:
            self.k = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))).to(data_type)
            self.v = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddimv))).to(data_type)
            if is_splitm:
                maxKvSeqlen = max(self.kv_seqlen)
                self.k = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(self.layer_id[0] + 1, batch, maxKvSeqlen, kv_head * self.embeddim))).to(data_type)
                self.v = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(self.layer_id[0] + 1, batch, maxKvSeqlen, kv_head * self.embeddimv))).to(data_type)
        else:
            # kv cache shape: (num_blocks, block_size, num_heads, head_size)
            self.k_cache = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(num_blocks, block_size, kv_head, embeddim))).to(data_type)
            self.v_cache = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(num_blocks, block_size, kv_head, embeddim))).to(data_type)

            batch = len(kv_seqLen)
            max_context_len = max(kv_seqLen)
            max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
            block_tables = []   # (batch, max_num_blocks_per_seq)
            offset = 0
            for i in range(batch):
                num_blocks_cur_seq = (kv_seqLen[i] + block_size - 1) // block_size
                # padding block table with 0
                block_table = [
                    random.randint(0, num_blocks-1) if j < num_blocks_cur_seq else 0 for j in range(max_num_blocks_per_seq)
                ]
                offset += num_blocks_cur_seq
                block_tables.append(block_table)
            self.block_tables = torch.from_numpy(np.array(block_tables)).to(torch.int32)
            self.k = torch.stack([self.k_cache[self.block_tables[torch.tensor(i, dtype=torch.long)].to(torch.long)].reshape(-1, kv_head * self.embeddim)[:max_context_len, :] for i in range(batch)])
            self.v = torch.stack([self.v_cache[self.block_tables[torch.tensor(i, dtype=torch.long)].to(torch.long)].reshape(-1, kv_head * self.embeddim)[:max_context_len, :] for i in range(batch)])
            self.k = self.k.reshape(1, batch, max_context_len, kv_head * self.embeddim)
            self.v = self.v.reshape(1, batch, max_context_len, kv_head * self.embeddim)

        if self.fav3:
            self.is_int8_flag = True
            self.q_scale, self.q_offset, _ = self.quant_per_head(self.q, heads, embeddim, (self.q_ntokens, heads * self.embeddim))
            self.k_scale, self.k_offset, _ = self.quant_per_head(self.k, kv_head, embeddim, (self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))
            self.v_scale, self.v_offset, _ = self.quant_per_head(self.v, kv_head, embeddim, (self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))
            self.k_scale = (self.k_scale.view(kv_head, 1) * torch.ones([kv_head, heads // kv_head])).view(-1)
            self.k_offset= (self.k_offset.view(kv_head, 1) * torch.ones([kv_head, heads // kv_head])).view(-1)
            self.v_scale = (self.v_scale.view(kv_head, 1) * torch.ones([kv_head, heads // kv_head])).view(-1)
            self.v_offset= (self.v_offset.view(kv_head, 1) * torch.ones([kv_head, heads // kv_head])).view(-1)
            self.offline_scale = torch.from_numpy(np.random.uniform(1 / 127, 3 / 127, size=(heads))).to(torch.float32)

            self.q_int8 = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(self.q_ntokens, heads * self.embeddim))).to(torch.int8)
            self.k_int8 = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))).to(torch.int8)
            self.v_int8 = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddimv))).to(torch.int8)
        
        self.gen_mask(batch, heads, data_type, mask_type, window_size, is_compress, cache_type)
        print("**********data gen shape***********")
        print(f"q shape: {self.q.shape}")
        print(f"k shape: {self.k.shape}")
        print(f"v shape: {self.v.shape}")
        print(f"layer_id shape: {self.layer_id.shape}")
        print(f"mask shape: {self.mask.shape}")

    def quant_per_head(self, data, heads, embeddim, shape):
        temp = data.view(-1, heads, self.embeddim).to(torch.float32)
        scale = torch.stack([self.fav3_quant(temp[:, i, :], data_min = -1, data_max = 1, symmetric = True)[0] for i in range(heads)])
        offset = torch.stack([self.fav3_quant(temp[:, i, :], data_min = -1, data_max = 1, symmetric = True)[1] for i in range(heads)])
        int8_data = torch.zeros_like(temp)
        for i in range(heads):
            int8_data[:, i, :] = ((temp[:, i, :] / scale[i]).round_() + offset[i])
        int8_data = int8_data.view(shape).to(torch.int8)
        return scale, offset, int8_data

    def fav3_quant(self, data, data_min = 0, data_max = 0, symmetric = False, bit = 8):
        n = 2 ** (bit - 1)
        if symmetric:
            quant_min, quant_max = -(n - 1), (n - 1)
        else:
            quant_min, quant_max = -n, (n - 1)
        span = quant_max - quant_min
        if data_min == data_max:
            data_max = data.max().item()
            data_min = data.min().item()
        if symmetric:
            scale = max(data_max, -data_min) / (float(span) / 2)
            offset = 0
        else:
            scale = (data_max - data_min) / float(span)
            offset = (data_min * quant_min + data_max * quant_max) / (data_min - data_max)
        # 量化公式：x / scale + offset
        return torch.tensor(float(scale), dtype = torch.float), torch.tensor(int(offset), dtype = torch.float)

    def get_alibi_slopes(self, n_heads):
        n = 2 ** math.floor(math.log2(n_heads))
        m0 = 2.0 ** (-8.0 / n)
        slopes = torch.pow(m0, torch.arange(1, n + 1))
        if n < n_heads:
            m1 = 2.0 ** ( -4.0 / n)
            mm = torch.pow(m1, torch.arange(1, 1 + 2 * (n_heads - n), 2))
            slopes = torch.cat([slopes, mm])
        return slopes

    def get_alibi_bias(self, n_heads, max_seqlen):
        if not self.left_align:
            self.bias = torch.arange(max_seqlen)
            self.bias = self.bias[None, :] - self.bias[:, None]
            if (self.is_sqrt):
                self.bias = torch.sqrt(torch.abs(self.bias)) * torch.sign(self.bias)
            bias = torch.empty(
                n_heads,
                max_seqlen,
                max_seqlen
            )[:, :max_seqlen, :max_seqlen].copy_(self.bias)
            self.alibi_slopes = self.get_alibi_slopes(n_heads)
        else:
            self.bias = torch.arange(max_seqlen, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(n_heads, max_seqlen, -1)
            self.alibi_slopes = torch.Tensor(self.get_interleave(n_heads))
            bias = self.bias
        bias = bias * self.alibi_slopes[:, None, None]
        return bias

    def get_interleave(self, n, alibi_bias_max=8.0):
        def get_interleave_power_of_2(n, alibi_bias_max):
            if n == 0:
                return 0
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]
        if math.log2(n).is_integer():
            return get_interleave_power_of_2(n, alibi_bias_max)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_interleave_power_of_2(closest_power_of_2, alibi_bias_max) + \
                self.get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

    def gen_swa_cmp(self, max_seq, window_size):
        print("self.pre_mask_coff", self.pre_mask_coff)
        swa_mask = np.ones(shape=(1, 512, 512)) * self.pre_mask_coff
        print("gen_swa_cmp ", swa_mask.shape)
        pp_n = 128 if self.embeddim <= 128 else 64
        pp_n = 128 if self.embeddim != self.embeddimv else pp_n
        if window_size <= pp_n * 3:
            true_size = window_size
        elif window_size % pp_n == 0:
            true_size = pp_n * 3
        else:
            true_size = pp_n * 2 + window_size % pp_n
        triu_mask = np.triu(swa_mask, 1)
        tril_mask = np.tril(swa_mask, -true_size)
        print("gen_swa_cmp ", swa_mask.shape)
        print("gen_swa_cmp ", tril_mask.shape)
        swa_mask = triu_mask + tril_mask
        swa_mask = torch.from_numpy(swa_mask).to(torch.float32)
        print("gen_swa_cmp ", swa_mask.shape)
        return swa_mask

    def gen_razor_fusion_mask(self, razorLen, tileQ, tileKv, textQLen, textKvLen, preTokens, nextTokens, baseM):
        np.set_printoptions(threshold=np.inf)
        
        mask_sizeQ = razorLen * tileQ + textQLen
        mask_sizeK = razorLen * tileKv + textKvLen
        logging.info("generate razor mask:", razorLen, tileQ, tileKv, textQLen, textKvLen, preTokens, nextTokens, baseM)
        mask = np.zeros((mask_sizeQ, mask_sizeK), dtype=int)
        preTokensBlock = preTokens // baseM
        nextTokensBlock = nextTokens // baseM
        idx = razorLen // baseM * baseM
        mask[:, int(idx) : int(razorLen)] = 0
        mask[int(idx) : int(razorLen), :] = 0
        for i in range((razorLen + baseM - 1) // baseM):
            start =  i - preTokensBlock + 1 if i >= preTokensBlock else 0
            end =  i + nextTokensBlock if i < preTokensBlock else start + preTokensBlock + nextTokensBlock - 1
            end = (razorLen + baseM - 1) // baseM if end > (razorLen + baseM - 1) // baseM else end
            for j in range(start, end):
                mask[i * baseM : (i + 1) * baseM, j * baseM : (j + 1) * baseM] = 1
        mask[razorLen :, :] = 0
        mask[:, razorLen :] = 0
        for i in range(tileQ):
            for j in range(tileKv):
                mask[i * razorLen : (i + 1) * razorLen, j * razorLen : (j + 1) * razorLen] = mask[0 : razorLen, 0 : razorLen]

        mask[razorLen * tileQ : , :] = 1
        mask[: , razorLen * tileKv :] = 1
        mask = mask[None, None, :]
        mask = 1 - mask
        return mask * -10000

    def gen_swa_mask(self, max_seq, window_size, pre_mask_coff, cache_type=0):
        swa_mask = np.ones(shape=self.mask_info[0]) * pre_mask_coff
        logging.info(f"gen_swa_mask: window_size {window_size} max_seq {max_seq} self.kv_seqLen {self.kv_seqLen}")
        if window_size < max_seq and self.is_decoder:
            if cache_type == 1:
                for idx, kvseqlen in enumerate(self.kv_seqLen):
                    swa_mask[idx, :, :window_size] = 0
            else:
                for idx, kvseqlen in enumerate(self.kv_seqLen):
                    swa_mask[idx, :, kvseqlen - window_size: kvseqlen] = 0
        elif window_size < max_seq or self.is_compress:
            triu_mask = np.triu(swa_mask, 1)
            tril_mask = np.tril(swa_mask, -window_size)
            swa_mask = triu_mask + tril_mask
        else:
            swa_mask = np.triu(swa_mask, 1)
        return swa_mask

    def gen_mask(self, batch, heads, data_type, mask_type, window_size, is_compress, cache_type=0):
        import random
        q_max_seq = self.max_seq
        kv_max_seq = self.max_seq
        mask_type_dict = {
            # 四维的alibi mask
            MASK_TYPE_ALIBI_WITH_BATCH : ((batch, heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :, :q_s, :kv_s]))),
            MASK_TYPE_ALIBI_WITH_PREFIX_BATCH : ((batch, heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :, kv_s-q_s:kv_s, :kv_s]))),
            # 三维的alibi mask
            MASK_TYPE_ALIBI_NO_BATCH : ((heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
            MASK_TYPE_ALIBI_NO_BATCH_WITH_PREFIX : ((heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, kv_s-q_s:kv_s, :kv_s]))),
            MASK_TYPE_NO_HEAD : ((batch, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            MASK_TYPE_NO_HEAD_DECODER : ((batch, 1, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            MASK_TYPE_NO_BATCH : ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
            MASK_TYPE_NO_BATCH_WITH_PREFIX : ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, kv_s-q_s:kv_s, :kv_s]))),
            MASK_TYPE_SWA : ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
            MASK_TYPE_SWA_DECODER : ((batch, 1, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            # 不加mask
            MASK_TYPE_RAZOR_FUSION : ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:q_s, :kv_s]))),
            MASK_TYPE_NO_MASK : ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: 0))
        }
        # kernel中mask的系数
        if data_type == torch.float16:
            post_mask_coff = 1
            pre_mask_coff = -10000.0
        elif data_type == torch.bfloat16 and self.is_alibi:
            post_mask_coff = 1
            pre_mask_coff = -float("inf")
        elif data_type == torch.float32 and self.is_alibi:
            post_mask_coff = 1
            pre_mask_coff = 1
        else:
            post_mask_coff = -3e38
            pre_mask_coff = 1
        if data_type == torch.float16:
            if self.window_size > 0:
                select_zero = False
            elif self.is_alibi or self.long_seq:
                select_zero = False
            else:
                select_zero = True
        elif data_type == torch.bfloat16:
            if self.window_size > 0:
                select_zero = False
            elif self.is_alibi:
                select_zero = False
            elif self.dynamic_batch or self.is_decoder:
                select_zero = True
            else:
                select_zero = False
        else:
            if self.is_alibi or self.is_decoder:
                select_zero = True
            else:
                select_zero = False
        if self.is_triu_mask:
            select_zero = False

        self.mask_info = mask_type_dict[mask_type]
        mask = np.ones(shape=self.mask_info[0]) * pre_mask_coff
        mask = np.triu(mask, 1)
        zero_indice = random.choices(range(self.max_seq), k = 300)
        if self.window_size > 0:
            mask = self.gen_swa_mask(self.max_seq, window_size, pre_mask_coff, cache_type)
        if self.is_alibi:
            self.alibi_bias = self.get_alibi_bias(heads, self.max_seq)
            mask += self.alibi_bias.numpy()
        if select_zero:
            mask.flat[zero_indice] = 0
        if self.is_razor_fusion:
            mask = self.gen_razor_fusion_mask(self.razorLen, self.tileQ, self.tileKv, self.textQLen, self.textKvLen,
                                                self.preTokens, self.nextTokens, self.baseM)
            post_mask_coff = 1
        self.mask = torch.from_numpy(mask).to(torch.float32)
        self.post_mask_coff = post_mask_coff
        self.pre_mask_coff = pre_mask_coff

    def quantize_tensor_symmetric(self, x, prev_max_abs_vals=None, num_bits=8):
        if x.dtype != torch.float:
            x = x.to(torch.float)

        quant_min = -2 ** (num_bits - 1)
        quant_max = 2 ** (num_bits - 1) - 1

        current_max_abs_vals = x.abs().max(dim=1).values
        if prev_max_abs_vals is not None:
            max_abs_vals = torch.max(prev_max_abs_vals, current_max_abs_vals)
        else:
            max_abs_vals = current_max_abs_vals
        scales = max_abs_vals / (quant_max)
        x_q = torch.clamp(torch.round(x / scales.unsqueeze(1)), quant_min, quant_max)
        x_q = torch.round(x_q)
        x_q = x_q.to(torch.int8)
        return x_q, scales, max_abs_vals

    def dequantize_tensor(self, x_q, scales, value):
        x_deq = x_q.to(torch.float32)
        scales = scales.unsqueeze(1)
        x_deq = x_deq * value
        x_deq = x_deq * scales
        return x_deq

    def online_softmax(self, s_qk, q_s, v_slice, heads, kv_head, embed, online, dtype):
        ans = None
        group_num = heads // kv_head
        for head_idx in range(heads):
            s_head_idx = s_qk[head_idx]
            O = torch.zeros((q_s, embed)).to(dtype)
            Br = q_s
            Bc = 128
            self.row_block_size = Br
            self.col_block_size = Bc
            d = embed
            V_mat = v_slice[head_idx // group_num]
            Tr = q_s // Br
            Tc = q_s // Bc

            d = embed
            Tr = q_s // Br
            Tc = q_s // Bc

            start_row_idx = 0
            start_col_idx = 0

            for i in range(Tr):

                Oi = torch.zeros((Br, d)).to(dtype)  # shape Br x d
                li = torch.zeros((Br, 1)).to(dtype)  # shape Br x 1
                mi = torch.full((Br, 1), -torch.inf).to(dtype)  # shape Br x 1
                pp_max_num = None

                for j in range(Tc):

                    Sij = s_head_idx[i *  Br : (i + 1) * Br, start_col_idx + j * Bc : start_col_idx + (j + 1) * Bc].to(dtype)

                    Vj = V_mat[start_col_idx + j * Bc : start_col_idx + (j + 1) * Bc, :]

                    mi_new = torch.max(
                        torch.column_stack([mi, torch.max(Sij, dim=1).values[:, None]]), dim=1
                    ).values[:, None].to(dtype)
                    Pij_hat = torch.exp((Sij - mi_new).to(torch.float32))
                    Pij_hat = Pij_hat.to(dtype)
                    li = torch.exp((mi - mi_new).to(torch.float32)).to(dtype) * li + torch.sum(Pij_hat, dim=1)[:, None]
                    if self.is_int8_flag:
                        if online:
                            x_q, scales, pp_max_num = self.quantize_tensor_symmetric(Pij_hat, pp_max_num)
                            if pp_max_num == None:
                                pp_max_num = pp_max_num
                            pv = x_q.to(torch.int32) @ Vj.to(torch.int32)
                            Oi = Oi * torch.exp((mi - mi_new).to(torch.float32)).to(dtype) + self.dequantize_tensor(pv, scales, self.v_scale[head_idx]).to(dtype)
                        else:
                            x_q = Pij_hat / self.offline_scale[head_idx]
                            x_q = torch.round(x_q.to(torch.float32))
                            pv = x_q.to(torch.int32) @ Vj.to(torch.int32)
                            pv = pv.to(torch.float32)
                            value = self.v_scale[head_idx] * self.offline_scale[head_idx]
                            Oi = Oi * torch.exp((mi - mi_new).to(torch.float32)).to(dtype) + (pv * value).to(dtype)
                    else:
                        Oi = Oi * torch.exp((mi - mi_new).to(torch.float32)).to(dtype) + Pij_hat @ Vj.to(dtype)

                    mi = mi_new

                if (q_s % Bc != 0):
                    Bc = q_s % Bc
                    start_row_idx = (q_s // self.row_block_size) * self.row_block_size
                    start_col_idx = (q_s // self.col_block_size) * self.col_block_size

                    Sij = s_head_idx[i *  Br : (i + 1) * Br, start_col_idx : start_col_idx + Bc].to(dtype)
                    Vj = V_mat[start_col_idx : start_col_idx + Bc, :]
                    mi_new = torch.max(
                        torch.column_stack([mi, torch.max(Sij, dim=1).values[:, None]]), dim=1
                    ).values[:, None].to(dtype)
                    Pij_hat = torch.exp((Sij - mi_new).to(torch.float32))
                    Pij_hat = Pij_hat.to(dtype)
                    li = torch.exp((mi - mi_new).to(torch.float32)).to(dtype) * li + torch.sum(Pij_hat, dim=1)[:, None]
                    if self.is_int8_flag:
                        if online:
                            x_q, scales, pp_max_num = self.quantize_tensor_symmetric(Pij_hat, pp_max_num)
                            if pp_max_num == None:
                                pp_max_num = pp_max_num
                            pv = x_q.to(torch.int32) @ Vj.to(torch.int32)
                            Oi = Oi * torch.exp((mi - mi_new).to(torch.float32)).to(dtype) + self.dequantize_tensor(pv, scales, self.v_scale[head_idx]).to(dtype)
                        else:
                            x_q = Pij_hat / self.offline_scale[head_idx]
                            x_q = torch.round(x_q.to(torch.float32))
                            pv = x_q.to(torch.int32) @ Vj.to(torch.int32)
                            pv = pv.to(torch.float32)
                            value = self.v_scale[head_idx] * self.offline_scale[head_idx]
                            Oi = Oi * torch.exp((mi - mi_new).to(torch.float32)).to(dtype) + (pv * value).to(dtype)
                    else:
                        Oi = Oi * torch.exp((mi - mi_new).to(torch.float32)).to(dtype) + Pij_hat @ Vj.to(dtype)
                Oi = Oi / li

                O[i * Br : (i + 1) * Br, :] = Oi

            if ans is None:
                ans = O
            else:
                ans = torch.cat((ans, O), 1)
        return ans

    def gen_out_tensor(self, online=False):
        q_offset = 0
        k_offset = 0
        v_offset = 0
        batch = self.batch
        dynamic_batch = self.dynamic_batch
        batch_state = self.batch_state
        heads = self.heads
        is_decoder = self.is_decoder
        embed = self.embeddim
        embedv = self.embeddimv
        max_seq = self.max_seq
        q_seqlen = self.q_seqlen
        kv_seqlen = self.kv_seqLen
        kv_head = self.kv_head
        mask = self.mask
        is_mask = self.is_mask
        is_razor_fusion = self.is_razor_fusion
        q = self.q
        k = self.k
        v = self.v
        if self.fav3:
            q = self.q_int8
            k = self.k_int8
            v = self.v_int8
        q_ntokens = self.q_ntokens
        kv_ntokens = self.kv_ntokens
        layer_id = self.layer_id[0]
        s = None
        _p = None
        out = None
        ans_concat = None
        ans_concat_true = None
        out_true = None

        self.encoder_logN = torch.tensor([2.0] * self.max_seq).to(torch.float32)
        self.encoder_logN.uniform_(1, 2)
        self.decoder_logN = torch.tensor([2.0] * batch).to(torch.float32)
        self.decoder_logN.uniform_(1, 2)
        for idx in range(batch):
            if dynamic_batch and batch_state[idx] == 0 and not is_decoder:
                continue
            if dynamic_batch and batch_state[idx] == 0:
                output = torch.zeros([heads, q_s, embedv])
                output = torch.permute(output, (1, 0, 2))
                if out is None:
                    out = output
                    if not self.fav3:
                        out_true = output
                else:
                    out = torch.cat((out, output), 0)
                    if not self.fav3:
                        out_true = torch.cat((out_true, output), 0)
                q_offset += q_s
                k_offset += max_seq
                v_offset += max_seq
                continue
            q_s = q_seqlen[idx]
            kv_s = kv_seqlen[idx]
            q_slice = q[q_offset:q_offset + q_s][:]
            q_slice = q_slice.view(q_s, heads, embed)
            q_slice = torch.permute(q_slice, (1, 0, 2))
            k_slice = k[layer_id][idx][:kv_s][:]
            k_slice = k_slice.view(kv_s, kv_head, embed)
            k_slice_t = torch.permute(k_slice, (1, 2, 0))   # get K^T
            v_slice = v[layer_id][idx][:kv_s][:]
            v_slice = v_slice.view(kv_s, kv_head, embedv)
            v_slice = torch.permute(v_slice, (1, 0, 2))

            if self.fav3:
                score = self.group_mm_torch(heads, kv_head, q_slice, k_slice_t, torch.int32)
            else:
                score = self.group_mm_torch(heads, kv_head, q_slice, k_slice_t)
            if self.fav3:
                # score:[heads,m,n]
                score = score.to(torch.float32)
                score = score * self.q_scale.view(heads, 1, 1)
                score = score.to(torch.float16)

            if s is None:
                s = score.view([-1, ])
            else:
                s = torch.cat((s, score.view([-1, ])), 0)

            if self.scaleType == ScaleType.SCALE_LOGN_FP32.value:
                if is_decoder:
                    score *= self.decoder_logN[idx]
                else:
                    score *= self.encoder_logN[None, :q_s, None]

            if self.fav3:
                score = score * torch.tensor(self.tor, dtype=torch.float16)
            else:
                score *= self.tor

            if self.is_clamp == 1:
                clamp_min_brc = np.ones((score.shape)) * self.clamp_min
                clamp_max_brc = np.ones((score.shape)) * self.clamp_max
                score = np.float16(np.maximum(score, clamp_min_brc))
                score = torch.from_numpy(np.float16(np.minimum(score, clamp_max_brc)))
            temp_mask = self.mask_info[1](self.mask, idx, q_s, kv_s) * self.post_mask_coff
            if is_mask or is_razor_fusion:
                score = score + temp_mask

            s_qk = score
            s_qk_true = score.to(torch.float32)
            score = score.numpy().astype(np.float32)

            if self.is_int8_flag:
                ans = self.online_softmax(s_qk, q_s, v_slice, heads, kv_head, embed, online, torch.float16)
                if ans_concat is None:
                    ans_concat = ans
                else:
                    ans_concat = torch.cat((ans_concat, ans), 0)

                ans_true = self.online_softmax(s_qk_true, q_s, v_slice, heads, kv_head, embed, online, torch.float32)
                if ans_concat_true is None:
                    ans_concat_true = ans_true
                else:
                    ans_concat_true = torch.cat((ans_concat_true, ans_true), 0)

            score_max = np.max(score, axis=-1)
            score = score - score_max.reshape((heads, q_s, 1))
            score_exp = np.exp(score)
            score_sum = np.sum(score_exp, axis=-1)

            if _p is None:
                _p = score_exp.astype(np.float32).reshape([-1, ])
            else:
                _p = np.concatenate(
                    (_p, score_exp.astype(np.float32).reshape([-1, ])), 0)
            if self.fav3:
                p = score_exp
                p = p * 127
                p = torch.from_numpy(p).to(torch.int8)
            else:
                p_true = (score_exp / score_sum.reshape((heads, q_s, 1)))
                p_true = torch.from_numpy(p_true)
                p = p_true.to(torch.bfloat16)
                o_true = self.group_mm_torch(heads, kv_head, p_true, v_slice)

            o = self.group_mm_torch(heads, kv_head, p, v_slice)
            if self.fav3:
                o = o.to(torch.float)
                v_scale = self.v_scale
                v_scale = v_scale.view(heads, 1, 1)
                o = o * v_scale
                o = o / 127
                o = o / score_sum.reshape((heads, q_s, 1))
            else:
                o_true = o_true.view(heads, q_s, embedv)
                o_true = torch.permute(o_true, (1, 0, 2)).contiguous()
            o = o.view(heads, q_s, embedv)
            o = torch.permute(o, (1, 0, 2)).contiguous()
            if out is None:
                out = o
                if not self.fav3:
                    out_true = o_true
            else:
                out = torch.cat((out, o), 0)
                if not self.fav3:
                    out_true = torch.cat((out_true, o_true), 0)

            q_offset += q_s
            k_offset += max_seq
            v_offset += max_seq
        # golden data
        print("now is: ", q_ntokens, heads, embedv)

        if self.is_int8_flag:
            ans_concat = ans_concat.view(q_ntokens, heads * embedv)
            ans_concat_true = ans_concat_true.view(q_ntokens, heads * embedv)
            self.golden_out = ans_concat
            self.golden_out_true = ans_concat_true
        else:
            out = out.view(q_ntokens, heads * embedv)
            self.golden_out = out.to(self.data_type)
            out_true = out_true.view(q_ntokens, heads * embedv)
            self.golden_out_true = out_true.to(torch.float32)

        if self.no_cache:
            self.k = self.close_pack(self.k.to(torch.float32), kv_seqlen).to(self.data_type)
            self.v = self.close_pack(self.v.to(torch.float32), kv_seqlen).to(self.data_type)
            if self.fav3:
                self.k_int8 = self.close_pack(self.k_int8.to(torch.float32), kv_seqlen).to(torch.int8)
                self.v_int8 = self.close_pack(self.v_int8.to(torch.float32), kv_seqlen).to(torch.int8)
        if self.long_seq:
            self.max_seq = 128
            self.gen_mask(self.batch, self.heads, self.data_type, self.mask_type, 0, False, 0)

    def gen_out_tensor_bnsd(self):
        q_offset = 0
        k_offset = 0
        v_offset = 0
        batch = self.batch
        dynamic_batch = self.dynamic_batch
        batch_state = self.batch_state
        heads = self.heads
        is_decoder = self.is_decoder
        embed = self.embeddim
        embedv = self.embeddimv
        max_seq = self.max_seq
        q_seqlen = self.q_seqlen
        kv_seqlen = self.kv_seqLen
        kv_head = self.kv_head
        mask = self.mask
        is_mask = self.is_mask
        q = self.q
        k = self.k
        v = self.v
        q_ntokens = self.q_ntokens
        kv_ntokens = self.kv_ntokens
        layer_id = self.layer_id[0]
        s = None
        _p = None
        out = None
        obsnd = torch.zeros(batch, max_seq, heads, embedv)
        out_true_bnsd = torch.zeros(batch, max_seq, heads, embedv)
        kbsnd=k.view(layer_id+1,batch,max_seq,kv_head,embed)
        vbsnd=v.view(layer_id+1,batch,max_seq,kv_head,embedv)
        qbsnd = torch.zeros(batch, max_seq, heads, embed)
        self.encoder_logN = torch.tensor([2.0] * self.max_seq).to(torch.float32)
        self.encoder_logN.uniform_(1, 2)
        self.decoder_logN = torch.tensor([2.0] * batch).to(torch.float32)
        self.decoder_logN.uniform_(1, 2)
        for idx in range(batch):
            if dynamic_batch and batch_state[idx] == 0 and not is_decoder:
                continue
            if dynamic_batch and batch_state[idx] == 0:
                output = torch.zeros([heads, q_s, embedv])
                output = torch.permute(output, (1, 0, 2))
                if out is None:
                    out = output
                else:
                    out = torch.cat((out, output), 0)
                q_offset += q_s
                k_offset += max_seq
                v_offset += max_seq
                continue
            # todo bs,n,d 转b，n，s，d
            q_s = q_seqlen[idx]
            kv_s = kv_seqlen[idx]
            q_slice = q[q_offset:q_offset + q_s][:]
            q_slice = q_slice.view(q_s, heads, embed)
            for q_s_idx in range(q_s):
               qbsnd[idx][q_s_idx] = q_slice[q_s_idx][:]
            q_slice = torch.permute(q_slice, (1, 0, 2))
            k_slice = k[layer_id][idx][:kv_s][:]
            k_slice = k_slice.view(kv_s, kv_head, embed)
            k_slice_t = torch.permute(k_slice, (1, 2, 0))   # get K^T
            v_slice = v[layer_id][idx][:kv_s][:]
            v_slice = v_slice.view(kv_s, kv_head, embedv)
            v_slice = torch.permute(v_slice, (1, 0, 2))

            score = self.group_mm_torch(heads, kv_head, q_slice, k_slice_t)
            if s is None:
                s = score.view([-1, ])
            else:
                s = torch.cat((s, score.view([-1, ])), 0)
            score = score * self.tor
            if self.scaleType == ScaleType.SCALE_LOGN_FP32.value:
                if is_decoder:
                    score *= self.decoder_logN[idx]
                else:
                    score *= self.encoder_logN[None, :q_s, None]
            if self.is_clamp == 1:
                clamp_min_brc = np.ones((score.shape)) * self.clamp_min
                clamp_max_brc = np.ones((score.shape)) * self.clamp_max
                score = np.float16(np.maximum(score, clamp_min_brc))
                score = torch.from_numpy(np.float16(np.minimum(score, clamp_max_brc)))
            temp_mask = self.mask_info[1](self.mask, idx, q_s, kv_s) * self.post_mask_coff
            if is_mask:
                score = score + temp_mask
            score = score.numpy().astype(np.float32)
            score_max = np.max(score, axis=-1)
            score = score - score_max.reshape((heads, q_s, 1))
            score_exp = np.exp(score)
            score_sum = np.sum(score_exp, axis=-1)

            if _p is None:
                _p = score_exp.astype(np.float32).reshape([-1, ])
            else:
                _p = np.concatenate(
                    (_p, score_exp.astype(np.float32).reshape([-1, ])), 0)

            p_true = (score_exp / score_sum.reshape((heads, q_s, 1)))
            p_true = torch.from_numpy(p_true)
            o_true = self.group_mm_torch(heads, kv_head, p_true, v_slice)
            o_true = o_true.view(heads, q_s, embedv)
            o_true = torch.permute(o_true, (1, 0, 2)).contiguous()

            #根据数据类型转换
            p = p_true.to(torch.bfloat16)
            o = self.group_mm_torch(heads, kv_head, p, v_slice)
            o = o.view(heads, q_s, embedv)
            o = torch.permute(o, (1, 0, 2)).contiguous()
            if out is None:
                out = o
                out_true = o_true
            else:
                out = torch.cat((out, o), 0)
                out_true = torch.cat((out_true, o_true), 0)

            for i in range(0,q_s):
                obsnd[idx][i] = o[i]
                out_true_bnsd[idx]=out_true[i]
            q_offset += q_s
            k_offset += max_seq
            v_offset += max_seq
        obnsd = torch.permute(obsnd, (0, 2, 1,3))
        out_true_bnsd = torch.permute(out_true_bnsd, (0, 2, 1,3))
        self.qbnsd = torch.permute(qbsnd, (0, 2, 1, 3)).to(self.data_type)
        self.kbnsd = torch.permute(kbsnd, (0, 1, 3, 2, 4)).to(self.data_type)
        self.vbnsd = torch.permute(vbsnd, (0, 1, 3, 2, 4)).to(self.data_type)
        # golden data
        out = out.view(q_ntokens, heads * embedv)
        out_true = out_true.view(q_ntokens, heads * embedv)
        if(self.is_decoder == 1):
            self.golden_out = out
            self.golden_out_true = out_true.to(torch.float32)
        else:
            self.golden_out = obnsd.to(self.data_type)
            self.golden_out_true = out_true_bnsd.to(torch.float32)
        logging.debug(f"golden_out shape: {self.golden_out.shape}")

        if self.no_cache:
            self.k = self.close_pack(self.k.to(torch.float32), kv_seqlen).to(self.data_type)
            self.v = self.close_pack(self.v.to(torch.float32), kv_seqlen).to(self.data_type)
        if self.long_seq:
            self.max_seq = 128
            self.gen_mask(self.batch, self.heads, self.data_type, self.mask_type)

    def gen_out_tensor_bnsd_splitm(self):
        q_offset = 0
        k_offset = 0
        v_offset = 0
        batch = self.batch
        dynamic_batch = self.dynamic_batch
        batch_state = self.batch_state
        heads = self.heads
        is_decoder = self.is_decoder
        embed = self.embeddim
        embedv = self.embeddimv
        max_seq = self.max_seq
        q_seqlen = self.q_seqlen
        kv_seqlen = self.kv_seqLen
        kv_head = self.kv_head
        mask = self.mask
        is_mask = self.is_mask
        q = self.q
        k = self.k
        v = self.v
        q_ntokens = self.q_ntokens
        kv_ntokens = self.kv_ntokens
        layer_id = self.layer_id[0]
        s = None
        _p = None
        out = None
        maxQSeqlen = max(q_seqlen)
        obsnd = torch.zeros(batch, maxQSeqlen, heads, embedv)
        out_true_bnsd = torch.zeros(batch, maxQSeqlen, heads, embedv)
        maxKvSeqlen = max(kv_seqlen)
        kbsnd=k.view(layer_id+1,batch,maxKvSeqlen,kv_head,embed)
        vbsnd=v.view(layer_id+1,batch,maxKvSeqlen,kv_head,embedv)
        qbsnd = torch.zeros(batch, maxQSeqlen, heads, embed)
        for idx in range(batch):
            if dynamic_batch and batch_state[idx] == 0 and not is_decoder:
                continue
            if dynamic_batch and batch_state[idx] == 0:
                output = torch.zeros([heads, q_s, embedv])
                output = torch.permute(output, (1, 0, 2))
                if out is None:
                    out = output
                else:
                    out = torch.cat((out, output), 0)
                q_offset += q_s
                k_offset += max_seq
                v_offset += max_seq
                continue
            # todo bs,n,d 转b，n，s，d
            q_s = q_seqlen[idx]
            kv_s = kv_seqlen[idx]
            q_slice = q[q_offset:q_offset + q_s][:]
            q_slice = q_slice.view(q_s, heads, embed)
            for q_s_idx in range(q_s):
                qbsnd[idx][q_s_idx] = q_slice[q_s_idx][:]
            q_slice = torch.permute(q_slice, (1, 0, 2))
            k_slice = k[layer_id][idx][:kv_s][:]
            k_slice = k_slice.view(kv_s, kv_head, embed)
            k_slice_t = torch.permute(k_slice, (1, 2, 0))   # get K^T
            v_slice = v[layer_id][idx][:kv_s][:]
            v_slice = v_slice.view(kv_s, kv_head, embedv)
            v_slice = torch.permute(v_slice, (1, 0, 2))

            score = self.group_mm_torch(heads, kv_head, q_slice, k_slice_t)
            if s is None:
                s = score.view([-1, ])
            else:
                s = torch.cat((s, score.view([-1, ])), 0)
            score = score * self.tor
            score = score.numpy().astype(np.float32)
            score_max = np.max(score, axis=-1)
            score = score - score_max.reshape((heads, q_s, 1))
            score_exp = np.exp(score)
            score_sum = np.sum(score_exp, axis=-1)

            if _p is None:
                _p = score_exp.astype(np.float32).reshape([-1, ])
            else:
                _p = np.concatenate(
                    (_p, score_exp.astype(np.float32).reshape([-1, ])), 0)

            p_true = (score_exp / score_sum.reshape((heads, q_s, 1)))
            p_true = torch.from_numpy(p_true)
            o_true = self.group_mm_torch(heads, kv_head, p_true, v_slice)
            o_true = o_true.view(heads, q_s, embedv)
            o_true = torch.permute(o_true, (1, 0, 2)).contiguous()

            #根据数据类型转换
            p = p_true.to(torch.bfloat16)
            o = self.group_mm_torch(heads, kv_head, p, v_slice)
            o = o.view(heads, q_s, embedv)
            o = torch.permute(o, (1, 0, 2)).contiguous()

            if out is None:
                out = o
                out_true = o_true
            else:
                out = torch.cat((out, o), 0)
                out_true = torch.cat((out_true, o_true), 0)
            for i in range(0,q_s):
                obsnd[idx][i] = o[i]
                out_true_bnsd[idx][i]=out_true[i]
            q_offset += q_s
            k_offset += kv_s
            v_offset += kv_s
        obnsd = torch.permute(obsnd, (0, 2, 1,3))
        out_true_bnsd = torch.permute(out_true_bnsd, (0, 2, 1,3))
        self.qbnsd = torch.permute(qbsnd, (0, 2, 1, 3)).to(self.data_type)
        self.kbnsd = torch.permute(kbsnd, (0, 1, 3, 2, 4)).to(self.data_type)
        self.vbnsd = torch.permute(vbsnd, (0, 1, 3, 2, 4)).to(self.data_type)
        # golden data
        out = out.view(q_ntokens, heads * embedv)
        out_true = out_true.view(q_ntokens, heads * embedv)
        self.golden_out = obnsd.to(self.data_type)
        self.golden_out_true = out_true_bnsd.to(torch.float32)
        logging.debug(f"golden_out shape: {self.golden_out.shape}")

        if self.no_cache:
            self.k = self.close_pack(self.k.to(torch.float32), kv_seqlen).to(self.data_type)
            self.v = self.close_pack(self.v.to(torch.float32), kv_seqlen).to(self.data_type)

    def gen_seq_len(self, batch, seq_len):
        ntokens = sum(seq_len)
        return seq_len, ntokens

    def compare_output_data(self, out, golden, ratios):
        error_count = 0
        strict_error_count = 0
        fp16_min_normal = 1.0 / (1 << 14)
        golden = golden.flatten().to(torch.float32)
        out = out.flatten().to(torch.float32)
        out_len = out.shape[0]
        diff = torch.abs(golden - out)
        max_diff = diff.max().item()
        limit_error = torch.maximum(torch.abs(golden * ratios[0]), torch.tensor(ratios[1]))
        strict_limit_error = torch.maximum(torch.abs(golden * ratios[2]), torch.tensor(ratios[3]))
        error_count = torch.gt(diff, limit_error).sum().item()
        strict_error_count = torch.gt(diff, strict_limit_error).sum().item()
        logging.info(f"maxDiff {max_diff}")
        logging.info("1/1000 Accuracy is %f",  1 - float(error_count) / out_len)
        logging.info("5/1000 Accuracy is %f",  1 - float(strict_error_count) / out_len)
        if self.data_type == torch.bfloat16:
            logging.debug("accuracy is correct in old standard: %r", (float(strict_error_count) / out_len) <= ratios[2])
        else:
            logging.debug("accuracy is correct in old standard: %r", (float(strict_error_count) / out_len) <= ratios[0])
        calc_times = self.heads * self.max_seq + 4
        if self.data_type == torch.bfloat16:
            if calc_times < 2048:
                error = 2**(-7)
            else :
                error = 2**(-6)
            error_threshold = torch.clamp(torch.abs(golden), min = 1) * error
            res = (diff <= error_threshold).all().item()
            logging.debug("accuracy is correct in new standard: %r", res)
            return res
        elif self.data_type == torch.float16:
            if calc_times < 2048:
                error = 2**(-8)
            else :
                error = 2**(-7)
            error_threshold = torch.clamp(torch.abs(golden), min = 1) * error
            res = (diff <= error_threshold).all().item()
            logging.debug("accuracy is correct in new standard: %r", res)
            return res
        else :
            if calc_times < 2048:
                error = 2**(-11)
            elif calc_times >= 2048 and calc_times < 16384:
                error = 2**(-10)
            else:
                error = 2**(-14)
            error_threshold = torch.clamp(torch.abs(golden), min = 1) * error
            res = (diff <= error_threshold).all().item()
            logging.debug("accuracy is correct in new standard: %r", res)
            return res

    def group_mm_torch(self, heads, group_num, A, B, dtype=torch.float32):
        group_head = heads // group_num
        score = None
        for i in range(group_num):
            group_score = torch.matmul(A[i * group_head: (i + 1) * group_head, :, :].to(dtype), B[i:(i + 1), :, :].to(dtype))
            if score is None:
                score = group_score
            else:
                score = torch.cat((score, group_score), 0)
        return score

    def golden_calc(self, in_tensors):
        golden_out = self.golden_out.clone().detach().requires_grad_(True).half().npu()
        return [golden_out]

    def golden_compare(self, out_tensors, golden_tensors):
        print("max(golden_out): ", torch.max(golden_tensors[0].clone().detach().half().npu()).item(),)
        print("min(golden_out): ", torch.min(golden_tensors[0].clone().detach().half().npu()).item(),)
        print("max(actual out): ", torch.max(out_tensors[0].clone().detach().half().npu()).item(),)
        print("min(actual out): ", torch.min(out_tensors[0].clone().detach().half().npu()).item(),)
        # nan/inf
        result_single = self.compare_output_data(out_tensors[0].clone().detach().half().npu(),
                                                 golden_tensors[0].clone().detach().half().npu(),
                                                 [0.001, 0.001, 0.005, 0.005])
        if self.is_int8_flag:
            result_double = compare_cv(self.golden_out_true.clone().detach().half().npu(),
                                       golden_tensors[0].clone().detach().half().npu(), out_tensors[0])
            return (result_double or result_single)
        else:
            result_double = compare_cv(self.golden_out_true.clone().detach().half().npu(),
                                       golden_tensors[0].clone().detach().half().npu(), out_tensors[0])
            return (result_double or result_single)


    def test_swa_decoder(self):
        """
            is_decoder = 1, no_cache=False, "maskType": MASK_TYPE_SLIDING_WINDOW_NORM
            qselen[i] = 1 for all i (decoder)
            kv_seqLen[i] = 114 for all i
        """
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.data_type = torch.float16
        data_type = self.data_type
        self.batch = 8
        batch = self.batch
        self.kv_head = 32  # kv_head num
        kv_head = self.kv_head
        self.is_decoder = 1  # prefill or decoder
        self.heads = 32  # llama7b  hidden_size 4096
        self.embeddim = 128
        self.embeddim_v = self.embeddim
        tor = 1
        self.dynamic_batch = False
        kv_seqLen = [114] * batch
        qSeqLen = [1] * batch
        self.max_seq = 256
        self.window_size = 16
        self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, [1] * batch)
        self.kv_seqlen, self.kv_ntokens = self.gen_seq_len(batch, kv_seqLen)
        self.layer_id = torch.from_numpy(np.array([0], dtype=np.int32)).to(torch.int32)
        self.q_max_seq = np.max(self.q_seqlen)
        self.kv_max_seq = np.max(self.kv_seqlen)
        tor = np.float32(1.0 / math.sqrt(1.0 * self.embeddim))

        self.q_scale = 1
        self.qk_scale = tor
        self.cache_type = 0
        
        OP_NAME = "SelfAttentionOperation"
        OP_PARAM = {"type": 1}
        self.set_data_params(cache_type=self.cache_type, is_decoder=self.is_decoder, batch=batch, kv_head=kv_head, heads=self.heads,
                             embeddim=self.embeddim, max_seq=self.max_seq, kv_seqLen=kv_seqLen,
                             data_type=data_type, long_seq = True, window_size=self.window_size,
                             op_type=OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD_DECODER,
                             no_cache=False, is_sqrt=False, tor=tor, q_seqlens=self.q_seqlen)
        self.gen_out_tensor()
        self.window_size = 16
        param = json.dumps(
            {"headNum": self.heads, "qScale": float(self.q_scale), "qkScale": float(self.qk_scale), "maskType": 7,
             "kvcacheCfg": 1, "calcType": 2, "windowSize": self.window_size})
        
        self.param_seqlen = self.q_seqlen
        self.param_token_offset = self.kv_seqlen
        run_param = json.dumps({"tokenOffset": self.param_token_offset, "seqLen": self.param_seqlen, "maskType": 7})
        self.execute_with_param(OP_NAME, param, run_param,
                                [self.q.npu(), self.k.npu(), self.v.npu(),
                                 torch.tensor(self.kv_seqlen).to(torch.int32).npu(),
                                 torch.tensor(self.q_seqlen).to(torch.int32).npu(), self.layer_id.npu()])


    def test_swa_encoder_cache(self):
        """
            is_decoder = 0, no_cache=False, "maskType": MASK_TYPE_SLIDING_WINDOW_NORM， cacheType = 1
            qselen = kv_seqLen = [33, 512, ...]
        """
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.data_type = torch.float16
        data_type = self.data_type
        self.batch = 8
        batch = self.batch
        self.kv_head = 33  # kv_head num
        kv_head = self.kv_head
        self.is_decoder = 0  # prefill or decoder
        self.heads = 33  # llama7b  hidden_size 4096
        self.embeddim = 128
        self.embeddim_v = self.embeddim
        self.dynamic_batch = False
        kv_seqLen = [self.heads, 512] * (self.batch // 2)
        self.max_seq = max(kv_seqLen)
        
        self.window_size = 16
        self.cacheType = 1
        self.kv_seqlen, self.kv_ntokens = self.gen_seq_len(batch, kv_seqLen)
        self.q_seqlen, self.q_ntokens = self.kv_seqlen, self.kv_ntokens
        
        self.q_max_seq = np.max(self.q_seqlen)
        self.kv_max_seq = np.max(self.kv_seqlen)
        tor = np.float32(1.0 / math.sqrt(1.0 * self.embeddim))
        
        self.q_scale = 1
        self.qk_scale = tor
        self.cache_type = 1
        
        OP_NAME = "SelfAttentionOperation"
        OP_PARAM = {"type": 1}
        print(f" self.q_ntokens 1  {self.q_ntokens}")
        self.set_data_params(cache_type=self.cache_type, is_decoder=self.is_decoder, batch=batch, kv_head=kv_head,
                             heads=self.heads, embeddim=self.embeddim, max_seq=self.max_seq, kv_seqLen=kv_seqLen,
                             data_type=data_type, long_seq = False, op_type=OP_PARAM["type"], mask_type = MASK_TYPE_SWA,
                             no_cache=False, is_sqrt=False, tor=tor)
        self.gen_out_tensor()
        self.window_size = 16
        self.q_scale = 1
        self.qk_scale = tor
        param = json.dumps(
            {"headNum": self.heads, "qScale": float(self.q_scale), "qkScale": float(self.qk_scale), "maskType": 7,
             "kvcacheCfg": 1, "calcType": 1,
             "windowSize": self.window_size, "cacheType": self.cacheType})
        self.param_seqlen = self.kv_seqlen
        self.param_token_offset = self.kv_seqlen
        run_param = json.dumps({"tokenOffset": self.param_token_offset, "seqLen": self.param_seqlen, "maskType": 7})
        self.execute_with_param(OP_NAME, param, run_param,
                                [self.q.npu(), self.k.npu(), self.v.npu(), 
                                 self.mask.reshape(self.q_max_seq, self.kv_max_seq).to(data_type).npu(),
                                 torch.tensor(self.kv_seqlen).to(torch.int32).npu(),
                                 torch.tensor(self.kv_seqlen).to(torch.int32).npu(), self.layer_id.npu()])

    def test_swa_decoder_cache(self):
        """
            is_decoder = 0, no_cache=False, "maskType": MASK_TYPE_SLIDING_WINDOW_NORM， cacheType = 1
            qselen = kv_seqLen = [33, 512, ...]
        """
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.data_type = torch.bfloat16
        data_type = self.data_type
        self.batch = 8
        batch = self.batch
        self.kv_head = 32  # kv_head num
        self.is_decoder = 1  # prefill or decoder
        self.heads = 32  # llama7b  hidden_size 4096
        self.embeddim = 128
        self.embeddim_v = self.embeddim
        self.dynamic_batch = False
        kv_seqLen = [32, 1024] * 4
        self.max_seq = 1024
        self.window_size = 64
        self.cache_type = 1
        self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, [1] * batch)
        self.kv_seqlen, self.kv_ntokens = self.gen_seq_len(batch, kv_seqLen)
        self.layer_id = torch.from_numpy(np.array([0], dtype=np.int32)).to(torch.int32)
        self.q_max_seq = np.max(self.q_seqlen)
        self.kv_max_seq = np.max(self.kv_seqlen)
        tor = np.float32(1.0 / math.sqrt(1.0 * self.embeddim))

        self.q_scale = 1
        self.qk_scale = tor
        self.cache_type = 1
        
        OP_NAME = "SelfAttentionOperation"
        OP_PARAM = {"type": 1}
        self.set_data_params(cache_type=self.cache_type, is_decoder=self.is_decoder, batch=batch, kv_head=self.kv_head,
                             heads=self.heads, embeddim=self.embeddim, max_seq=self.max_seq, kv_seqLen=kv_seqLen,
                             data_type=data_type, long_seq = False, op_type=OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD_DECODER,
                             no_cache=False, is_sqrt=False, tor=tor, q_seqlens=self.q_seqlen, window_size=self.window_size)
        self.gen_out_tensor()
        self.window_size = 64
        
        param = json.dumps(
            {"headNum": self.heads, "qScale": float(self.q_scale), "qkScale": float(self.qk_scale), "maskType": 7,
             "kvcacheCfg": 1, "calcType": 2, "windowSize": self.window_size, "cacheType": 1})
        self.param_seqlen = self.q_seqlen
        self.param_token_offset = self.kv_seqlen
        run_param = json.dumps({"tokenOffset": self.param_token_offset, "seqLen": self.param_seqlen, "maskType": 7})
        self.execute_with_param(OP_NAME, param, run_param,
                                [self.q.npu(), self.k.npu(), self.v.npu(),
                                 torch.tensor(self.kv_seqlen).to(torch.int32).npu(),
                                 torch.tensor(self.q_seqlen).to(torch.int32).npu(), self.layer_id.npu()])

    def test_swa_encoder(self):
        """
            is_decoder = 0, no_cache=False, "maskType": MASK_TYPE_SLIDING_WINDOW_NORM， cacheType = 0
            qselen = kv_seqLen = [32, 256, ...] + norm swa mask
        """
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.data_type = torch.bfloat16
        data_type = self.data_type
        self.batch = 8
        batch = self.batch
        self.kv_head = 32        # kv_head num
        kv_head = self.kv_head
        self.is_decoder = 0       # prefill or decoder
        self.heads = 32          # llama7b  hidden_size 4096
        self.embeddim = 128
        self.embeddim_v = self.embeddim
        self.dynamic_batch = False
        kv_seqLen = [32, 256] * 4
        self.max_seq = 256
        self.window_size = 16
        self.cacheType = 0
        self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, kv_seqLen)
        self.kv_seqlen, self.kv_ntokens = self.gen_seq_len(batch, kv_seqLen)
        self.layer_id = torch.from_numpy(np.array([0], dtype=np.int32)).to(torch.int32)
        self.q_max_seq = np.max(self.q_seqlen)
        self.kv_max_seq = np.max(self.kv_seqlen)
        tor = np.float32(1.0 / math.sqrt(1.0 * self.embeddim))
    
        self.q_scale = 1
        self.qk_scale = tor
        self.cache_type = 1
        OP_NAME = "SelfAttentionOperation"
        OP_PARAM = {"type": 1}
        self.set_data_params(cache_type=self.cache_type, is_decoder=self.is_decoder, batch=batch, kv_head=self.kv_head,
                             heads=self.heads, embeddim=self.embeddim, max_seq=self.max_seq, kv_seqLen=kv_seqLen,
                             data_type=data_type, long_seq = False, op_type=OP_PARAM["type"], mask_type = MASK_TYPE_SWA,
                             no_cache=False, is_sqrt=False, tor=tor, q_seqlens=self.q_seqlen)
        self.gen_out_tensor()
        self.window_size = 16
        
        param = json.dumps(
            {"headNum": self.heads, "qScale": float(self.q_scale), "qkScale": float(self.qk_scale), "maskType": 7,
             "kvcacheCfg": 1, "calcType": 1, "windowSize": self.window_size, "cacheType": 0})
        self.param_seqlen = self.q_seqlen
        self.param_token_offset = self.kv_seqlen
        run_param = json.dumps({"tokenOffset": self.param_token_offset, "seqLen": self.param_seqlen, "maskType": 7})
        self.execute_with_param(OP_NAME, param, run_param,
                                [self.q.npu(), self.k.npu(), self.v.npu(),
                                 self.mask.reshape(self.q_max_seq, self.kv_max_seq).to(data_type).npu(),
                                 torch.tensor(self.kv_seqlen).to(torch.int32).npu(),
                                 torch.tensor(self.q_seqlen).to(torch.int32).npu(), self.layer_id.npu()])

    
    def test_swa_encoder_compress_mask(self):
        """
            is_decoder = 0, no_cache=False, "maskType": MASK_TYPE_SLIDING_WINDOW_COMPRESS， cacheType = 0
            qselen = kv_seqLen = [32, 256, ...] + compress swa mask
        """
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.data_type = torch.bfloat16
        data_type = self.data_type
        self.batch = 8
        batch = self.batch
        self.kv_head = 32        # kv_head num
        kv_head = self.kv_head
        self.is_decoder = 0       # prefill or decoder
        self.heads = 32          # llama7b  hidden_size 4096
        self.embeddim = 128
        self.embeddim_v = self.embeddim
        kv_seqLen = [32, 256] * 4
        self.max_seq = 256
        self.window_size = 16
        self.cacheType = 0
        self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, kv_seqLen)
        self.kv_seqlen, self.kv_ntokens = self.gen_seq_len(batch, kv_seqLen)
        self.layer_id = torch.from_numpy(np.array([0], dtype=np.int32)).to(torch.int32)
        self.q_max_seq = np.max(self.q_seqlen)
        self.kv_max_seq = np.max(self.kv_seqlen)
        tor = np.float32(1.0 / math.sqrt(1.0 * self.embeddim))
        
        self.q_scale = 1
        self.qk_scale = tor
        self.cache_type = 1
        OP_NAME = "SelfAttentionOperation"
        OP_PARAM = {"type": 1}
        self.set_data_params(cache_type=self.cache_type, is_decoder=self.is_decoder, batch=batch, kv_head=self.kv_head,
                             heads=self.heads, embeddim=self.embeddim, max_seq=self.max_seq, kv_seqLen=kv_seqLen,
                             data_type=data_type, long_seq = False, op_type=OP_PARAM["type"], mask_type = MASK_TYPE_SWA,
                             no_cache=False, is_sqrt=False, tor=tor, q_seqlens=self.q_seqlen)
        self.gen_out_tensor()
        self.window_size = 16
        attention_mask = self.gen_swa_cmp(self.max_seq, self.window_size).to(data_type).npu()

        param = json.dumps(
            {"headNum": self.heads, "qScale": float(self.q_scale), "qkScale": float(self.qk_scale), "maskType": 8,
             "kvcacheCfg": 1, "calcType": 1, "windowSize": self.window_size, "cacheType": 0})
        self.param_seqlen = self.q_seqlen
        self.param_token_offset = self.kv_seqlen
        run_param = json.dumps({"tokenOffset": self.param_token_offset, "seqLen": self.param_seqlen, "maskType": 7})
        self.execute_with_param(OP_NAME, param, run_param,
                                [self.q.npu(), self.k.npu(), self.v.npu(),
                                 attention_mask.reshape(512, 512).to(data_type).npu(),
                                 torch.tensor(self.kv_seqlen).to(torch.int32).npu(),
                                 torch.tensor(self.q_seqlen).to(torch.int32).npu(), self.layer_id.npu()])

    def test_operation_logn(self):
        """
            is_decoder = 1, no_cache=False, "maskType": MASK_TYPE_NORM
            qseqlen = [1] * batch
            kv_seqLen = [32, 1024] * 4
        """
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        mask_type = MASK_TYPE_NO_HEAD_DECODER
        data_type = torch.bfloat16
        batch = 8
        kv_head = 32  # kv_head num
        is_decoder = 1  # prefill or decoder
        heads = 32  # llama7b  hidden_size 4096
        embeddim = 128
        embeddimv = np.random.randint(1, embeddim)
        max_seq = 1024
        tor = 1
        dynamic_batch = False
        kv_seqLen = [32, 1024] * 4
        qSeqLen = [1] * batch
        self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, [1] * batch)
        self.kv_seqlen, self.kv_ntokens = self.gen_seq_len(batch, kv_seqLen)
        self.q_max_seq = np.max(self.q_seqlen)
        self.kv_max_seq = np.max(self.kv_seqlen)
        tor = np.float32(1.0 / math.sqrt(1.0 * embeddim))
        self.set_data_params(mask_type=mask_type, tor=tor, q_seqlens=self.q_seqlen, kv_seqLen=self.kv_seqlen, data_type=data_type, batch=batch, kv_head=kv_head, 
                            is_decoder=is_decoder, heads=heads, embeddim=embeddim, embeddimv=embeddimv, max_seq=max_seq, dynamic_batch=dynamic_batch)
        self.gen_out_tensor()
        OP_NAME = "SelfAttentionOperation"
        self.q_scale = 1
        self.qk_scale = tor
        param = json.dumps(
            {"headNum": self.heads, "qScale": float(self.q_scale), "qkScale": float(self.qk_scale), "maskType": 1,
             "kvcacheCfg": 1, "calcType": 2})
        self.param_seqlen = self.q_seqlen
        self.param_token_offset = self.kv_seqlen
        run_param = json.dumps({"tokenOffset": self.param_token_offset, "seqLen": self.param_seqlen})
        # pdb.set_trace()
        self.execute_with_param(OP_NAME, param, run_param,
                                [self.q.npu(), self.k.npu(), self.v.npu(), self.mask.to(data_type).npu(),
                                 torch.tensor(self.kv_seqlen).to(torch.int32).npu(),
                                 torch.tensor(self.q_seqlen).to(torch.int32).npu(), self.layer_id.npu()])
    
    def test_operation_split_kvcache_success_float16(self):
        """
            is_decoder = 1, no_cache=False, "maskType": MASK_TYPE_NO_HEAD_DECODER, cache_type = 1
            qseqlen = [1, ...] 
            kv_seqLen = [114, ...]
        """
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        mask_type = MASK_TYPE_NO_HEAD_DECODER    
        self.data_type = torch.float16
        data_type = self.data_type
        self.batch = 22
        batch = self.batch
        self.kv_head = 44       # kv_head num
        kv_head = self.kv_head
        self.is_decoder = 1       # prefill or decoder
        self.heads = 44          # llama7b  hidden_size 4096
        self.embeddim = 256
        self.embeddim_v = self.embeddim
        self.max_seq = 256
        tor = 1
        self.dynamic_batch = False
        kv_seqLen = [114] * batch
        qSeqLen = [1] * batch
        self.is_clamp = 0
        self.clamp_min = 0
        self.clamp_max = 0
        self.is_triu_mask = False
        self.long_seq = False
        self.is_alibi = False
        self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, [1] * batch)
        self.kv_seqlen, self.kv_ntokens = self.gen_seq_len(batch, kv_seqLen)
        self.layer_id = torch.from_numpy(np.array([0], dtype=np.int32)).to(torch.int32)
        self.q_max_seq = np.max(self.q_seqlen)
        self.kv_max_seq = np.max(self.kv_seqlen)
        self.cache_type = 1
        self.window_size = 0
        self.is_compress = False
        OP_NAME = "SelfAttentionOperation"
        OP_PARAM = {"type": 1}
        tor = np.float32(1.0 / math.sqrt(1.0 * self.embeddim))
        self.set_data_params(cache_type=self.cache_type, is_decoder=self.is_decoder, batch=batch, kv_head=self.kv_head,
                             heads=self.heads, embeddim=self.embeddim, embeddimv=self.embeddim_v, max_seq=self.max_seq, kv_seqLen=kv_seqLen,
                             data_type=data_type, op_type=OP_PARAM["type"], mask_type = mask_type,
                             no_cache=False, is_sqrt=False, tor=tor, q_seqlens=self.q_seqlen)
        q = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(self.q_ntokens, self.heads * self.embeddim)))
        
        self.q = q.to(data_type)
        self.k_list = []
        self.v_list = []
        for i in range(self.batch):
            self.k_list.append(torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(1, 1, self.max_seq, kv_head * self.embeddim))).to(data_type).npu())
            self.v_list.append(torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(1, 1, self.max_seq, kv_head * self.embeddim_v))).to(data_type).npu())
        
        self.k = torch.cat(self.k_list, 1).cpu()
        self.v = torch.cat(self.v_list, 1).cpu()
 
        for i in range(self.batch):
            self.k_list[i] = self.k_list[i].squeeze().npu()
            self.v_list[i] = self.v_list[i].squeeze().npu()
        self.gen_out_tensor()
    
        self.q_scale = 1
        self.qk_scale = tor
        param = json.dumps({"headNum": self.heads, "qScale": float(self.q_scale), "qkScale": float(self.qk_scale), "maskType": 1, "kvcacheCfg":1, "calcType":2})
        self.param_seqlen = self.q_seqlen
        self.param_token_offset = self.kv_seqlen
        run_param = json.dumps({"tokenOffset": self.param_token_offset, "seqLen": self.param_seqlen, "byPass": "true"})
        #pdb.set_trace()
        self.execute_with_param_and_tensor_list(OP_NAME, param, run_param,
                     [self.q.npu(), self.k.npu(), self.v.npu(),self.mask.to(data_type).npu(),torch.tensor(self.kv_seqlen).to(torch.int32).npu(), torch.tensor(self.q_seqlen).to(torch.int32).npu(), self.layer_id.npu()],
                     [self.k_list, self.v_list], ["kCache", "vCache"])
    
    def test_operation_split_kvcache_success_bfloat16(self):
        """
            is_decoder = 1, no_cache=False, "maskType": MASK_TYPE_NO_HEAD_DECODER, cache_type = 1
            qseqlen = [1, ...] 
            kv_seqLen = [114, ...]
        """
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        mask_type = MASK_TYPE_NO_HEAD_DECODER    
        self.data_type = torch.bfloat16
        data_type = self.data_type
        self.batch = 22
        batch = self.batch
        self.kv_head = 44       # kv_head num
        kv_head = self.kv_head
        self.is_decoder = 1       # prefill or decoder
        self.heads = 44          # llama7b  hidden_size 4096
        self.embeddim = 256
        self.embeddim_v = self.embeddim
        self.max_seq = 256
        tor = 1
        self.dynamic_batch = False
        kv_seqLen = [114] * batch
        qSeqLen = [1] * batch
        self.is_clamp = 0
        self.clamp_min = 0
        self.clamp_max = 0
        self.is_triu_mask = False
        self.long_seq = False
        self.is_alibi = False
        self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, [1] * batch)
        self.kv_seqlen, self.kv_ntokens = self.gen_seq_len(batch, kv_seqLen)
        self.layer_id = torch.from_numpy(np.array([0], dtype=np.int32)).to(torch.int32)
        self.q_max_seq = np.max(self.q_seqlen)
        self.kv_max_seq = np.max(self.kv_seqlen)
        self.cache_type = 1
        self.window_size = 0
        self.is_compress = False
        
        q = torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(self.q_ntokens, self.heads * self.embeddim)))
        tor = np.float32(1.0 / math.sqrt(1.0 * self.embeddim))
        #self.q = (q * tor).to(data_type)
        
        OP_NAME = "SelfAttentionOperation"
        OP_PARAM = {"type": 1}
        self.set_data_params(cache_type=self.cache_type, is_decoder=self.is_decoder, batch=batch, kv_head=self.kv_head,
                             heads=self.heads, embeddim=self.embeddim, embeddimv=self.embeddim_v, max_seq=self.max_seq, kv_seqLen=kv_seqLen,
                             data_type=data_type, long_seq = False, op_type=OP_PARAM["type"], mask_type = mask_type,
                             tor=tor, q_seqlens=self.q_seqlen)
        self.q = q.to(data_type)
        self.k_list = []
        self.v_list = []
        for i in range(self.batch):
            self.k_list.append(torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(1, 1, self.max_seq, self.kv_head * self.embeddim))).to(data_type).npu())
            self.v_list.append(torch.from_numpy(np.random.uniform(-5.0, 5.0, size=(1, 1, self.max_seq, self.kv_head * self.embeddim_v))).to(data_type).npu())
        
        self.k = torch.cat(self.k_list, 1).cpu()
        self.v = torch.cat(self.v_list, 1).cpu()
 
        for i in range(self.batch):
            self.k_list[i] = self.k_list[i].squeeze().npu()
            self.v_list[i] = self.v_list[i].squeeze().npu()
        self.gen_out_tensor()
        
        self.q_scale = 1
        self.qk_scale = tor
        param = json.dumps({"headNum": self.heads, "qScale": float(self.q_scale), "qkScale": float(self.qk_scale), "maskType": 1, "kvcacheCfg":1, "calcType":2})
        self.param_seqlen = self.q_seqlen
        self.param_token_offset = self.kv_seqlen
        run_param = json.dumps({"tokenOffset": self.param_token_offset, "seqLen": self.param_seqlen, "byPass": "true"})
        #pdb.set_trace()
        self.execute_with_param_and_tensor_list(OP_NAME, param, run_param,
                     [self.q.npu(), self.k.npu(), self.v.npu(), self.mask.to(data_type).npu(),torch.tensor(self.kv_seqlen).to(torch.int32).npu(), torch.tensor(self.q_seqlen).to(torch.int32).npu(), self.layer_id.npu()],
                     [self.k_list, self.v_list], ["kCache", "vCache"])
    
    def test_encoder_operation_mask_free_fp16(self):
        """
            is_decoder = 0, no_cache=True, "maskType": MASK_TYPE_NO_BATCH,
            qseqlen = kv_seqLen = [1024]
        """
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 12          
        embeddim = 128
        max_seq = 1024
        tor = 1
        kv_seqLen = [1024]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        data_type = torch.float16


        self.set_data_params(dynamic_batch = dynamic_batch, 
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads, 
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = 2001, mask_type = MASK_TYPE_ALIBI_WITH_BATCH, no_cache = True)
        print("embeddimv: ", self.embeddimv)
        self.gen_out_tensor()
        param_seqlen = self.kv_seqLen
        self.alibi_slopes *= -1
        mask = np.ones((256,256)) * 60000
        mask = np.triu(mask, 1)
        self.mask = self.bias[:256, :256] * -1 + mask
        self.mask = self.mask.to(torch.float16)
        print(f"===============self.mask {self.mask.shape}")
        print(f"===============self.mask {torch.max(self.mask)} {torch.min(self.mask)}")
        print(self.alibi_slopes)
        OP_NAME = "SelfAttentionOperation"
        PARAM = json.dumps({"headNum": 12, "qkScale": 1, "kvHeadNum": 1,
                                "calcType": 3, "maskType": 4, "isTriuMask": 1, "kernelType": 1})
        RUN_PARAM = json.dumps({"seqLen": param_seqlen})
        print(self.q.npu().contiguous().shape, self.k.npu().contiguous().shape, self.v.npu().contiguous().shape, self.mask.npu().contiguous().shape, torch.from_numpy(np.array(self.kv_seqLen).astype(np.int32)).npu().contiguous().shape, self.alibi_slopes.npu().contiguous().shape, param_seqlen)
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [
            self.q.npu().contiguous(), self.k.npu().contiguous(), self.v.npu().contiguous(), self.mask.npu().contiguous(), torch.from_numpy(np.array(self.kv_seqLen).astype(np.int32)).npu().contiguous(), self.alibi_slopes.npu().contiguous()
        ])

    def test_flash_attention_case_fa_encoder_withcache_bf16_maskfree(self):
        """
            is_decoder = 0, no_cache=True, "maskType": MASK_TYPE_CAUSAL_MASK,
            qseqlen = [seqlen] * batch
            kv_seqLen = [seqlen + 128 * random.randint(1, 4)] * batch
        """
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        # [b,ms,ms]
        batch = random.randint(1, 16)
        kv_head = random.randint(1, 5)  # kv_head num
        isdecoder = 0  # prefill or decoder
        heads = kv_head * random.randint(1, 4)  # head num
        embeddim = 128
        max_seq = 128 * 100
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        seqlen = random.randint(1, 4096)
        q_seqlens = [seqlen] * batch
        kv_seqLen = [seqlen + 128 * random.randint(1, 4)] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "SelfAttentionOperation"
        OP_PARAM = {"type": UNPAD_FLASH_ATTENTION_ENCODER_PREFIX_CACHE_ND, "qSeqLen": q_seqlens,
                    "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp": is_clamp, "clampMin": clamp_min, "clampMax": clamp_max,
                    "maskType": MASK_TYPE_CAUSAL_MASK, "kvHead": kv_head,
                    "isTriuMask": 1, "alibiLeftAlign": 0, "isAlibiMaskSqrt": 0}
        data_type = random.choice([torch.bfloat16, torch.float16])
        print(
            f"---batch:{batch}---kv_head:{kv_head}---q_seqlens:{q_seqlens}---kv_seqLen:{kv_seqLen}---kv_head:{kv_head}---heads:{heads}---data_type:{data_type}---")
        self.set_data_params(dynamic_batch=dynamic_batch,
                             is_decoder=isdecoder, batch=batch, kv_head=kv_head, heads=heads,
                             embeddim=embeddim, max_seq=max_seq, kv_seqLen=kv_seqLen,
                             is_clamp=is_clamp, clamp_max=clamp_max, clamp_min=clamp_min,
                             data_type=data_type,
                             op_type=OP_PARAM["type"], mask_type=MASK_TYPE_ALIBI_NO_BATCH_WITH_PREFIX,
                             no_cache=True, tor=tor, q_seqlens=q_seqlens,
                             num_blocks=num_blocks, block_size=block_size, is_triu_mask=True, is_mask=True)
        self.gen_out_tensor()
        PARAM = json.dumps(
            {"headNum": heads, "calcType": CAL_TYPE_PREFIX_ENCODER, "maskType": MASK_TYPE_CAUSAL_MASK,
             "kvHeadNum": kv_head, "isTriuMask": 1, "qkScale": tor, "kernelType": KERNELTYPE_HIGH_PRECISION})
        RUN_PARAM = json.dumps({"seqLen": q_seqlens, "kvSeqLen": kv_seqLen, "CalcType": CAL_TYPE_PREFIX_ENCODER,
                                "maskType": MASK_TYPE_CAUSAL_MASK})
        q_seqlen = np.array(q_seqlens)
        q_seqlen = torch.from_numpy(q_seqlen).to(torch.int32).npu()
        kv_seqLen = np.array(kv_seqLen)
        kv_seqLen = torch.from_numpy(kv_seqLen).to(torch.int32).npu()
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [self.q.npu(), self.k_cache.npu(), self.v_cache.npu(),
                                                                self.block_tables.npu(), q_seqlen, kv_seqLen])
    def test_self_attention_encoder_operation_alibi_bf16(self):
        """
            is_decoder = 1, no_cache=False, "maskType": MASK_TYPE_NO_HEAD_DECODER, cache_type = 1
            qseqlen = kv_seqLen = [1024]
        """
        batch = random.randint(1, 10)
        kv_head = random.randint(1, 32)  # kv_head num
        isdecoder = 0  # prefill or decoder
        heads = kv_head * random.randint(1, 5)
        embeddim = random.choice([32, 64, 128])
        max_seq = random.randint(1, 2048)
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [max_seq] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        data_type = torch.bfloat16
        print(f"--batch:{batch}--kv_head:{kv_head}--heads:{heads}--embeddim:{embeddim}--max_seq:{max_seq}")

        self.set_data_params(dynamic_batch=dynamic_batch,
                             is_decoder=isdecoder, batch=batch, kv_head=kv_head, heads=heads,
                             embeddim=embeddim, max_seq=max_seq, kv_seqLen=kv_seqLen,
                             is_clamp=is_clamp, clamp_max=clamp_max, clamp_min=clamp_min,
                             data_type=data_type, is_alibi=True, tor=tor,
                             op_type=10, mask_type=MASK_TYPE_ALIBI_WITH_BATCH, no_cache=True)
        self.gen_out_tensor()
        self.mask = self.mask.to(torch.bfloat16)
        data = [self.q, self.k, self.v, self.mask, self.kv_seqLen, self.golden_out]
        param_seqlen = data[4]
        data[4] = torch.from_numpy(np.array(data[4]).astype(np.int32))
        in_tensors = [tensor.npu().contiguous() for tensor in data]
        OP_NAME = "SelfAttentionOperation"
        PARAM = json.dumps({"headNum": heads, "qkScale": 1.0 / math.sqrt(1.0 * embeddim), "kvHeadNum": kv_head,
                            "calcType": 3, "maskType": 2, "isTriuMask": 1, "kernelType": 0})
        RUN_PARAM = json.dumps({"seqLen": param_seqlen})
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [
            in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4]
        ])

    def test_self_attention_encoder_operation_compress_mask_swa_cycle_cache(self):
        """
            is_decoder = 0, no_cache=False, "maskType": MASK_TYPE_ALIBI_WITH_PREFIX_BATCH, cache_type = 1
            qseqlen = kv_seqLen = [128...]
        """
        if operation_test.get_soc_version() == 'Ascend910B':
            kv_head = 2
            window_size = 32
            mask_type = 8
            data = self.calc_expect_func(2, 1024, 2, 128, window_size, mask_type, group_num=kv_head)
            param_seqlen = data[4].tolist()
            in_tensors = [torch.from_numpy(tensor) for tensor in data]
            in_tensors = [tensor.npu() for tensor in in_tensors]
            a = [print(tensor.dtype, tensor.device) for tensor in in_tensors]

            OP_NAME = "SelfAttentionOperation"
            print("now qseqlen is ", self.q_seqlen)
            self.set_data_params(kv_head=kv_head, mask_type=mask_type, heads=self.heads, embeddim=self.embeddim, embeddimv=self.embeddimv, kv_seqLen=self.kv_seqLen, batch=2, window_size=window_size,
                                 no_cache=True)
            self.gen_out_tensor()
            PARAM = json.dumps({"headNum": kv_head, "qkScale": (1 / float(math.sqrt(128))), "kvHeadNum": kv_head, \
                "maskType": mask_type, "calcType": 3, "windowSize": 32, "cacheType": 1})
            RUN_PARAM = json.dumps({"seqLen": param_seqlen})
            print(PARAM, RUN_PARAM)
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        
        self.mask = gen_swa_cmp(window_size, self.embeddim).astype('float16')
        self.golden_out = torch.reshape(self.golden_out, (2048, 2, 128))
        self.golden_out_true = torch.reshape(self.golden_out_true, (2048, 2, 128))
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [
            torch.reshape(self.q, (2048, 2, 128)).npu(), torch.reshape(self.k, (2048, 2, 128)).npu(), torch.reshape(self.v, (2048, 2, 128)).npu(), torch.from_numpy(self.mask).npu(), torch.from_numpy(self.q_seqlen.astype(np.int32)).npu()
        ])

if __name__ == '__main__':
    unittest.main()
