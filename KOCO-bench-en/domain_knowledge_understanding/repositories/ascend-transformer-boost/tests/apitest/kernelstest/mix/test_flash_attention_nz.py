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
import unittest
import math
import numpy as np
import random
import sys, os
import torch
import op_test
sys.path.append('../')
from precision_calcu import *
np.random.seed(43)

class TestFlashAttentionNz(op_test.OpTest):
    def compare_output_data(self, out, golden, ratios):
        error_count = 0
        strict_error_count = 0
        alibi_error_count = 0
        fp16_min_normal = 1.0 / (1 << 14)
        out = out.flatten()
        golden = golden.flatten()
        out_len = out.shape[0]
        diff = torch.abs(golden - out)
        max_diff = diff.max().item()
        logging.info(f"maxDiff {max_diff}")
        golden = golden.to(torch.float32)
        out = out.to(torch.float32)
        if self.is_alibi:
            alibi_limit_error = torch.maximum(
                torch.abs(golden * ratios[4]), torch.tensor(ratios[5]))
            alibi_error_count = torch.gt(diff, alibi_limit_error).sum().item()
            logging.info("5/1000 Accuracy is %f",  1 -
                         float(alibi_error_count) / out_len)
            logging.info("accuracy is correct: %r", ((float(alibi_error_count) / out_len) <= ratios[4]))
        else:
            limit_error = torch.maximum(
                torch.abs(golden * ratios[0]), torch.tensor(ratios[1]))
            strict_limit_error = torch.maximum(
                torch.abs(golden * ratios[2]), torch.tensor(ratios[3]))
            error_count = torch.gt(diff, limit_error).sum().item()
            strict_error_count = torch.gt(
                diff, strict_limit_error).sum().item()
            logging.info("1/1000 Accuracy is %f",
                         1 - float(error_count) / out_len)
            logging.info("3/1000 Accuracy is %f",  1 -
                         float(strict_error_count) / out_len)
            logging.info("accuracy is correct: %r", (float(strict_error_count) / out_len) <= ratios[0])

        # 新精度标准fp16 参考精度标准v0.3浮点计算单标杆
        # 计算次数 两个matmul + 一个softmax
        calc_times = self.embed * self.max_seq + 4
        if calc_times < 2048:
            error = 2**(-8)
        else :
            error = 2**(-7)
        error_threshold = torch.clamp(torch.abs(golden), min = 1) * error
        return (diff <= error_threshold).all()

    def set_data_params(self, is_mask=True, is_batch_mask=False, is_decoder=False, variate_seq=False, is_alibi=False, is_alibi_128=False, is_alibi_256=False, \
                        left_align=False, is_sqrt=False, is_BNSD=False, is_logn_scale=False, is_swa=False, is_swa_compress=False, is_cycle_cache=False):
        self.is_mask = is_mask
        self.is_batch_mask = is_batch_mask
        self.is_decoder = is_decoder
        self.variate_seq = variate_seq
        self.is_alibi = is_alibi
        self.is_alibi_128 = is_alibi_128
        self.is_alibi_256 = is_alibi_256
        self.left_align = left_align
        self.is_sqrt = is_sqrt
        self.is_BNSD = is_BNSD
        self.is_logn_scale = is_logn_scale
        self.is_swa = is_swa
        self.is_swa_compress = is_swa_compress
        self.is_cycle_cache = is_cycle_cache

    def get_data_params(self):
        ret = (self.is_mask, self.is_batch_mask,self.is_decoder, self.variate_seq,self.is_alibi, self.is_alibi_128,
        self.is_alibi_256, self.is_BNSD, self.is_swa, self.is_swa_compress, self.is_cycle_cache)
        return ret

    def shape_nd_to_nz(self, shape, dtype='float16'):
        assert len(shape) >= 2
        batch = shape[:-2]
        a, b = shape[-2], shape[-1]
        a0, b0 = 16, 16
        return list(batch) + [math.ceil(b / b0), math.ceil(a / a0), a0, b0]

    def gen_axes_for_transpose(self, offset, base):
        return [x for x in range(offset)] + [x + offset for x in base]

    def convert_nd_to_nz(self, x):
        array_trans = self.gen_axes_for_transpose(
            len(x.shape) - 2, [2, 0, 1, 3])
        x_shape = self.shape_nd_to_nz(x.shape, dtype=x.dtype)
        *_, n1, m1, m0, n0 = x_shape
        return x.reshape(x_shape[:-4] + [m1, m0, n1, n0]).permute(*array_trans)

    def gen_seq_len(self, batch, max_seq, variate_seq=False):
        if variate_seq:
            num = max_seq // 16
            seqlen_aligned_arange = np.arange(1, min(batch + 1, num + 1)) * 16
            if batch > num:
                seqlen_aligned_remain = np.random.randint(
                    1, max_seq, size=(batch - num))
                seqlen_aligned_remain[:] = (
                    (seqlen_aligned_remain[:] + 15) // 16) * 16
                seqlen_aligned = np.concatenate(
                    (seqlen_aligned_arange, seqlen_aligned_remain), 0)
            else:
                seqlen_aligned = seqlen_aligned_arange
            sp_list = np.random.randint(0, 15, size=(batch))
            seqlen = seqlen_aligned - sp_list
        else:
            max_seq_aligned = (max_seq + 15) // 16 * 16
            sp_list = np.ones((batch,)) * (max_seq_aligned - max_seq)
            sp_list = sp_list.astype(np.int32)
            seqlen = np.ones((batch,)) * max_seq
            seqlen = seqlen.astype(np.int32)
            seqlen_aligned = np.ones((batch,)) * max_seq_aligned
            seqlen_aligned = seqlen_aligned.astype(np.int32)

        ntokens = seqlen.sum()
        return seqlen, seqlen_aligned, ntokens

    def group_matmul(self, heads, kvheads, mat_a, mat_b):
        group_heads = heads // kvheads
        score = None
        for i in range(kvheads):
            group_score = torch.matmul(mat_a[i * group_heads: (i + 1) * group_heads].to(torch.float32),
                                        mat_b[i:(i + 1), :, :].to(torch.float32))
            if score is None:
                score = group_score
            else:
                score = torch.cat((score, group_score), 0)
        return score
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
    def get_alibi_slopes(self, n_heads):
        n = 2 ** math.floor(math.log2(n_heads))
        m0 = 2.0 ** (-8.0 / n)
        slopes = torch.pow(m0, torch.arange(1, n + 1))
        if n < n_heads:
            m1 = 2.0 ** (-4.0 / n)
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
            self.alibi_slopes = self.alibi_slopes.half()
            bias = self.bias
        bias = bias * self.alibi_slopes[:, None, None]
        return bias

    def calc_logn(self, seqlen=1):
        if self.is_logn_scale:
            m = 8192 # seq_length in qwen config
            base = 3 * m
            self.logn_array = np.array([
                    math.log(i, m) if i > m else 1
                    for i in range(base, base + seqlen)
                ]).astype(np.float32)
        else:
            self.logn_array = []


    def calc_data(self, shape: tuple, window_size = 0):
        layer = 2
        batch, seqlen, heads, kv_head, embed, max_seq, mask_dim = shape
        self.embed, self.max_seq = embed, max_seq
        is_mask, is_batch_mask, is_decoder, variate_seq, is_alibi, is_alibi_128, is_alibi_256, is_BNSD, is_swa, is_swa_compress, is_cycle_cache = self.get_data_params()
        if is_decoder:
            q_seqlen, q_seqlen_aligned, q_ntokens = self.gen_seq_len(
                batch, 1, variate_seq)
        else:
            q_seqlen, q_seqlen_aligned, q_ntokens = self.gen_seq_len(
                batch, seqlen, variate_seq)
        kv_seqlen, kv_seqlen_aligned, kv_ntokens = self.gen_seq_len(
            batch, seqlen, variate_seq)

        q_ntokens = q_seqlen_aligned.sum()
        kv_ntokens = kv_seqlen_aligned.sum()

        layer_id = torch.tensor([layer - 1], dtype=torch.int32)
        q = torch.empty((q_ntokens, heads * embed), dtype=torch.float16).uniform_(-1.0, 1.0)
        k = torch.empty((batch * seqlen, kv_head * embed), dtype=torch.float16).uniform_(-1.0, 1.0)
        v = torch.empty((batch * seqlen, kv_head * embed), dtype=torch.float16).uniform_(-1.0, 1.0)
        if is_BNSD and not is_decoder:
            q = torch.empty((batch * seqlen, heads * embed), dtype=torch.float16).uniform_(-1.0, 1.0)
            q_max = torch.zeros((batch * max_seq, heads * embed), dtype=torch.float16)
            for i in range(batch):
                q_max[max_seq * i:(max_seq * i) + seqlen] = q[seqlen * i:(seqlen * i) + seqlen]
            q = q_max

        k_max = torch.zeros((layer, batch * max_seq, kv_head * embed), dtype=torch.float16)
        v_max = torch.zeros((layer, batch * max_seq, kv_head * embed), dtype=torch.float16)

        k_max[layer_id[0]][:batch * seqlen] = k
        v_max[layer_id[0]][:batch * seqlen] = v
        if is_mask:
            if is_alibi:
                bsz_heads = batch * heads if mask_dim == 4 else heads
                mask = torch.ones((bsz_heads, max_seq, max_seq), dtype=torch.float16)
                mask *= -60000
                mask = torch.triu(mask, diagonal=1)
                self.alibi_bias = self.get_alibi_bias(heads, max_seq)
                mask = mask.reshape(batch, heads, max_seq, max_seq) if mask_dim == 4 else mask
                mask += self.alibi_bias

            elif (is_swa or is_swa_compress) and window_size > 0:
                mask = torch.ones((1, max_seq, max_seq), dtype=torch.float16)
                mask_triu = torch.triu(mask, diagonal=1)
                mask_trid = torch.tril(mask, diagonal=(-window_size))
                mask = mask_triu + mask_trid
                mask *= -10000.0
            elif mask_dim == 2:
                mask = torch.ones((1, max_seq, max_seq), dtype=torch.float16)
                mask *= -10000.0
                mask = torch.triu(mask, diagonal=1)
            else:
                mask = torch.ones((batch, max_seq, max_seq), dtype=torch.float16)
                mask *= -10000.0
                mask = torch.triu(mask, diagonal=1)

        elif is_decoder:
            if is_alibi:
                bsz_heads = batch * heads if mask_dim == 4 else heads
                mask = torch.ones((bsz_heads, 16, max_seq), dtype=torch.float16)
                mask *= -10000
                mask = torch.triu(mask, diagonal=1)
                for i in range(bsz_heads):
                    temp = torch.zeros(16, max_seq).to(torch.float16)
                    temp.uniform_(-2, 2)
                    mask[i] += temp
                mask = mask.reshape(batch, heads, 16, max_seq) if mask_dim == 4 else mask
            elif mask_dim == 2:
                mask = torch.zeros((1, 16, max_seq), dtype=torch.float16)
                mask[0, :1, :2] = -10000
            else:
                mask = torch.zeros((batch, 16, max_seq), dtype=torch.float16)
                for i in range(batch):
                    mask[i, :1, :i] = -10000

        if self.is_decoder:
            self.calc_logn(batch)
        else:
            self.calc_logn(seqlen)
        q_offset = 0
        k_offset = 0
        v_offset = 0
        input1_nd = None
        input2 = None
        input3 = None
        out = None
        for idx in range(batch):
            s_align = q_seqlen_aligned[idx]
            kv_align = kv_seqlen_aligned[idx]
            s = q_seqlen[idx]
            kv = kv_seqlen[idx]
            q_slice = q[q_offset:q_offset + s][:]
            q_slice = q_slice.reshape(s, heads, embed)
            q_slice = q_slice.permute(1, 0, 2)  # (heads, token, emd)
            # 计算
            k_slice = k_max[layer_id[0]][k_offset:k_offset + kv][:]
            k_slice = k_slice.reshape(kv, kv_head, embed)
            # heads*max_seqlen*embed
            k_slice = k_slice.permute(1, 0, 2)
            # 输入
            k_slice_max = k_max[layer_id[0]][k_offset:k_offset + max_seq][:]
            k_slice_max = k_slice_max.reshape(max_seq, kv_head, embed)
            # heads*max_seqlen*embed
            k_slice_max = k_slice_max.permute(1, 0, 2)
            k_slice_t = k_slice.permute(0, 2, 1)   # get K^T
            # 计算
            v_slice = v_max[layer_id[0]][v_offset:v_offset + kv][:]
            v_slice = v_slice.reshape(kv, kv_head, embed)
            v_slice = v_slice.permute(1, 0, 2)
            # 输入
            v_slice_max = v_max[layer_id[0]][v_offset:v_offset + max_seq][:]
            v_slice_max = v_slice_max.reshape(max_seq, kv_head, embed)
            v_slice_max = v_slice_max.permute(1, 0, 2)
            score = self.group_matmul(heads, kv_head, q_slice, k_slice_t)
            scorehigh_pre = score.to(torch.float16)
            if self.is_logn_scale:
                if self.is_decoder:
                    scorehigh_pre *= self.logn_array.astype(np.float16)[idx]
                else:
                    scorehigh_pre *= self.logn_array.astype(np.float16)[None, :s, None]

            tor = float(1/math.sqrt(1.0 * embed))
            if self.is_logn_scale:
                if self.is_decoder:
                    tor *= self.logn_array[idx]
                else:
                    tor *= self.logn_array[None, :s, None]
            score = score * tor
            torhigh_pre = tor = torch.tensor(1/math.sqrt(1.0 * embed), dtype=torch.float16)
            scorehigh_pre = (scorehigh_pre * torhigh_pre).to(torch.float16)

            if mask_dim != 0:
                if is_alibi:
                    if mask_dim == 4:
                        score += mask[idx][:, :s, :kv].to(torch.float32)
                        scorehigh_pre += mask[idx][:, :s, :kv].to(torch.float16)
                    else:
                        score += mask[:, :s, :kv].to(torch.float32)
                        scorehigh_pre += mask[:, :s, :kv].to(torch.float16)
                elif mask_dim == 2:
                    score += mask[0][:s, :kv].to(torch.float32)
                    scorehigh_pre += mask[0][:s, :kv].to(torch.float16)
                else:
                    score = score + mask[idx, :s, :kv].to(torch.float32)
                    scorehigh_pre = scorehigh_pre + mask[idx, :s, :kv].to(torch.float16)
            if is_decoder and is_swa and (not is_cycle_cache) and kv > window_size:
                window_mask = torch.zeros((s, max_seq)).to(torch.float16)
                window_mask[:s, :(kv - window_size)] = -10000
                score += window_mask[:s, :kv].to(torch.float32)
                scorehigh_pre += window_mask[:s, :kv].to(torch.float16)
            score_max,_ = torch.max(score, dim=-1)
            score = score - score_max.reshape((heads, s, 1))
            score_exp = torch.exp(score.to(torch.float32))
            score_sum = torch.sum(score_exp, dim=-1)
            p = score_exp / score_sum.reshape((heads, s, 1))
            score_maxhigh_pre,_ = torch.max(scorehigh_pre, dim=-1)
            scorehigh_pre = scorehigh_pre - score_maxhigh_pre.reshape((heads, s, 1))
            score_exphigh_pre = torch.exp(scorehigh_pre.to(torch.float32))
            score_sumhigh_pre = torch.sum(score_exphigh_pre, dim=-1).to(torch.float32)
            phigh_pre=score_exphigh_pre.to(torch.float16)/(score_sumhigh_pre.reshape((heads, s, 1))).to(torch.float16)
            o_mat = self.group_matmul(heads, kv_head, p,v_slice)
            o_mat = o_mat.reshape(heads, s, embed)
            o_mathigh_pre = self.group_matmul(heads, kv_head, phigh_pre, v_slice)
            o_mathigh_pre = o_mathigh_pre.reshape(heads, s, embed)
            if is_BNSD and not is_decoder:
                q_pad = torch.zeros(heads, max_seq, embed).to(torch.float16)
                o_pad = torch.zeros(heads, max_seq, embed).to(torch.float16)
                o_padhigh_pre = torch.zeros(heads, max_seq, embed).to(torch.float16)
            else:
                q_pad = torch.zeros(heads, s_align, embed).to(torch.float16)
                o_pad = torch.zeros(heads, s_align, embed)
                o_padhigh_pre = torch.zeros(heads, s_align, embed)
            q_pad[:, :s, :] = q_slice
            o_pad[:, :s, :] = o_mat
            o_padhigh_pre[:, :s, :] = o_mathigh_pre
            if input1_nd is None:
                input1_nd = q_pad
            else:
                input1_nd = torch.cat((input1_nd, q_pad), 1) if not is_BNSD else torch.cat((input1_nd, q_pad), 0)
            input2_slice = self.convert_nd_to_nz(k_slice_max)
            if input2 is None:
                input2 = input2_slice.reshape([-1, 16, 16])
            else:
                input2 = torch.cat(
                    (input2, input2_slice.reshape([-1, 16, 16])), 0)
            input3_slice = self.convert_nd_to_nz(v_slice_max)
            if input3 is None:
                input3 = input3_slice.reshape([-1, 16, 16])
            else:
                input3 = torch.cat(
                    (input3, input3_slice.reshape([-1, 16, 16])), 0)
            if out is None:
                out = o_pad
                outhigh_pre = o_padhigh_pre
            else:
                out = torch.cat((out, o_pad), 1) if not is_BNSD else torch.cat((out, o_pad), 0)
                outhigh_pre = torch.cat((outhigh_pre, o_padhigh_pre), 1) if not is_BNSD else torch.cat((outhigh_pre, o_padhigh_pre), 0)

            if is_BNSD and not is_decoder:
                q_offset += max_seq
            else:
                q_offset += s
            k_offset += max_seq
            v_offset += max_seq
        q = self.convert_nd_to_nz(input1_nd)
        if is_BNSD and not is_decoder:
            self.q = q.reshape(batch * heads, embed // 16, max_seq, 16)
        elif is_BNSD and is_decoder:
            self.q = q.reshape(batch * heads, embed // 16, 16, 16)
        else:
            self.q = q.reshape(1, heads * embed // 16, q_ntokens, 16)

        input2_shape0, input2_shape1, input2_shape2 = input2.shape

        k = torch.zeros(layer, input2_shape0, input2_shape1,
                     input2_shape2).to(torch.float16)
        k[layer_id[0]] = input2
        self.k = k.reshape(layer, batch, kv_head * embed // 16, max_seq, 16) if not is_BNSD else k.reshape(layer, batch * kv_head, embed // 16, max_seq, 16)
        input3_shape0, input3_shape1, input3_shape2 = input3.shape
        v = torch.zeros(layer, input3_shape0, input3_shape1,
                     input3_shape2).to(torch.float16)

        v[layer_id[0]] = input3
        self.v = v.reshape(layer, batch, kv_head * embed // 16, max_seq, 16) if not is_BNSD else v.reshape(layer, batch * kv_head, embed // 16, max_seq, 16)
        if mask_dim == 4 and is_alibi_128:
            mask = mask[0, :, :, :128]
        mask = self.convert_nd_to_nz(mask)
        if is_mask:
            if is_alibi:
                if mask_dim == 4:
                    if is_alibi_128:
                        self.mask = mask.reshape(heads, 128 // 16, max_seq, 16)
                    else:
                        self.mask = mask.reshape(batch * heads, max_seq // 16, max_seq, 16)
                else:
                    self.mask = mask.reshape(heads, max_seq // 16, max_seq, 16)
            elif (is_swa or is_swa_compress):
                self.mask = mask.reshape(1, max_seq // 16, max_seq, 16)
            elif mask_dim == 2:
                self.mask = mask.reshape(1, max_seq // 16, max_seq, 16)
            else:
                self.mask = mask.reshape(batch, max_seq // 16, max_seq, 16)
        elif is_decoder:
            if is_alibi:
                if mask_dim == 4:
                    self.mask = mask.reshape(
                        batch * heads, max_seq // 16, 16, 16)
                else:
                    self.mask = mask.reshape(heads, max_seq // 16, 16, 16)
            elif mask_dim == 2:
                self.mask = mask.reshape(1, max_seq // 16, 16, 16)
            else:
                self.mask = mask.reshape(batch, max_seq // 16, 16, 16)
        if is_alibi_256:
            if self.left_align:
                mask = torch.ones((256, 256)) * -float('inf')
                mask = torch.triu(mask, diagonal=1).to(torch.float16)
                mask = self.bias[0, :256, :256] + mask
                mask = self.convert_nd_to_nz(mask)
                self.mask = mask.reshape(1, 16, 256, 16)
            else:
                self.alibi_slopes *= -1
                mask = torch.ones((256, 256)) * 60000
                mask = torch.triu(mask, diagonal=1)
                if max_seq < 256:
                    self.bias = torch.tensor(np.pad(self.bias, ((0, 256 - max_seq), (0, 256 - max_seq)), "constant"))
                mask = self.bias[:256, :256] * -1 + mask
                mask = self.convert_nd_to_nz(mask)
                self.mask = mask.reshape(1, 16, 256, 16)
        else:
            self.alibi_slopes = []
        self.layer_id = layer_id
        if is_BNSD and not is_decoder:
            out_nz = self.convert_nd_to_nz(out).reshape(batch * heads, embed // 16, max_seq, 16)
            out_nzhigh_pre = self.convert_nd_to_nz(outhigh_pre.to(torch.float16)).reshape(batch * heads, embed // 16, max_seq, 16)
        elif is_BNSD and is_decoder:
            out_nz = self.convert_nd_to_nz(out).reshape(batch * heads, embed // 16, 16, 16)
            out_nzhigh_pre = self.convert_nd_to_nz(outhigh_pre.to(torch.float16)).reshape(batch * heads, embed // 16, 16, 16)
        else:
            out_nz = self.convert_nd_to_nz(out).reshape(1, heads * embed // 16, q_ntokens, 16)
            out_nzhigh_pre = self.convert_nd_to_nz(outhigh_pre.to(torch.float16)).reshape(1, heads * embed // 16, q_ntokens, 16)
        self.golden_out = out_nz
        self.golden_outhigh_pre = out_nzhigh_pre.to(torch.float32)
        self.q_seqlen = q_seqlen
        self.q_ntokens = q_ntokens
        self.kv_seqlen = kv_seqlen
        self.kv_ntokens = kv_ntokens

        logging.debug("**********data gen shape***********")
        logging.debug(f"q shape: {q.shape}")
        logging.debug(f"k shape: {k.shape}")
        logging.debug(f"v shape: {v.shape}")
        logging.debug(f"layer_id shape: {layer_id.shape}")
        logging.debug(f"mask shape: {mask.shape}")
    def calc_data_pa(self, shape: tuple, window_size = 0):
        batch, seqlen, heads, kv_head, embed, max_seq, mask_dim = shape
        self.embed, self.max_seq = embed, max_seq
        is_mask, is_batch_mask, is_decoder, variate_seq, is_alibi, is_alibi_128, is_alibi_256, is_BNSD, is_swa, is_swa_compress, is_cycle_cache = self.get_data_params()
        if is_decoder:
            q_seqlen, q_seqlen_aligned, q_ntokens = self.gen_seq_len(
                batch, 1, variate_seq)
        else:
            q_seqlen, q_seqlen_aligned, q_ntokens = self.gen_seq_len(
                batch, seqlen, variate_seq)

        q_ntokens = q_seqlen.sum()
        q_ntokens_aligned = (q_ntokens + 15) // 16 * 16
        q = torch.empty((q_ntokens_aligned,heads * embed), dtype=torch.float16).uniform_(-1.0, 1.0)
        k = torch.empty((q_ntokens_aligned,kv_head * embed), dtype=torch.float16).uniform_(-1.0, 1.0)
        v = torch.empty((q_ntokens_aligned,kv_head * embed), dtype=torch.float16).uniform_(-1.0, 1.0)
        max_seq = (seqlen + 15) // 16 * 16
        if is_alibi:
            bsz_heads = batch * heads if mask_dim == 4 else heads
            mask = torch.ones((bsz_heads, max_seq, max_seq), dtype=torch.float16)
            mask *= -60000
            mask = torch.triu(mask, diagonal=1)
            self.alibi_bias = self.get_alibi_bias(heads, max_seq)
            mask = mask.reshape(
                (batch, heads, max_seq, max_seq)) if mask_dim == 4 else mask
            mask += self.alibi_bias
        elif (is_swa or is_swa_compress) and window_size > 0:
            mask = torch.ones((1, max_seq, max_seq), dtype=torch.float16)
            mask_triu = torch.triu(mask, diagonal=1)
            mask_trid = torch.tril(mask, diagonal=(-window_size))
            mask = mask_triu + mask_trid
            mask *= -10000.0
        elif mask_dim == 2:
            mask = torch.ones((1, max_seq, max_seq), dtype=torch.float16)
            mask *= -10000.0
            mask = torch.triu(mask, diagonal=1)
        else:
            mask = torch.ones((batch, max_seq, max_seq), dtype=torch.float16)
            mask *= -10000.0
            mask = torch.triu(mask, diagonal=1)

        self.calc_logn(seqlen)
        q_offset = 0
        k_offset = 0
        v_offset = 0
        input1_nd = None
        input2_nd = None
        input3_nd = None
        out = None

        for idx in range(batch):
            s_align = q_seqlen_aligned[idx]
            s = q_seqlen[idx]

            q_slice = q[q_offset:q_offset + s][:]
            q_slice = q_slice.reshape(s, heads, embed)
            q_slice = q_slice.permute(1, 0, 2)# (head, token, emd)

            # 计算
            k_slice = k[k_offset:k_offset + s][:]
            k_slice = k_slice.reshape(s, kv_head, embed)
            k_slice = k_slice.permute(1, 0, 2)

            # 输入
            k_slice_t = k_slice.permute(0, 2, 1)   # get K^T

            # 计算
            v_slice = v[v_offset:v_offset + s][:]
            v_slice = v_slice.reshape(s, kv_head, embed)
            # heads*max_seqlen*embed
            v_slice = v_slice.permute(1, 0, 2)
            score = self.group_matmul(heads, kv_head, q_slice, k_slice_t)
            tor = float(1/math.sqrt(1.0 * embed))

            scorehigh_pre = score.to(torch.float16)
            if self.is_logn_scale:
                scorehigh_pre *= self.logn_array.astype(np.float16)[None, :seqlen, None]
            torhigh_pre = tor = torch.tensor(1/math.sqrt(1.0 * embed), dtype=torch.float16)
            scorehigh_pre = (scorehigh_pre * torhigh_pre).to(torch.float16)
            if self.is_logn_scale:
                score *= self.logn_array[None, :seqlen, None]
            score = score * tor
            if mask_dim != 0:
                if is_alibi:
                    if mask_dim == 4:
                        score += mask[idx][:, :s, :s].to(torch.float32)
                        scorehigh_pre += mask[idx][:, :s, :s].to(torch.float16)
                    else:
                        score += mask[:, :s, :s].to(torch.float32)
                        scorehigh_pre += mask[:, :s, :s].to(torch.float16)
                else:
                    score = score + mask[0][:s, :s].to(torch.float32)
                    scorehigh_pre += mask[0][:s, :s].to(torch.float16)

            score_max,_ = torch.max(score, dim=-1)
            score = score - score_max.reshape((heads, s, 1))
            score_exp = torch.exp(score.to(torch.float32))
            score_sum = torch.sum(score_exp, dim=-1)
            p = score_exp / score_sum.reshape((heads, s, 1))

            score_maxhigh_pre,_ = torch.max(scorehigh_pre, dim=-1)
            scorehigh_pre = scorehigh_pre - score_maxhigh_pre.reshape((heads, s, 1))
            score_exphigh_pre = torch.exp(scorehigh_pre.to(torch.float32))
            score_sumhigh_pre = torch.sum(score_exphigh_pre, dim=-1).to(torch.float32)
            phigh_pre=score_exphigh_pre.to(torch.float16)/(score_sumhigh_pre.reshape((heads, s, 1))).to(torch.float16)

            o_mat = self.group_matmul(heads, kv_head, p, v_slice)
            o_mat = o_mat.reshape(heads, s, embed)
            o_mathigh_pre = self.group_matmul(heads, kv_head, phigh_pre, v_slice)
            o_mathigh_pre = o_mathigh_pre.reshape(heads, s, embed)
            if input1_nd is None:
                input1_nd = q_slice
            else:
                input1_nd = torch.cat((input1_nd, q_slice), 1)


            if input2_nd is None:
                input2_nd = k_slice
            else:
                input2_nd = torch.cat((input2_nd, k_slice), 1)


            if input3_nd is None:
                input3_nd = v_slice
            else:
                input3_nd = torch.cat((input3_nd, v_slice), 1)

            if out is None:
                out = o_mat
                outhigh_pre = o_mathigh_pre
            else:
                 out = torch.cat((out, o_mat), 1)
                 outhigh_pre = torch.cat((outhigh_pre, o_mathigh_pre), 1)

            q_offset += s
            k_offset += s
            v_offset += s
        q_nd = torch.zeros((heads, q_ntokens_aligned, embed), dtype=torch.float16)
        q_nd[:, :q_ntokens, :] = input1_nd
        k_nd = torch.zeros((kv_head, q_ntokens_aligned, embed), dtype=torch.float16)
        k_nd[:, :q_ntokens, :] = input2_nd
        v_nd = torch.zeros((kv_head, q_ntokens_aligned, embed), dtype=torch.float16)
        v_nd[:, :q_ntokens, :] = input3_nd
        o_nd = torch.zeros((heads, q_ntokens_aligned, embed), dtype=torch.float16)
        o_nd[:, :q_ntokens, :] = out
        o_ndhigh_pre = torch.zeros((heads, q_ntokens_aligned, embed), dtype=torch.float16)

        o_ndhigh_pre[:, :q_ntokens, :] = outhigh_pre
        q = self.convert_nd_to_nz(q_nd)
        self.q = q.reshape(
            1, heads * embed // 16, q_ntokens_aligned, 16)
        k = self.convert_nd_to_nz(k_nd)
        self.k = k.reshape(1, kv_head * embed // 16, q_ntokens_aligned, 16)
        v = self.convert_nd_to_nz(v_nd)
        self.v = v.reshape(1, kv_head * embed // 16, q_ntokens_aligned, 16)
        if mask_dim == 4 and is_alibi_128:
            mask = mask[0, :, :, :128]
        mask = self.convert_nd_to_nz(mask)
        if is_alibi:
            if mask_dim == 4:
                if is_alibi_128:
                    self.mask = mask.reshape(heads, 128 // 16, max_seq, 16)
                else:
                    self.mask = mask.reshape(batch * heads, max_seq // 16, max_seq, 16)
            else:
                self.mask = mask.reshape(heads, max_seq // 16, max_seq, 16)
        elif mask_dim == 2:
            self.mask = mask.reshape(1, max_seq // 16, max_seq, 16)
        else:
            self.mask = mask.reshape(batch, max_seq // 16, max_seq, 16)
        if is_alibi_256:
            self.alibi_slopes *= -1
            mask = torch.ones((256, 256)) * 60000
            mask = torch.triu(mask, diagonal=1).to(torch.float16)
            if max_seq < 256:
                self.bias = torch.tensor(np.pad(self.bias, ((0, 256 - max_seq), (0, 256 - max_seq)), "constant"))
            mask = self.bias[:256, :256] * -1 + mask
            mask = self.convert_nd_to_nz(mask)
            self.mask = mask.reshape(1, 16, 256, 16)
        else:
            self.alibi_slopes = []
        # golden data
        out_nz = self.convert_nd_to_nz(o_nd).reshape(1, heads * embed // 16, q_ntokens_aligned, 16)
        self.golden_out = out_nz
        out_nzhigh_pre = self.convert_nd_to_nz(o_ndhigh_pre.to(torch.float16)).reshape(1, heads * embed // 16, q_ntokens_aligned, 16)
        self.golden_outhigh_pre = out_nzhigh_pre.to(torch.float32)
        self.q_seqlen = q_seqlen
        self.q_ntokens = q_ntokens
        self.kv_seqlen = q_seqlen
        self.kv_ntokens = q_ntokens
        self.layer_id = torch.tensor([], dtype=torch.int32)
        logging.debug("**********data gen shape***********")
        logging.debug(f"q shape: {q.shape}")
        logging.debug(f"k shape: {k.shape}")
        logging.debug(f"v shape: {v.shape}")
        logging.debug(f"mask shape: {mask.shape}")

    def golden_calc(self, in_tensors):
        golden_out = torch.tensor(self.golden_out)
        return [golden_out]

    def golden_compare(self, out_tensors, golden_tensors):
        result_old = self.compare_output_data(out_tensors[0].half(), golden_tensors[0].half(), [0.001, 0.001, 0.003, 0.003, 0.005, 0.005])
        result_double = compare_cv(golden_tensors[0], self.golden_outhigh_pre, out_tensors[0].to(torch.float32))
        return result_old or result_double
    @op_test.only_310p
    def test_flash_attention_case_encoder_cache_with_left_alibi_mask(self):
        batch = 1
        qkv_seq = 256
        kv_head = 1
        heads = 1
        embeddim = 128
        max_seq = 2064
        tor = math.sqrt(1 / embeddim)
        mask_dim = 4
        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 7, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, 
                    "isTriuMask" : 1, "maskType": 2, "kvHead": kv_head, "alibiLeftAlign": 1, "precType": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])
        self.set_data_params(is_alibi=True, is_alibi_256=True, left_align=True)
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute(
            [
                torch.tensor(self.q).half(),
                torch.tensor(self.k).half(),
                torch.tensor(self.v).half(),
                torch.tensor(self.layer_id).int(),
                torch.tensor(self.mask).half(),
                torch.tensor(self.alibi_slopes).float(),
                torch.tensor([]).float()
            ],
            [torch.tensor(attention_out).half()]
            )
    @op_test.only_310p
    def test_flash_attention_case_encoder_cache_mask_none(self):
        batch = 1
        qkv_seq = 13

        kv_head = 8
        heads = 32
        embeddim = 128
        max_seq = 2064
        tor = math.sqrt(1 / embeddim)
        mask_dim = 0

        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 17, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, 
        "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 0, "precType": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params()
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))

        logging.info("**********input shape***********")
        logging.info(f"q shape: {self.q.shape}")
        logging.info(f"k shape: {self.k.shape}")
        logging.info(f"v shape: {self.v.shape}")
        logging.info(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        
        return  self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), 
            torch.tensor(torch.tensor(self.layer_id)).int(), torch.tensor([]).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])
    @op_test.only_310p_and_910a
    def test_flash_attention_case_encoder_cache_no_mask(self):
        logging.debug("======================test flash attention nz encoder cache norm mask dim3======================")
        import random
        random.seed(0)
        np.random.seed(123) 
        torch.manual_seed(0)
        batch = 1
        qkv_seq = 256
        kv_head = 1      # kv_head num
        heads = 1          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 256
        tor = math.sqrt(1 / embeddim)
        mask_dim = 0

        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 17, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, 
        "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params()
        self.calc_data((batch,qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))

        logging.info("**********input shape***********")
        logging.info(f"q shape: {self.q.shape}")
        logging.info(f"k shape: {self.k.shape}")
        logging.info(f"v shape: {self.v.shape}")
        logging.info(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), 
        torch.tensor(torch.tensor(self.layer_id)).int(), torch.tensor([]).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])
    @op_test.only_310p_and_910a
    def test_flash_attention_case_encoder_mask_none_BNSD(self):
        logging.debug("======================test flash attention nz encoder no mask BNSD======================")
        batch = 5
        qkv_seq = 1024
        kv_head = 4        # kv_head num
        heads = 4          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        tor = math.sqrt(1 / embeddim)
        mask_dim = 0

        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 17, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 0, "dataDimOrder": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_BNSD=True)
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor([]).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])
    @op_test.only_310p
    def test_flash_attention_case_decoder_mask_none_BNSD(self):
        logging.debug("======================test flash attention nz decoder no mask BNSD======================")
        batch = 3
        qkv_seq = 256
        kv_head = 2        # kv_head num
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        mask_dim = 0
        tor = math.sqrt(1 / embeddim)
        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 18, "qSeqLen": [1] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 0, "dataDimOrder": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_mask=False, is_decoder=True, is_BNSD=True)
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor([]).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p
    def test_flash_attention_case_decoder_norm_mask_dim3_BNSD(self):
        logging.debug("======================test flash attention nz decoder norm mask dim3 BNSD======================")
        batch = 3
        qkv_seq = 256
        kv_head = 2        # kv_head num
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 96
        max_seq = 2048
        mask_dim = 3
        tor = math.sqrt(1 / embeddim)
        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 18, "qSeqLen": [1] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 1, "dataDimOrder": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_mask=False, is_decoder=True, is_BNSD=True)
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p_and_910a
    def test_flash_attention_case_encoder_norm_mask_dim3_BNSD(self):
        logging.debug("======================test flash attention nz encoder norm mask dim3 BNSD======================")
        batch = 2
        qkv_seq = 1240
        kv_head = 4        # kv_head num
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        tor = math.sqrt(1 / embeddim)
        mask_dim = 3

        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 17, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 1, "dataDimOrder": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_BNSD=True)
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p_and_910a
    def test_flash_attention_case_encoder_norm_mask_dim2_BNSD(self):
        logging.debug("======================test flash attention nz encoder norm mask dim2 BNSD======================")
        batch = 2
        qkv_seq = 1241
        kv_head = 8        # kv_head num
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        tor = math.sqrt(1 / embeddim)
        mask_dim = 2

        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 17, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 1, "isTriuMask": 1, "dataDimOrder": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_BNSD=True)
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p_and_910a
    def test_flash_attention_case_encoder_norm_mask_long_seq_BNSD(self):
        logging.debug("======================test flash attention nz encoder norm long seq BNSD======================")
        batch = 2
        qkv_seq = 2048
        kv_head = 8        # kv_head num
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        tor = math.sqrt(1 / embeddim)
        mask_dim = 2

        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 17, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 1, "isTriuMask": 1, "dataDimOrder": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_BNSD=True)
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))
        mask = torch.ones((128, 128), dtype=torch.float16)
        mask = torch.triu(mask, diagonal=1)
        mask *= -10000.0
        mask = mask.to(torch.float16)
        mask = self.convert_nd_to_nz(mask)
        self.mask = mask.reshape(1, 8, 128, 16)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p
    def test_flash_attention_case_decoder_norm_mask_dim2_BNSD(self):
        logging.debug("======================test flash attention nz decoder norm mask dim2 BNSD======================")
        batch = 3
        qkv_seq = 1241
        kv_head = 2        # kv_head num
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        mask_dim = 2
        tor = math.sqrt(1 / embeddim)
        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 18, "qSeqLen": [1] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 1, "dataDimOrder": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_mask=False, is_decoder=True, is_BNSD=True)
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p_and_910a
    def test_flash_attention_encoder_left_alibi_dim4_BNSD(self):
        logging.debug("======================test flash attention nz encoder left alibi mask dim4 BNSD======================")
        batch = 1
        qkv_seq = 128
        kv_head = 64        # kv_head num  1
        heads = 64          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 128
        mask_dim = 4
        tor = math.sqrt(1 / embeddim)
        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 17, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 2, "dataDimOrder": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_alibi=True, is_BNSD=True)
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))
        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p
    def test_flash_attention_nz_case_decoder_alibi_mask_dim3_BNSD(self):
        logging.debug("======================test flash attention nz decoder alibi mask dim3 BNSD======================")
        batch = 3
        qkv_seq = 511
        kv_head = 8        # kv_head num
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 96
        max_seq = 768
        mask_dim = 3
        tor = math.sqrt(1 / embeddim)
        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 18, "qSeqLen": [1] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 2, "dataDimOrder": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_mask=False, is_decoder=True, is_alibi=True, is_BNSD=True)
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p_and_910a
    def test_flash_attention_case_encoder_norm_mask_dim3(self):
        logging.debug("======================test flash attention nz encoder norm mask dim3======================")
        batch = 2
        qkv_seq = 1240
        kv_head = 4        # kv_head num
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        tor = math.sqrt(1 / embeddim)
        mask_dim = 3

        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 17, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params()
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p_and_910a
    def test_flash_attention_case_encoder_norm_mask_dim2(self):
        logging.debug("======================test flash attention nz encoder norm mask dim2======================")
        batch = 2
        qkv_seq = 1241
        kv_head = 8        # kv_head num
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        tor = math.sqrt(1 / embeddim)
        mask_dim = 2

        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 17, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 1, "isTriuMask": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params()
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p_and_910a
    def test_flash_attention_case_encoder_norm_mask_long_seq(self):
        logging.debug("======================test flash attention nz encoder norm long seq======================")
        batch = 2
        qkv_seq = 2048
        kv_head = 8        # kv_head num
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        tor = math.sqrt(1 / embeddim)
        mask_dim = 2

        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 17, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 1, "isTriuMask": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params()
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))
        mask = torch.ones((128, 128), dtype=torch.float16)
        mask = torch.triu(mask, diagonal=1)
        mask *= -10000.0
        mask = mask.to(torch.float16)
        mask = self.convert_nd_to_nz(mask)
        self.mask = mask.reshape(1, 8, 128, 16)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p_and_910a
    def test_flash_attention_case_encoder_mask_none(self):
        logging.debug("======================test flash attention nz encoder no mask======================")
        batch = 2
        qkv_seq = 1024 + 256 + 88
        kv_head = 4        # kv_head num
        heads = 4          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        tor = math.sqrt(1 / embeddim)
        mask_dim = 0

        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 17, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params()
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor([]).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p
    def test_flash_attention_case_decoder_norm_mask_dim3(self):
        logging.debug("======================test flash attention nz decoder norm mask dim3======================")
        batch = 3
        qkv_seq = 256
        kv_head = 2        # kv_head num
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 96
        max_seq = 2048
        mask_dim = 3
        tor = math.sqrt(1 / embeddim)
        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 18, "qSeqLen": [1] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_mask=False, is_decoder=True)
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p
    def test_flash_attention_case_decoder_norm_mask_dim2(self):
        logging.debug("======================test flash attention nz decoder norm mask dim2======================")
        batch = 3
        qkv_seq = 1241
        kv_head = 2        # kv_head num
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        mask_dim = 2
        tor = math.sqrt(1 / embeddim)
        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 18, "qSeqLen": [1] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_mask=False, is_decoder=True)
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p
    def test_flash_attention_case_decoder_mask_none(self):
        logging.debug("======================test flash attention nz decoder no mask======================")
        batch = 3
        qkv_seq = 256
        kv_head = 2        # kv_head num
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        mask_dim = 0
        tor = math.sqrt(1 / embeddim)
        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 18, "qSeqLen": [1] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_mask=False, is_decoder=True)
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor([]).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p
    def test_flash_attention_case_decoder_alibi_mask_dim4(self):
        logging.debug("======================test flash attention nz decoder alibi mask dim4======================")
        batch = 3
        qkv_seq = 511
        kv_head = 8        # kv_head num
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 768
        mask_dim = 4
        tor = math.sqrt(1 / embeddim)
        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 18, "qSeqLen": [1] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 2}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_mask=False, is_decoder=True, is_alibi=True)
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p
    def test_flash_attention_case_decoder_norm_mask_logn_dim3(self):
        logging.info("======================test flash attention nz decoder norm mask dim3======================")
        batch = 3
        qkv_seq = 256
        kv_head = 32  # kv_head num for qwen
        heads = 32
        embeddim = 96
        max_seq = 2048
        mask_dim = 3
        tor = math.sqrt(1 / embeddim)
        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 18, "qSeqLen": [1] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 1, "scaleType": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_mask=False, is_decoder=True, is_logn_scale=True)
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor(self.logn_array).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p_and_910a
    def test_flash_attention_encoder_left_alibi_dim4(self):
        logging.debug("======================test flash attention nz encoder left alibi mask dim4======================")
        batch = 1
        qkv_seq = 128
        kv_head = 64        # kv_head num  1
        heads = 64          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 128
        mask_dim = 4
        tor = math.sqrt(1 / embeddim)
        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 17, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 2}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_alibi=True)
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))
        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p
    def test_flash_attention_nz_case_decoder_alibi_mask_dim3(self):
        logging.debug("======================test flash attention nz decoder alibi mask dim3======================")
        batch = 3
        qkv_seq = 511
        kv_head = 8        # kv_head num
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 96
        max_seq = 768
        mask_dim = 3
        tor = math.sqrt(1 / embeddim)
        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 18, "qSeqLen": [1] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 2}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_mask=False, is_decoder=True, is_alibi=True)
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p_and_910a
    def test_flash_attention_encoder_alibi_dim4(self):
        logging.debug("======================test flash attention nz encoder alibi mask dim4======================")
        batch = 2
        qkv_seq = 257
        kv_head = 16        # kv_head num
        heads = 16          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 1024
        mask_dim = 4
        tor = math.sqrt(1 / embeddim)

        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 17, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 2}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_alibi=True)
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p_and_910a
    def test_flash_attention_encoder_alibi_dim3(self):
        logging.debug("======================test flash attention nz encoder alibi mask dim3======================")
        batch = 8
        qkv_seq = 15
        kv_head = 8        # kv_head num
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 1024
        mask_dim = 3
        tor = math.sqrt(1 / embeddim)

        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 17, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 2}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_alibi=True)
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p_and_910a
    def test_flash_attention_case_no_cache_alibi_dim4(self):
        logging.debug("======================test flash attention nz encoder no cache alibi mask dim4======================")
        batch = 5
        qkv_seq = 511
        kv_head = 8       # kv_head num
        heads = 8         # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 80
        tor = math.sqrt(1 / embeddim)
        mask_dim = 4

        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 2005, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 2}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_alibi=True)
        self.calc_data_pa((batch, qkv_seq, heads, kv_head,
                          embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), self.layer_id, torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p_and_910a
    def test_flash_attention_nz_case_no_cache_alibi_dim4(self):
        logging.debug("======================test flash attention nz encoder no cache alibi mask dim4======================")
        batch = 5
        qkv_seq = 511
        kv_head = 8       # kv_head num
        heads = 8         # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 80
        tor = math.sqrt(1 / embeddim)
        mask_dim = 3

        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 2005, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 2}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_alibi=True)
        self.calc_data_pa((batch, qkv_seq, heads, kv_head,
                          embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), self.layer_id, torch.tensor(self.mask).half(), torch.tensor(self.alibi_slopes).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p_and_910a
    def test_flash_attention_case_encoder_no_cache_norm_mask_dim3(self):
        logging.debug("======================test flash attention nz encoder no cache norm mask dim3======================")
        batch = 2
        qkv_seq = 419
        kv_head = 2        # kv_head num
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 4096
        tor = math.sqrt(1 / embeddim)
        mask_dim = 3

        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 2005, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params()
        self.calc_data_pa((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor([]).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p_and_910a
    def test_flash_attention_case_encoder_no_cache_norm_mask_dim2(self):
        logging.debug("======================test flash attention nz encoder no cache norm mask dim2======================")
        batch = 2
        qkv_seq = 1024
        kv_head = 4        # kv_head num
        heads = 4          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        tor = math.sqrt(1 / embeddim)
        mask_dim = 2

        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 2005, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 1, "isTriuMask": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params()
        self.calc_data_pa((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor([]).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p_and_910a
    def test_flash_attention_case_encoder_no_cache_norm_mask_long_seq(self):
        logging.debug("======================test flash attention nz encoder no cache norm mask long seq======================")
        batch = 2
        qkv_seq = 2048
        kv_head = 4        # kv_head num
        heads = 4          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        tor = math.sqrt(1 / embeddim)
        mask_dim = 2

        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 2005, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 1, "isTriuMask": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params()
        self.calc_data_pa((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))
        mask = torch.ones((128, 128), dtype=torch.float16)
        mask = torch.triu(mask, diagonal=1)
        mask *= -10000.0
        mask = mask.to(torch.float16)
        mask = self.convert_nd_to_nz(mask)
        self.mask = mask.reshape(1, 8, 128, 16)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor([]).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p_and_910a
    def test_flash_attention_case_no_cache_mask_none(self):
        logging.debug("======================test flash attention nz encoder no cache no mask======================")
        batch = 1
        qkv_seq = 419
        kv_head = 16       # kv_head num
        heads = 16         # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 80
        tor = math.sqrt(1 / embeddim)
        mask_dim = 0

        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 2005, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_alibi=True)
        self.calc_data_pa((batch, qkv_seq, heads, kv_head,
                          embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), self.layer_id, torch.tensor([]).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])
    
    @op_test.only_310p_and_910a
    def flash_attention_encoder(self, param_lists):
        OP_NAME = "UnpadFlashAttentionNzOperation"
        input_formats = [
            self.format_nz,
            self.format_nz,
            self.format_nz,
            self.format_nd,
            self.format_nz,
            self.format_nd,
            self.format_nd,
        ]
        output_formats = [self.format_nz]
        (
            batch,
            qkv_seq,
            kv_head,
            heads,
            embeddim,
            max_seq,
            mask_dim,
            tor,
            maskType,
            isTriuMask,
            type,
            is_alibi,
            is_alibi_128,
            is_alibi_256,
            is_sqrt,
            is_logn_scale
        ) = (
            param_lists["batch"],
            param_lists["qkv_seq"],
            param_lists["kv_head"],
            param_lists["heads"],
            param_lists["embeddim"],
            param_lists["max_seq"],
            param_lists["mask_dim"],
            param_lists["tor"],
            param_lists["maskType"],
            param_lists["isTriuMask"],
            param_lists["type"],
            param_lists["is_alibi"],
            param_lists["is_alibi_128"],
            param_lists["is_alibi_256"],
            param_lists["is_sqrt"],
            param_lists["is_logn_scale"]
        )
        OP_PARAM = {
            "type": type,
            "qSeqLen": [qkv_seq] * batch,
            "kvSeqLen": [qkv_seq] * batch,
            "headSize": heads,
            "tor": tor,
            "kvHead": kv_head,
            "maskType": maskType,
            "isTriuMask": isTriuMask,
            "isAlibiMaskSqrt": is_sqrt,
            "scaleType": 1 if is_logn_scale else 0,
        }
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(input_formats)
        self.set_output_formats(output_formats)

        self.set_data_params(
            is_alibi=is_alibi, is_alibi_128=is_alibi_128, is_alibi_256=is_alibi_256, is_sqrt=is_sqrt, is_logn_scale=is_logn_scale
        )
        if param_lists["is_cache"]:
            self.calc_data((batch, qkv_seq, heads, kv_head, embeddim, max_seq, mask_dim))
        else:
            self.calc_data_pa((batch, qkv_seq, heads, kv_head, embeddim, max_seq, mask_dim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")
        attention_out = np.zeros_like(self.q)
        return self.execute(
            [
                torch.tensor(self.q).half(),
                torch.tensor(self.k).half(),
                torch.tensor(self.v).half(),
                torch.tensor(self.layer_id).int(),
                torch.tensor(self.mask).half(),
                torch.tensor(self.alibi_slopes).half(),
                torch.tensor(self.logn_array).half(),
            ],
            [torch.tensor(attention_out).half()],
        )

    def test_flash_attention_encoder_alibi_test1(self):
        param_lists = {}
        param_lists["batch"] = 4
        param_lists["qkv_seq"] = 512
        param_lists["kv_head"] = 17
        param_lists["heads"] = 17
        param_lists["embeddim"] = 128
        param_lists["max_seq"] = 1024
        param_lists["mask_dim"] = 4
        param_lists["tor"] = math.sqrt(1 / param_lists["embeddim"])
        param_lists["maskType"] = 2
        param_lists["isTriuMask"] = 1
        param_lists["is_cache"] = True
        param_lists["type"] = 7
        param_lists["is_alibi"] = True
        param_lists["is_alibi_128"] = False
        param_lists["is_alibi_256"] = True
        param_lists["is_sqrt"] = False
        param_lists["is_logn_scale"] = False
        logging.debug("======================test flash attention encoder alibi_256 test1======================")
        self.flash_attention_encoder(param_lists)

    def test_flash_attention_encoder_alibi_test2(self):
        param_lists = {}
        param_lists["batch"] = 1
        param_lists["qkv_seq"] = 63
        param_lists["kv_head"] = 1
        param_lists["heads"] = 8
        param_lists["embeddim"] = 128
        param_lists["max_seq"] = 2048
        param_lists["mask_dim"] = 4
        param_lists["tor"] = math.sqrt(1 / param_lists["embeddim"])
        param_lists["maskType"] = 2
        param_lists["isTriuMask"] = 1
        param_lists["is_cache"] = True
        param_lists["type"] = 7
        param_lists["is_alibi"] = True
        param_lists["is_alibi_128"] = False
        param_lists["is_alibi_256"] = True
        param_lists["is_sqrt"] = True
        param_lists["is_logn_scale"] = False
        param_lists["precType"] = 1
        logging.debug("======================test flash attention encoder alibi_256_is_sqrt test2======================")
        self.flash_attention_encoder(param_lists)

    def test_flash_attention_encoder_alibi_test3(self):
        param_lists = {}
        param_lists["batch"] = 4
        param_lists["qkv_seq"] = 512
        param_lists["kv_head"] = 17
        param_lists["heads"] = 17
        param_lists["embeddim"] = 128
        param_lists["max_seq"] = 1024
        param_lists["mask_dim"] = 4
        param_lists["tor"] = math.sqrt(1 / param_lists["embeddim"])
        param_lists["maskType"] = 2
        param_lists["isTriuMask"] = 1
        param_lists["is_cache"] = True
        param_lists["type"] = 7
        param_lists["is_alibi"] = True
        param_lists["is_alibi_128"] = True
        param_lists["is_alibi_256"] = False
        param_lists["is_sqrt"] = False
        param_lists["is_logn_scale"] = False
        logging.debug("======================test flash attention encoder alibi_128 test3======================")
        self.flash_attention_encoder(param_lists)

    def test_flash_attention_encoder_alibi_no_cache_test1(self):
        param_lists = {}
        param_lists["batch"] = 4
        param_lists["qkv_seq"] = 512
        param_lists["kv_head"] = 17
        param_lists["heads"] = 17
        param_lists["embeddim"] = 128
        param_lists["max_seq"] = 1024
        param_lists["mask_dim"] = 4
        param_lists["tor"] = math.sqrt(1 / param_lists["embeddim"])
        param_lists["maskType"] = 2
        param_lists["isTriuMask"] = 1
        param_lists["is_cache"] = False
        param_lists["type"] = 2005
        param_lists["is_alibi"] = True
        param_lists["is_alibi_128"] = False
        param_lists["is_alibi_256"] = True
        param_lists["is_sqrt"] = False
        param_lists["is_logn_scale"] = False
        logging.debug("======================test flash attention encoder alibi_256 no cache test1======================")
        self.flash_attention_encoder(param_lists)

    def test_flash_attention_encoder_alibi_no_cache_test2(self):
        param_lists = {}
        param_lists["batch"] = 7
        param_lists["qkv_seq"] = 512
        param_lists["kv_head"] = 17
        param_lists["heads"] = 17
        param_lists["embeddim"] = 128
        param_lists["max_seq"] = 2048
        param_lists["mask_dim"] = 4
        param_lists["tor"] = math.sqrt(1 / param_lists["embeddim"])
        param_lists["maskType"] = 2
        param_lists["isTriuMask"] = 1
        param_lists["is_cache"] = False
        param_lists["type"] = 2005
        param_lists["is_alibi"] = True
        param_lists["is_alibi_128"] = False
        param_lists["is_alibi_256"] = True
        param_lists["is_sqrt"] = True
        param_lists["is_logn_scale"] = False
        logging.debug("======================test flash attention encoder alibi_256_is_sqrt no cache test2======================")
        self.flash_attention_encoder(param_lists)

    def test_flash_attention_encoder_alibi_no_cache_test3(self):
        param_lists = {}
        param_lists["batch"] = 7
        param_lists["qkv_seq"] = 512
        param_lists["kv_head"] = 1
        param_lists["heads"] = 17
        param_lists["embeddim"] = 128
        param_lists["max_seq"] = 2048
        param_lists["mask_dim"] = 4
        param_lists["tor"] = math.sqrt(1 / param_lists["embeddim"])
        param_lists["maskType"] = 2
        param_lists["isTriuMask"] = 1
        param_lists["is_cache"] = False
        param_lists["type"] = 2005
        param_lists["is_alibi"] = True
        param_lists["is_alibi_128"] = True
        param_lists["is_alibi_256"] = False
        param_lists["is_sqrt"] = False
        param_lists["is_logn_scale"] = False
        logging.debug("======================test flash attention encoder alibi_128 no cache test3======================")
        self.flash_attention_encoder(param_lists)

    def test_flash_attention_encoder_norm_mask_logn_no_cache_dim2(self):
        param_lists = {}
        param_lists["batch"] = 7
        param_lists["qkv_seq"] = 513
        param_lists["kv_head"] = 32 # kv head for qwen model
        param_lists["heads"] = 32
        param_lists["embeddim"] = 128
        param_lists["max_seq"] = 2048
        param_lists["mask_dim"] = 2
        param_lists["tor"] = math.sqrt(1 / param_lists["embeddim"])
        param_lists["maskType"] = 1
        param_lists["isTriuMask"] = 1
        param_lists["is_cache"] = False
        param_lists["type"] = 2005
        param_lists["is_alibi"] = False
        param_lists["is_alibi_128"] = False
        param_lists["is_alibi_256"] = False
        param_lists["is_sqrt"] = False
        param_lists["is_logn_scale"] = True
        logging.info("======================test flash attention encoder logn no cache test1======================")
        self.flash_attention_encoder(param_lists)

    @op_test.only_310p
    def test_flash_attention_case_encoder_swa_mask(self):
        logging.debug("======================test flash attention nz encoder swa mask======================")
        batch = 1
        qkv_seq = 1024
        kv_head = 40        # kv_head num
        heads = 40          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 10240
        tor = math.sqrt(1 / embeddim)
        mask_dim = 2
        window_size = 600

        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 17, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "windowSize": window_size, "maskType": 4, "isTriuMask": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_swa = True)
        self.calc_data((batch, qkv_seq, heads, kv_head,
                    embeddim, max_seq, mask_dim), window_size = window_size)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p
    def test_flash_attention_case_encoder_swa_compress(self):
        logging.debug("======================test flash attention nz encoder swa mask compress======================")
        batch = 2
        qkv_seq = 2047
        kv_head = 8        # kv_head num
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 4096
        tor = math.sqrt(1 / embeddim)
        mask_dim = 2
        window_size = 576

        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 17, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "windowSize": window_size, "maskType": 5, "isTriuMask": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_swa_compress = True)
        self.calc_data((batch, qkv_seq, heads, kv_head,
                    embeddim, max_seq, mask_dim), window_size = window_size)

        swa_block_size = (16384 // embeddim // 16 * 16) if embeddim > 128 else 128
        if 0 < window_size <= 3 * swa_block_size:
            mask = torch.ones((1, 512, 512), dtype=torch.float16)
            mask_triu = torch.triu(mask, diagonal=1)
            mask_trid = torch.tril(mask, diagonal=(-window_size))
            mask = mask_triu + mask_trid
            mask *= -10000.0
        else:
            mask = torch.ones((1, 512, 512), dtype=torch.float16)
            mask_triu = torch.triu(mask, diagonal=1)
            mask_trid = torch.tril(mask, diagonal=(-(window_size % swa_block_size + 2 * swa_block_size)))
            mask = mask_triu + mask_trid
            mask *= -10000.0
        mask = self.convert_nd_to_nz(mask)
        self.mask = mask.reshape(1, 32, 512, 16)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p
    def test_flash_attention_case_encoder_no_cache_swa_mask(self):
        logging.debug("======================test flash attention nz encoder no cache swa_mask======================")
        batch = 2
        qkv_seq = 2048
        kv_head = 8        # kv_head num
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        tor = math.sqrt(1 / embeddim)
        mask_dim = 2
        window_size = 400

        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 2005, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "windowSize": window_size, "maskType": 4, "isTriuMask": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_swa = True)
        self.calc_data_pa((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim), window_size = window_size)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor([]).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p
    def test_flash_attention_case_encoder_no_cache_swa_compress(self):
        logging.debug("======================test flash attention nz encoder no cache swa mask compress======================")
        batch = 2
        qkv_seq = 1000
        kv_head = 32        # kv_head num
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 1000
        tor = math.sqrt(1 / embeddim)
        mask_dim = 2
        window_size = 867

        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 2005, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "windowSize": window_size, "maskType": 5, "isTriuMask": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_swa_compress = True)
        self.calc_data_pa((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim), window_size = window_size)

        swa_block_size = (16384 // embeddim // 16 * 16) if embeddim > 128 else 128
        if 0 < window_size <= 3 * swa_block_size:
            mask = torch.ones((1, 512, 512), dtype=torch.float16)
            mask_triu = torch.triu(mask, diagonal=1)
            mask_trid = torch.tril(mask, diagonal=(-window_size))
            mask = mask_triu + mask_trid
            mask *= -10000.0
        else:
            mask = torch.ones((1, 512, 512), dtype=torch.float16)
            mask_triu = torch.triu(mask, diagonal=1)
            mask_trid = torch.tril(mask, diagonal=(-(window_size % swa_block_size + 2 * swa_block_size)))
            mask = mask_triu + mask_trid
            mask *= -10000.0
        mask = self.convert_nd_to_nz(mask)
        self.mask = mask.reshape(1, 32, 512, 16)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor([]).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p
    def test_flash_attention_case_decoder_mask_none_swa_norm_cache(self):
        logging.debug("======================test flash attention nz decoder no mask swa norm cache======================")
        batch = 3
        qkv_seq = 1020
        kv_head = 32        # kv_head num
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        mask_dim = 0
        window_size = 999

        tor = math.sqrt(1 / embeddim)
        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 18, "qSeqLen": [1] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "windowSize": window_size, "maskType": 4, "cacheType": 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_mask=False, is_decoder=True, is_swa=True, is_cycle_cache=False)
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim), window_size = window_size)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor([]).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p
    def test_flash_attention_case_decoder_mask_none_swa_cycle_cache(self):
        logging.debug("======================test flash attention nz decoder no mask swa cycle cache======================")
        batch = 3
        qkv_seq = 3120
        kv_head = 32        # kv_head num
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        mask_dim = 0
        window_size = 1024

        tor = math.sqrt(1 / embeddim)
        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 18, "qSeqLen": [1] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "windowSize": window_size, "maskType": 4, "cacheType": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_mask=False, is_decoder=True, is_swa=True, is_cycle_cache=True)
        if qkv_seq > window_size:
            self.calc_data((batch, window_size, heads, kv_head,
                        embeddim, window_size, mask_dim), window_size = window_size)
        else:
            self.calc_data((batch, qkv_seq, heads, kv_head,
                        embeddim, window_size, mask_dim), window_size = window_size)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor([]).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p
    def test_flash_attention_case_encoder_cube_online_softmax(self):
        batch = 1
        qkv_seq = 537

        kv_head = 8
        heads = 16
        embeddim = 128
        max_seq = 2064
        tor = math.sqrt(1 / embeddim)
        mask_dim = 0

        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 17, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, 
        "headSize": heads, "tor": tor, "kvHead": kv_head, "maskType": 0, "precType": 4}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params()
        self.calc_data((batch, qkv_seq, heads, kv_head, embeddim, max_seq, mask_dim))

        logging.info("**********input shape***********")
        logging.info(f"q shape: {self.q.shape}")
        logging.info(f"k shape: {self.k.shape}")
        logging.info(f"v shape: {self.v.shape}")
        logging.info(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        
        return  self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), 
            torch.tensor(torch.tensor(self.layer_id)).int(), torch.tensor([]).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

    @op_test.only_310p
    def test_flash_attention_case_encoder_swa_mask_cube_online_softmax(self):
        logging.debug("======================test flash attention nz encoder swa mask======================")
        batch = 1
        qkv_seq = 1024
        kv_head = 40        # kv_head num
        heads = 40          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 10240
        tor = math.sqrt(1 / embeddim)
        mask_dim = 2
        window_size = 600

        OP_NAME = "UnpadFlashAttentionNzOperation"
        OP_PARAM = {"type": 17, "qSeqLen": [qkv_seq] * batch, "kvSeqLen": [qkv_seq] * batch, "headSize": heads, "tor": tor, "kvHead": kv_head, "windowSize": window_size, "maskType": 4, "isTriuMask": 1, "precType": 4}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats(
            [self.format_nz, self.format_nz, self.format_nz, self.format_nd, self.format_nz, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])

        self.set_data_params(is_swa = True)
        self.calc_data((batch, qkv_seq, heads, kv_head,
                    embeddim, max_seq, mask_dim), window_size = window_size)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([torch.tensor(self.q).half(), torch.tensor(self.k).half(), torch.tensor(self.v).half(), torch.tensor(self.layer_id).int(), torch.tensor(self.mask).half(), torch.tensor([]).half(), torch.tensor([]).half()],
                            [torch.tensor(attention_out).half()])

if __name__ == '__main__':
    unittest.main()
