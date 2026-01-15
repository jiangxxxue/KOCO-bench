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
import sys, os
import op_test
import torch
sys.path.append('../')
sys.path.append('../..')
from precision_calcu import *

MASK_TYPE_NO_MASK = 0
MASK_TYPE_NO_HEAD = 1
MASK_TYPE_NO_BATCH = 2
MASK_TYPE_ALIBI_WITH_BATCH = 3
MASK_TYPE_ALIBI_NO_BATCH = 4
MASK_TYPE_NO_HEAD_DECODER = 5

class TestFlashAttention(op_test.OpTest):
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


    def set_data_params(self, dynamic_batch=False, batch_state=None,
                       is_mask=True, is_decoder=False, is_alibi=False, alibi_dim=4, 
                       batch = 1, kv_head = 1, heads = 1, embeddim = 128, embeddimv = 0, max_seq = 2048,
                       kv_seqLen = [], is_clamp = 0, clamp_min = 0,
                       clamp_max = 0, data_type = torch.float16, op_type = 0, mask_type = 0, 
                       no_cache = False, long_seq = False, is_triu_mask = False, is_multi_layer = False,
                       is_sqrt = False, left_align = False, fav3 = False, tor = 1):
        self.dynamic_batch = dynamic_batch
        self.batch_state = batch_state
        self.is_mask = is_mask
        self.is_decoder = is_decoder
        self.is_alibi = is_alibi
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
        self.tor = tor
        self.online = False
        if self.embeddimv == 0:
            self.embeddimv = self.embeddim
        if is_decoder:
            self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, [1] * batch)
        else:
            self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, kv_seqLen)
        self.kv_seqlen, self.kv_ntokens = self.gen_seq_len(batch, kv_seqLen)
        # gen intensor for fa kernel
        if is_multi_layer:
            self.layer_id = torch.from_numpy(np.array([1], dtype=np.int32)).to(torch.int32)
        else:
            self.layer_id = torch.from_numpy(np.array([0], dtype=np.int32)).to(torch.int32)
        self.q_max_seq = np.max(self.q_seqlen)
        self.kv_max_seq = np.max(self.kv_seqlen)
        q = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.q_ntokens, heads * self.embeddim)))

        self.q = q.to(data_type)

        self.k = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))).to(data_type)
        self.v = self.k
        if self.fav3:
            self.q_scale, self.q_offset, self.q_int8 = self.quant_per_head(self.q, heads, embeddim, (self.q_ntokens, heads * self.embeddim))
            self.k_scale, self.k_offset, self.k_int8 = self.quant_per_head(self.k, kv_head, embeddim, (self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))
            self.v_scale, self.v_offset, self.v_int8 = self.quant_per_head(self.v, kv_head, embeddim, (self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))
            self.k_scale = (self.k_scale.view(kv_head, 1) * torch.ones([kv_head, heads // kv_head])).view(-1)
            self.k_offset= (self.k_offset.view(kv_head, 1) * torch.ones([kv_head, heads // kv_head])).view(-1)
            self.v_scale = (self.v_scale.view(kv_head, 1) * torch.ones([kv_head, heads // kv_head])).view(-1)
            self.v_offset= (self.v_offset.view(kv_head, 1) * torch.ones([kv_head, heads // kv_head])).view(-1)
            self.offline_scale = self.q_scale
        self.gen_mask(batch, heads, data_type, mask_type)

        logging.debug("**********data gen shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

    def quant_per_head(self, data, heads, embeddim, shape):
        temp = data.view(-1, heads, self.embeddim)
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
        # slopes = torch.ones(n_heads)
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
        # return torch.ones(n)
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
    
    def gen_mask(self, batch, heads, data_type, mask_type):
        import random
        q_max_seq = self.max_seq
        kv_max_seq = self.max_seq
        mask_type_dict = {
            # 四维的alibi mask
            MASK_TYPE_ALIBI_WITH_BATCH : ((batch, heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :, :q_s, :kv_s]))),
            # 三维的alibi mask
            MASK_TYPE_ALIBI_NO_BATCH : ((heads, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
            MASK_TYPE_NO_HEAD : ((batch, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            MASK_TYPE_NO_HEAD_DECODER : ((batch, 1, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            MASK_TYPE_NO_BATCH : ((1, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[:, :q_s, :kv_s]))),
            # 不加mask
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
            if self.is_alibi or self.long_seq:
                select_zero = False
            else:
                select_zero = True
        elif data_type == torch.bfloat16:
            if self.is_alibi:
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
        if self.is_alibi:
            self.alibi_bias = self.get_alibi_bias(heads, self.max_seq)
            mask += self.alibi_bias.numpy()
        if select_zero:
            mask.flat[zero_indice] = 0
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
                    if self.fav3:
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
                    if self.fav3:
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
            q_s = q_seqlen[idx]
            kv_s = kv_seqlen[idx]
            q_slice = q[q_offset:q_offset + q_s][:]
            q_slice = q_slice.view(q_s, heads, embed)
            q_slice = torch.permute(q_slice, (1, 0, 2))
            k_slice = k[layer_id][idx][:kv_s][:]
            k_slice = k_slice.view(kv_s, kv_head, embed)
            k_slice_t = torch.permute(k_slice, (1, 2, 0))   # get K^T
            v_slice = v[layer_id][idx][:kv_s][:]
            v_slice = v_slice.view(kv_s, kv_head, embed)
            v_slice = v_slice[:,:,:embedv]
            v_slice = torch.permute(v_slice, (1, 0, 2))

            if self.fav3:
                score = self.group_mm_torch(heads, kv_head, q_slice, k_slice_t, torch.int32)
            else:
                score = self.group_mm_torch(heads, kv_head, q_slice, k_slice_t)
            if self.fav3:
                # score:[heads,m,n]
                score = score.to(torch.float32)
                s_scale = self.q_scale * self.k_scale
                score = s_scale.view(heads, 1, 1) * score

            if s is None:
                s = score.view([-1, ])
            else:
                s = torch.cat((s, score.view([-1, ])), 0)

            scale = 1
            if not self.is_multi_layer:
                # 当前scale和tor保持一致，模型侧可能传入scale = np.float32(layer_id + 1)
                scale = np.float32(layer_id + 1)
            score = score * scale

            if self.fav3:
                score = score * torch.tensor(self.tor, dtype=torch.float16)
            else:
                score *= self.tor

            if self.is_clamp == 1:
                clamp_min_brc = np.ones((score.shape)) * self.clamp_min
                clamp_max_brc = np.ones((score.shape)) * self.clamp_max
                score = np.float16(np.maximum(score, clamp_min_brc))
                score = torch.from_numpy(np.float16(np.minimum(score, clamp_max_brc)))
            if is_mask:
                score = score + self.mask_info[1](self.mask, idx, q_s, kv_s) * self.post_mask_coff
            
            s_qk = score
            s_qk_true = score.to(torch.float32)
            score = score.numpy().astype(np.float32)
            if self.fav3:
                ans = self.online_softmax(s_qk, q_s, v_slice, heads, kv_head, embedv, online, torch.float16)
                if ans_concat is None:
                    ans_concat = ans
                else:
                    ans_concat = torch.cat((ans_concat, ans), 0)

                ans_true = self.online_softmax(s_qk_true, q_s, v_slice, heads, kv_head, embedv, online, torch.float32)
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
            
            p_high = (score_exp / score_sum.reshape((heads, q_s, 1)))
            p_high =  torch.from_numpy(p_high)
            p = p_high.to(torch.bfloat16)
            o = self.group_mm_torch(heads, kv_head, p, v_slice)
            o_high = self.group_mm_torch(heads, kv_head, p_high, v_slice)
            o = o.view(heads, q_s, embedv)
            o_high = o_high.view(heads, q_s, embedv)
            o = torch.permute(o, (1, 0, 2)).contiguous()
            o_high = torch.permute(o_high, (1, 0, 2)).contiguous()
            if out is None:
                out = o
                out_high = o_high
            else:
                out = torch.cat((out, o), 0)
                out_high = torch.cat((out_high, o_high), 0)

            q_offset += q_s
            k_offset += max_seq
            v_offset += max_seq
        
        # golden data
        if self.fav3:
            ans_concat = ans_concat.view(q_ntokens, heads * embedv)
            ans_concat_true = ans_concat_true.view(q_ntokens, heads * embedv)
            self.golden_out = ans_concat.to(self.data_type)
            self.golden_out_high = ans_concat_true.to(torch.float32)
        else:
            out = out.view(q_ntokens, heads * embedv)
            out_high = out_high.view(q_ntokens, heads * embedv)
            self.golden_out = out.to(self.data_type)
            self.golden_out_high = out_high.to(torch.float32)

        if self.no_cache:
            self.k = self.close_pack(self.k.to(torch.float32), kv_seqlen).to(self.data_type)
            self.v = self.close_pack(self.v.to(torch.float32), kv_seqlen).to(self.data_type)
            if self.fav3:
                self.k_int8 = self.close_pack(self.k_int8.to(torch.float32), kv_seqlen).to(torch.int8)
                self.v_int8 = self.close_pack(self.v_int8.to(torch.float32), kv_seqlen).to(torch.int8)
        if self.long_seq:
            self.max_seq = 128
            self.gen_mask(self.batch, self.heads, self.data_type, self.mask_type)

    def gen_seq_len(self, batch, seq_len):
        ntokens = sum(seq_len)
        return seq_len, ntokens

    def compare_output_data(self, out, golden, ratios):
        error_count = 0
        strict_error_count = 0
        alibi_error_count = 0
        fp16_min_normal = 1.0 / (1 << 14)
        len = out.shape[0] * out.shape[1]
        diff = torch.abs(golden - out)
        max_diff = diff.max().item()
        logging.info(f"maxDiff {max_diff}")
        if self.is_alibi:
            alibi_limit_error = torch.maximum(torch.abs(golden * ratios[4]), torch.tensor(ratios[5]))
            alibi_error_count = torch.gt(diff, alibi_limit_error).sum().item()
            logging.info("5/1000 Accuracy is %f",  1 - float(alibi_error_count) / len)
            return (float(alibi_error_count) / len) <= ratios[4]
        else:
            limit_error = torch.maximum(torch.abs(golden * ratios[0]), torch.tensor(ratios[1]))
            strict_limit_error = torch.maximum(torch.abs(golden * ratios[2]), torch.tensor(ratios[3]))
            error_count = torch.gt(diff, limit_error).sum().item()
            strict_error_count = torch.gt(diff, strict_limit_error).sum().item()
            logging.info("1/1000 Accuracy is %f",  1 - float(error_count) / len)
            logging.info("3/1000 Accuracy is %f",  1 - float(strict_error_count) / len)
            if self.data_type == torch.bfloat16 or not self.is_decoder:
                return (float(strict_error_count) / len) <= ratios[2]
            else:
                return (float(error_count) / len) <= ratios[0]

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
        golden_out = torch.tensor(self.golden_out).half()
        return [golden_out]

    def golden_compare(self, out_tensors, golden_tensors):
        result_double = compare_cv(self.golden_out_high, golden_tensors[0], out_tensors[0])
        result_old = self.compare_output_data(out_tensors[0].half(), golden_tensors[0].half(), [0.001, 0.001, 0.003, 0.003, 0.005, 0.005])
        return (result_double or result_old)

    @op_test.only_910b
    def test_flash_attention_mla_fp16(self):
        # unpad encoder
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 1          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 512
        max_seq = 2048
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [2048]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2008, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV,"kvHead": kv_head,"headSize": heads, "tor": tor}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 9)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch, 
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads, 
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH, no_cache = True, tor = tor)
        self.gen_out_tensor()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")
        self.layer_id = torch.tensor([], dtype=torch.int32)
        attention_out = np.zeros_like(self.golden_out)
        return self.execute([self.q, self.k, self.layer_id, self.mask.to(data_type),
                             torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.int),
                             torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.int),
                             torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_mla_fp16_high_precision_nomask(self):
        # unpad encoder
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 1          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 512
        max_seq = 2048
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [max_seq] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type":2010, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV,"kvHead": kv_head,"headSize": heads, "tor": tor, "maskType": 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 9)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch, 
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads, 
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_MASK, is_mask = False, no_cache = True, tor = tor)
        self.gen_out_tensor()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")
        self.layer_id = torch.tensor([], dtype=torch.int32)
        attention_out = np.zeros_like(self.golden_out)
        return self.execute([self.q, self.k, self.layer_id, torch.tensor([], dtype=data_type),
                             torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.int),
                             torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.int),
                             torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_mla_fp16_case2(self):
        # unpad encoder
        batch = 1
        kv_head = 16        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 16          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 512
        max_seq = 2048
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [500]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2008, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV,"kvHead": kv_head,"headSize": heads, "tor": tor}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 9)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch, 
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads, 
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH, no_cache = True, tor = tor)
        self.gen_out_tensor()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")
        self.layer_id = torch.tensor([], dtype=torch.int32)
        attention_out = np.zeros_like(self.golden_out)
        return self.execute([self.q, self.k, self.layer_id, self.mask.to(data_type),
                             torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.int),
                             torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.int),
                             torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_mla_bf16(self):
        # unpad encoder
        batch = 16
        kv_head = 2        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 2          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 512
        max_seq = 2048
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [500, 1024, 1500, 2048] * 4
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2010, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV, "kvHead": kv_head, "headSize": heads, "tor": tor}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 9)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch, 
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads, 
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH,no_cache = True, tor = tor)
        self.gen_out_tensor()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")
        self.layer_id = torch.tensor([], dtype=torch.int32)
        attention_out = np.zeros_like(self.golden_out.to(torch.float16))
        return self.execute([self.q, self.k, self.layer_id, self.mask.to(data_type),
                             torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.int),
                             torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.int),
                             torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=data_type)])

    @op_test.only_910b
    def test_flash_attention_mla_bf16_case2(self):
        # unpad encoder
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 512
        max_seq = 2048
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [500]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2010, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV, "kvHead": kv_head,"headSize": heads, "tor": tor}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 9)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch, 
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads, 
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH, no_cache = True, tor = tor)
        self.gen_out_tensor()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))
        self.layer_id = torch.tensor([], dtype=torch.int32)
        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.golden_out.to(torch.float16))
        return self.execute([self.q, self.k, self.layer_id, self.mask.to(data_type),
                             torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.int),
                             torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.int),
                             torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=data_type)])

    def test_flash_attention_case_fa_encoder_nocache_bf16_fav3_int8_online(self):
        self.set_support_910b_only()
        # unpad encoder
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 512
        max_seq = 2048
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [500]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        fav3=True
        online = True
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2010, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV, "kvHead": kv_head,"headSize": heads,
                    "tor": tor, "quantType": 3 if fav3 else 0, "outDataType": 27}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 9)
        self.set_output_formats([self.format_nd])

        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch, 
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads, 
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, no_cache = True, fav3=fav3, tor = tor)
        self.gen_out_tensor(online)
        attention_out = np.zeros_like(self.golden_out.to(torch.float16))
        if not self.fav3:
            return self.execute([self.q, self.k, torch.tensor([], dtype=torch.int), self.mask.to(data_type),
                            torch.tensor([], dtype=torch.float),torch.tensor([], dtype=torch.int),
                            torch.tensor([], dtype=torch.float),torch.tensor([], dtype=torch.int),torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])
        else:
            return self.execute([self.q_int8, self.k_int8, torch.tensor([], dtype=torch.int), self.mask.to(data_type),
                            torch.tensor(self.q_scale * self.k_scale, dtype=torch.float),
                            torch.tensor([], dtype=torch.int),
                            torch.tensor(self.v_scale, dtype=torch.float),
                            torch.tensor([], dtype=torch.int),
                            torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])
    
    def test_flash_attention_case_fa_encoder_nocache_bf16_fav3_int8_offline(self):
        self.set_support_910b_only()
        # unpad encoder
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 512
        max_seq = 2048
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [500]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        fav3=True
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2010, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV, "kvHead": kv_head,"headSize": heads,
                    "tor": tor, "quantType": 2 if fav3 else 0, "outDataType": 27}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 9)
        self.set_output_formats([self.format_nd])

        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch, 
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads, 
                             embeddim = embeddim, embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, no_cache = True, fav3 = fav3, tor = tor)
        self.gen_out_tensor()
        attention_out = np.zeros_like(self.golden_out.to(torch.float16))
        if not self.fav3:
            return self.execute([self.q, self.k, torch.tensor([], dtype=torch.int), self.mask.to(data_type),
                            torch.tensor([], dtype=torch.float),torch.tensor([], dtype=torch.int),
                            torch.tensor([], dtype=torch.float),torch.tensor([], dtype=torch.int),torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])
        else:
            return self.execute([self.q_int8, self.k_int8, torch.tensor([], dtype=torch.int), self.mask.to(data_type),
                            torch.tensor(self.q_scale * self.k_scale, dtype=torch.float),
                            torch.tensor([], dtype=torch.int),
                            torch.tensor(self.v_scale, dtype=torch.float),
                            torch.tensor([], dtype=torch.int),
                            torch.tensor(self.q_scale, dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])
    
    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_nocache_fp16_fav3_int8_online(self):
        return      # TODO: CI pytorch版本不支持，待升级版本
        # unpad encoder
        batch = 1
        kv_head = 16        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 16          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 512
        max_seq = 2048
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [500]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        fav3=True
        online = True
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2008, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV, "kvHead": kv_head,"headSize": heads,
                    "tor": tor, "quantType": 3 if fav3 else 0, "outDataType": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 9)
        self.set_output_formats([self.format_nd])

        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch, 
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads, 
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH, no_cache = True, fav3=fav3, tor = tor)
        self.gen_out_tensor(online)
        self.mask = np.reshape(self.mask, (max_seq, max_seq))
        attention_out = np.zeros_like(self.golden_out)
        if not self.fav3:
            return self.execute([self.q, self.k, torch.tensor([], dtype=torch.int), self.mask.to(data_type),
                            torch.tensor([], dtype=torch.float),torch.tensor([], dtype=torch.int),
                            torch.tensor([], dtype=torch.float),torch.tensor([], dtype=torch.int),torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.float16)])
        else:
            return self.execute([self.q_int8, self.k_int8, torch.tensor([], dtype=torch.int), self.mask.to(data_type),
                            torch.tensor(self.q_scale * self.k_scale, dtype=torch.float),
                            torch.tensor([], dtype=torch.int),
                            torch.tensor(self.v_scale, dtype=torch.float),
                            torch.tensor([], dtype=torch.int),
                            torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.float16)])
    
    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_nocache_fp16_fav3_int8_offline(self):
        return      # TODO: CI pytorch版本不支持，待升级版本
        # unpad encoder
        batch = 1
        kv_head = 16        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 16          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 512
        max_seq = 2048
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [500]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        fav3=True
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2008, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV, "kvHead": kv_head,"headSize": heads,
                    "tor": tor, "quantType": 2 if fav3 else 0, "outDataType": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 9)
        self.set_output_formats([self.format_nd])

        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch, 
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads, 
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH, no_cache = True, fav3=fav3, tor = tor)
        self.gen_out_tensor()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))
        attention_out = np.zeros_like(self.golden_out)
        if not self.fav3:
            return self.execute([self.q, self.k, torch.tensor([], dtype=torch.int), self.mask.to(data_type),
                            torch.tensor([], dtype=torch.float),torch.tensor([], dtype=torch.int),
                            torch.tensor([], dtype=torch.float),torch.tensor([], dtype=torch.int),torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.float16)])
        else:
            return self.execute([self.q_int8, self.k_int8, torch.tensor([], dtype=torch.int), self.mask.to(data_type),
                            torch.tensor(self.q_scale * self.k_scale, dtype=torch.float),
                            torch.tensor([], dtype=torch.int),
                            torch.tensor(self.v_scale, dtype=torch.float),
                            torch.tensor([], dtype=torch.int),
                            torch.tensor(self.q_scale, dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.float16)])
    
    @op_test.only_910b
    def test_flash_attention_mla_bf16_case3(self):
        # unpad encoder
        batch = 2
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 512
        max_seq = 2048
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [500, 800]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2010, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV, "kvHead": kv_head,"headSize": heads, "tor": tor}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 9)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch, 
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads, 
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH, tor = tor)
        self.gen_out_tensor()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))
        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")
        attention_out = np.zeros_like(self.golden_out.to(torch.float16))
        return self.execute([self.q, self.k, self.layer_id, self.mask.to(data_type),
                             torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.int),
                             torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.int),
                             torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=data_type)])
    
if __name__ == '__main__':
    unittest.main()
