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
sys.path.append('/opt/cl/ascend-op-common-lib/tests/pythontest')
from enum import Enum
import random
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(os.path.dirname(current_dir))
#将父级的父级目录添加到 sys.path
sys.path.append(parent_dir)
from precision_calcu import *

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
        self.q_max_seq = np.max(self.q_seqlen)
        self.kv_max_seq = np.max(self.kv_seqlen)
        q = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.q_ntokens, heads * self.embeddim)))

        self.q = q.to(data_type)
        if num_blocks is None:
            self.k = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))).to(data_type)
            self.v = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddimv))).to(data_type)
            if is_splitm:
                maxKvSeqlen = max(self.kv_seqlen)
                self.k = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, batch, maxKvSeqlen, kv_head * self.embeddim))).to(data_type)
                self.v = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, batch, maxKvSeqlen, kv_head * self.embeddimv))).to(data_type)
        else:
            # kv cache shape: (num_blocks, block_size, num_heads, head_size)
            self.k_cache = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_head, embeddim))).to(data_type)
            self.v_cache = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(num_blocks, block_size, kv_head, embeddim))).to(data_type)

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
        logging.debug("**********data gen shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

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

    def gen_swa_cmp(self, max_seq, window_size):
        swa_mask = np.ones(shape=(1, 512, 512)) * self.pre_mask_coff
        pp_n = 128 if self.embeddim <= 128 else 64
        pp_n = 128 if self.embeddim != self.embeddimv else pp_n
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
        swa_mask = torch.from_numpy(swa_mask).to(torch.float32)
        return swa_mask

    def gen_razor_fusion_mask(self, razorLen, tileQ, tileKv, textQLen, textKvLen, preTokens, nextTokens, baseM):
        np.set_printoptions(threshold=np.inf)
        
        mask_sizeQ = razorLen * tileQ + textQLen
        mask_sizeK = razorLen * tileKv + textKvLen
        print("generate razor mask:", razorLen, tileQ, tileKv, textQLen, textKvLen, preTokens, nextTokens, baseM)
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
        # golden_out = torch.tensor(self.golden_out).half()
        golden_out = torch.tensor(self.golden_out)
        return [golden_out]

    def golden_compare(self, out_tensors, golden_tensors):
        result_single = self.compare_output_data(out_tensors[0].half(), golden_tensors[0].half(), [0.001, 0.001, 0.005, 0.005])
        if self.is_int8_flag:
            result_double = compare_cv(self.golden_out_true, golden_tensors[0], out_tensors[0])
            return (result_double or result_single)
        else:
            result_double = compare_cv(self.golden_out_true, golden_tensors[0], out_tensors[0])
            return (result_double or result_single)
    @op_test.only_910b
    def test_flash_attention_clamp_encoder_bf16_nomask_bnsd_splitm_case1(self):
        batch = 2
        isdecoder = 0       # prefill or decoder
        kv_head = 1        # kv_head num
        heads = 3          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 7497
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqLen = [7497,4112]
        kv_seqLen = [5841,4112]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        scaleType = ScaleType.SCALE_TOR.value
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": q_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "isClamp" : is_clamp, "kvHead" : kv_head,
                    "clampMin" : clamp_min, "clampMax" : clamp_max, "isTriuMask": 0, "maskType": 0,"scaleType":scaleType,"dataShapeType":1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16
        self.set_data_params(no_cache=True,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, q_seqlens = q_seqLen,kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,scaleType = scaleType, is_splitm = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_MASK, is_mask = False, tor = tor)
        self.gen_out_tensor_bnsd_splitm()
        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")
        attention_out = np.zeros_like(self.qbnsd.to(torch.float16))
        self.kbnsd = self.kbnsd[0]
        self.vbnsd = self.vbnsd[0]
        return self.execute([self.qbnsd, self.kbnsd, self.vbnsd, torch.tensor([], dtype=torch.float), torch.tensor([], dtype=data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=data_type)])
    @op_test.only_910b
    def test_flash_attention_clamp_encoder_bf16_nomask_bnsd_splitm_case2(self):
        batch = 3
        q_seqLen = [4000,4024,128]
        kv_seqLen = [5841,3957,256]
        isdecoder = 0       # prefill or decoder
        kv_head = 1        # kv_head num
        heads = 1         # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 5841
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": q_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "isClamp" : is_clamp, "kvHead" : kv_head,
                    "clampMin" : clamp_min, "clampMax" : clamp_max, "isTriuMask": 0, "maskType": 0,"dataShapeType":1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16
        self.set_data_params(no_cache=True,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, q_seqlens = q_seqLen,kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False, is_splitm = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_MASK, is_mask = False, tor = tor)
        self.gen_out_tensor_bnsd_splitm()
        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")
        attention_out = np.zeros_like(self.qbnsd.to(torch.float16))
        self.kbnsd = self.kbnsd[0]
        self.vbnsd = self.vbnsd[0]
        return self.execute([self.qbnsd, self.kbnsd, self.vbnsd, torch.tensor([], dtype=torch.float), torch.tensor([], dtype=data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=data_type)])
    @op_test.only_910b
    def test_flash_attention_case_encoder_bf16_tor(self):
        batch = 2
        kv_head = 8        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8
        embeddim = 128
        max_seq = 4000
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [3031,2048]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        scaleType = ScaleType.SCALE_TOR.value
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 1, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "kvHead":kv_head,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 1, "isTriuMask": 1,"scaleType":scaleType}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])

        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, long_seq = True, scaleType = scaleType, tor = tor)
        self.gen_out_tensor()
        self.mask = self.mask.view(batch, 128, 128)[0, :, :]
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                            torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                             [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_swa_decoder(self):
        self.set_support_910b_only()
        np.random.seed(1)
        np.random.seed(1)
        np.random.seed(1)
        np.random.seed(1)
        np.random.seed(1)
        # decoder
        batch = 1
        kv_head = 1    # kv_head num
        isdecoder = 1       # prefill or decoder
        heads = 1       # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 3000
        # tor = 1.0 / math.sqrt(1.0 * embeddim)
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [128*10] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        window_size = 128*4+96
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 12, "qSeqLen":[1] * batch, "kvSeqLen": kv_seqLen, "headSize": heads,
                    "kvHead": kv_head, "windowSize":window_size,
                    "tor": tor, "maskType": 4, "cacheType":0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch, window_size = window_size, cache_type=0,
                            is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                            embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                            is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                            data_type = data_type, is_alibi = False, is_mask =True,
                            tor = tor,
                            op_type = OP_PARAM["type"], mask_type = MASK_TYPE_SWA_DECODER, is_multi_layer = False)
        self.gen_out_tensor()

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([self.q.to(torch.float16), self.k.to(torch.float16), self.v.to(torch.float16), self.layer_id,
                             torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case_swa_decoder_bf16(self):
        self.set_support_910b_only()
        # decoder
        batch = 1
        kv_head = 40    # kv_head num
        isdecoder = 1       # prefill or decoder
        heads = 40       # llama7b  hidden_size 4096
        embeddim = 256
        max_seq = 3000
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [128*10] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        window_size = 128*4+69
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 12, "qSeqLen":[1] * batch, "kvSeqLen": kv_seqLen, "headSize": heads,
                    "kvHead": kv_head, "windowSize":window_size,
                    "tor": tor, "maskType": 4, "cacheType":1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch, window_size = window_size, cache_type=1,
                            is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                            embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                            is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                            data_type = data_type, is_alibi = False, is_mask =True, tor = tor,
                            op_type = OP_PARAM["type"], mask_type = MASK_TYPE_SWA_DECODER, is_multi_layer = False)
        self.gen_out_tensor()

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q.to(torch.float16), self.k.to(torch.float16),
                             self.v.to(torch.float16), self.layer_id,
                             torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case_swa_encoder(self):
        self.set_support_910b_only()
        # unpad dynamic batch encoder
        batch = 1
        kv_head = 1 # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 1        # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 128*20
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        window_size = 128*7+89
        dynamic_batch = False
        # batchRunStatus = [1] * batch
        kv_seqLen = [128*20] * batch

        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "windowSize": window_size, "maskType": 4}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                            is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                            embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                            is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min, tor = tor,
                            data_type = data_type, is_alibi = False, window_size = window_size, no_cache=True,
                            op_type = OP_PARAM["type"], mask_type = MASK_TYPE_SWA)
        # self.golden_out = np.zeros_like(self.q)
        self.gen_out_tensor()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))

        logging.debug("**********input shape***********")
        logging.info(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([self.q, self.k, self.v, self.layer_id,
                             self.mask.to(data_type),
                             torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case_swa_compress(self):
        self.set_support_910b_only()
        # unpad dynamic batch encoder
        batch = 1
        kv_head = 40       # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 40        # llama7b  hidden_size 4096
        embeddim = 256
        max_seq = 4096
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        window_size = 128*3+58
        dynamic_batch = False
        # batchRunStatus = [1] * batch
        kv_seqLen = [128*7*1] * batch

        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "windowSize": window_size, "maskType": 5}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch, is_compress=True,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen, tor = tor,
                             data_type = data_type, is_alibi = False, window_size = window_size, no_cache=True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_SWA)
        self.gen_out_tensor()

        if self.is_compress:
            self.mask = self.gen_swa_cmp(max_seq, window_size)
            self.mask = np.reshape(self.mask, (512, 512))

        logging.debug("**********input shape***********")
        logging.info(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([self.q, self.k, self.v, self.layer_id,
                             self.mask.to(data_type),
                             torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case_swa_bf16_encoder(self):
        self.set_support_910b_only()
        # unpad dynamic batch encoder
        batch = 1
        kv_head = 1 # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 4        # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 128*20
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        window_size = 128*7-85
        dynamic_batch = False
        # batchRunStatus = [1] * batch
        kv_seqLen = [128*15+78] * batch

        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "kvHead": kv_head,
                    "windowSize": window_size, "maskType": 4}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                            is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                            embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                            is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min, tor = tor,
                            data_type = data_type, is_alibi = False, window_size = window_size, no_cache=True,
                            op_type = OP_PARAM["type"], mask_type = MASK_TYPE_SWA)
        # self.golden_out = np.zeros_like(self.q)
        self.gen_out_tensor()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))

        logging.debug("**********input shape***********")
        logging.info(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, self.layer_id,
                             self.mask.to(data_type),
                             torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_swa_compress_bf16(self):
        self.set_support_910b_only()
        # unpad dynamic batch encoder
        batch = 1
        kv_head = 1       # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 1        # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 4096
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        window_size = 128*8-96
        dynamic_batch = False
        # batchRunStatus = [1] * batch
        kv_seqLen = [128*20+1] * batch

        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "kvHead": kv_head,
                    "windowSize": window_size, "maskType": 5}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch, is_compress=True,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen, tor = tor,
                             data_type = data_type, is_alibi = False, window_size = window_size, no_cache=True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_SWA)
        self.gen_out_tensor()

        if self.is_compress:
            self.mask = self.gen_swa_cmp(max_seq, window_size)
            self.mask = np.reshape(self.mask, (512, 512))

        logging.debug("**********input shape***********")
        logging.info(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, self.layer_id,
                             self.mask.to(data_type),
                             torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])
    @op_test.only_910b
    def test_flash_attention_decoder_bnsd_case(self):
        # decoder
        batch = 2
        kv_head = 10
        isdecoder = 1
        heads = 20
        embeddim = 128
        max_seq = 256
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [256] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 12, "qSeqLen":[1] * batch, "kvSeqLen": kv_seqLen, "headSize": heads,"kvHead":kv_head,"tor": tor,
                    "dataShapeType":1 }
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16
        
        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH, tor = tor)
        self.gen_out_tensor_bnsd()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([self.q, self.kbnsd, self.vbnsd, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                        [torch.tensor(attention_out).half()])
    @op_test.only_910b
    def test_flash_attention_case_encoder_fp32_bnsd(self):
        batch = 2
        kv_head = 16        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 16
        embeddim = 128
        max_seq = 512
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [128,259]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2001, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "kvHead":kv_head,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "maskType": 0, "clampMax" : clamp_max, "isTriuMask": 1,
                    "dataShapeType":1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])

        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False, is_mask = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_MASK,
                             tor = tor)
        self.gen_out_tensor_bnsd()

        attention_out = np.zeros_like(self.qbnsd.to(torch.float16))
        return self.execute([self.qbnsd, self.kbnsd, self.vbnsd, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                            torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                             [torch.tensor(attention_out, dtype=torch.bfloat16)])
    @op_test.only_910b
    def test_flash_attention_case_encoder_fp32_no_cache_logN(self):
        batch = 2
        kv_head = 8        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 16
        embeddim = 128
        max_seq = 4000
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [3031,2048]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        scaleType = ScaleType.SCALE_LOGN_FP32.value
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2001, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "kvHead":kv_head,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 1, "isTriuMask": 1,"scaleType":scaleType}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])

        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD,
                             long_seq = True, scaleType = scaleType, tor = tor)
        self.gen_out_tensor()
        self.mask = self.mask.view(batch, 128, 128)[0, :, :]

        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                            torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), self.encoder_logN],
                             [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_encoder_fp32_no_cache_logN_nomask(self):
        batch = 2
        kv_head = 8        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 16
        embeddim = 128
        max_seq = 4000
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [3031,2048]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        scaleType = ScaleType.SCALE_LOGN_FP32.value
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2001, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "kvHead":kv_head,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 0, "scaleType":scaleType}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])

        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_MASK,
                             long_seq = True, scaleType = scaleType, tor = tor)
        self.gen_out_tensor()

        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, self.layer_id, torch.tensor([], dtype=torch.bfloat16), torch.tensor([], dtype=torch.float),
                            torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), self.encoder_logN],
                             [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_encoder_fp32_with_cache_logN(self):
        batch = 2
        kv_head = 8        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 16
        embeddim = 128
        max_seq = 4000
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [3031,2048]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        scaleType = ScaleType.SCALE_LOGN_FP32.value
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2001, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "kvHead":kv_head,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 1, "isTriuMask": 1,"scaleType":scaleType}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])

        data_type = torch.bfloat16
        
        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD,
                             long_seq = True, scaleType = scaleType, tor = tor)
        self.gen_out_tensor()
        self.mask = self.mask.view(batch, 128, 128)[0, :, :]
        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                            torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), self.encoder_logN],
                             [torch.tensor(attention_out, dtype=torch.bfloat16)])
    @op_test.only_910b
    def test_flash_attention_case_decoder_logN(self):
        # dynamic batch decoder
        batch = 2
        kv_head = 32        # kv_head num
        isdecoder = 1       # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 1024
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [1024] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        scaleType = ScaleType.SCALE_LOGN_FP32.value
        OP_PARAM = {"type": 12, "qSeqLen": [1] * batch, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
        "scaleType": scaleType}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD,
                             scaleType=scaleType, tor = tor)
        self.gen_out_tensor()

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                            torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), self.decoder_logN],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case0(self):
        self.set_support_910b_only()
        # unpad encoder
        batch = 28
        kv_head = 32        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 512
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [1, 2, 127, 505] * 7
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 1, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads,"kvHead":kv_head, "tor": tor}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16
        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH ,tor = tor)
        self.gen_out_tensor()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_encoder_bnsd_case0(self):
        # unpad encoder
        batch = 28
        kv_head = 32       # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32        # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 256
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [1, 2, 127, 255] * 7
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 1, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "kvHead":kv_head,"tor": tor,"dataShapeType":1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH, tor = tor)
        self.gen_out_tensor_bnsd()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))

        logging.debug("**********input shape***********")
        logging.debug(f"qbnsd shape: {self.qbnsd.shape}")
        logging.debug(f"kbnsd shape: {self.kbnsd.shape}")
        logging.debug(f"vbnsd shape: {self.vbnsd.shape}")
        attention_out = np.zeros_like(self.qbnsd)

        return  self.execute([self.qbnsd, self.kbnsd, self.vbnsd, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_encoder_bnsd_dynamic_batch_case0(self):
        # unpad encoder
        batch = 9
        kv_head = 2       # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 2        # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 32
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = True
        kv_seqLen = [14, 7, 0, 5, 0, 27, 0, 1, 9]
        batchRunStatus = [1, 1, 0, 1, 0, 1, 0, 1,1]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 4, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "kvHead":kv_head,"tor": tor,"dataShapeType":1,"batchRunStatus": batchRunStatus}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch, batch_state = batchRunStatus,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH, tor = tor)
        self.gen_out_tensor_bnsd()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))

        logging.debug("**********input shape***********")
        logging.debug(f"qbnsd shape: {self.qbnsd.shape}")
        logging.debug(f"kbnsd shape: {self.kbnsd.shape}")
        logging.debug(f"vbnsd shape: {self.vbnsd.shape}")
        attention_out = np.zeros_like(self.qbnsd)

        return  self.execute([self.qbnsd, self.kbnsd, self.vbnsd, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case1(self):
        # unpad dynamic batch encoder
        batch = 6
        kv_head = 32        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 600
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = True
        batchRunStatus = [1] * batch
        kv_seqLen = [1, 41, 88, 144, 323, 544]

        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 4, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "batchRunStatus": batchRunStatus}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch, batch_state = batchRunStatus,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, tor = tor)
        self.gen_out_tensor()

        self.k = np.reshape(self.k, (batch, max_seq, heads * embeddim))
        self.v = np.reshape(self.v, (batch, max_seq, heads * embeddim))
        self.mask = np.reshape(self.mask, (batch, max_seq, max_seq))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case2(self):
        # unpad dynamic batch encoder
        batch = 8
        kv_head = 8        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 1024
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = True
        batchRunStatus = [1, 1, 0, 1, 0, 1, 0, 1]
        kv_seqLen = [114, 707, 0, 1023, 0, 1, 0, 1]

        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 4, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "batchRunStatus": batchRunStatus}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch, batch_state = batchRunStatus,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, tor = tor)
        self.gen_out_tensor()

        self.k = np.reshape(self.k, (batch, max_seq, heads * embeddim))
        self.v = np.reshape(self.v, (batch, max_seq, heads * embeddim))
        self.mask = np.reshape(self.mask, (batch, max_seq, max_seq))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case3(self):
        # alibi
        batch = 2
        kv_head = 32        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 256
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [129] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 11, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_BATCH, tor = tor)
        self.gen_out_tensor()

        self.mask = np.reshape(self.mask, (batch, heads, max_seq, max_seq))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        res = self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])
        return res

    @op_test.only_910b
    def test_flash_attention_case4(self):
        # decoder
        batch = 8
        kv_head = 20        # kv_head num
        isdecoder = 1       # prefill or decoder
        heads = 20          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 256
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [256] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 12, "qSeqLen":[1] * batch, "kvSeqLen": kv_seqLen,"kvHead":kv_head,"headSize": heads, "tor": tor}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
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

        attention_out = np.zeros_like(self.q)
        self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                        [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_decoder_bnsd_case1(self):
        # decoder
        batch = 8
        kv_head = 10        # kv_head num
        isdecoder = 1       # prefill or decoder
        heads = 20         # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 256
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [256] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 12, "qSeqLen":[1] * batch, "kvSeqLen": kv_seqLen, "headSize": heads,"kvHead":kv_head,"tor": tor, "dataShapeType":1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH, tor = tor)
        self.gen_out_tensor_bnsd()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([self.q, self.kbnsd, self.vbnsd, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                        [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case_decoder_no_mask(self):
        # decoder
        batch = 8
        kv_head = 32        # kv_head num
        isdecoder = 1       # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 256
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [114] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 12, "qSeqLen":[1] * batch, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "maskType": 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False, is_mask = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD_DECODER,
                             is_multi_layer = True, tor = tor)
        self.gen_out_tensor()

        self.mask = np.reshape(self.mask, (batch, 1, max_seq))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([self.q, self.k, self.v, self.layer_id, torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case5(self):
        # dynamic batch decoder
        batch = 8
        kv_head = 32        # kv_head num
        isdecoder = 1       # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = True
        batchRunStatus = [1, 1, 0, 1, 0, 1, 0, 1]
        kv_seqLen = [114, 707, 0, 5, 0, 2047, 0, 1]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 13, "qSeqLen": [1] * batch, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "batchRunStatus": batchRunStatus}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch, batch_state = batchRunStatus,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, tor = tor)
        self.gen_out_tensor()

        self.k = np.reshape(self.k, (batch, max_seq, kv_head * embeddim))
        self.v = np.reshape(self.v, (batch, max_seq, kv_head * embeddim))
        self.mask = np.reshape(self.mask, (batch, max_seq, max_seq))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case_clamp_decoder(self):
        # unpad encoder
        batch = 2
        kv_head = 32        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 512
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [114, 512]
        is_clamp = 1
        clamp_min = -0.71
        clamp_max = 0.99
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 1, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
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

        attention_out = np.zeros_like(self.q)

        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                           [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case_clamp_encoder(self):
        # dynamic batch decoder
        batch = 2
        kv_head = 32        # kv_head num
        isdecoder = 1       # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 512
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [114, 512]
        is_clamp = 1
        clamp_min = -0.71
        clamp_max = 0.99
        dynamic_batch = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 1, "qSeqLen": [1] * batch, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD_DECODER, tor = tor)
        self.gen_out_tensor()

        self.mask = np.reshape(self.mask, (batch, 1, max_seq))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])


    @op_test.only_910b
    def test_flash_attention_case_clamp_encoder_longseq(self):
        # dynamic batch decoder
        batch = 2
        kv_head = 32        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 523
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [523, 512]
        is_clamp = 1
        clamp_min = -0.71
        clamp_max = 0.99
        dynamic_batch = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 1, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "isClamp" : is_clamp,
                    "clampMin" : clamp_min, "clampMax" : clamp_max, "isTriuMask": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH, long_seq = True, tor = tor)
        self.gen_out_tensor()

        self.mask = self.mask.view(128, 128).to(data_type)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask, torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case_clamp_encoder_bf16_nomask(self):
        # dynamic batch decoder
        batch = 2
        kv_head = 32        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 523
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [523, 512]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        batch_state = [1, 1]
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 4, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "isClamp" : is_clamp,
                    "clampMin" : clamp_min, "clampMax" : clamp_max, "isTriuMask": 0, "batchRunStatus": batch_state, "maskType": 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch, batch_state = batch_state,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_MASK, is_mask = False, tor = tor)
        self.gen_out_tensor()

        self.k = self.k.view(batch, max_seq, heads * embeddim)
        self.v = self.v.view(batch, max_seq, heads * embeddim)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, self.layer_id, torch.tensor([], dtype=data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=data_type)])

    @op_test.only_910b
    def test_flash_attention_clamp_encoder_bf16_nomask_bnsd_case3(self):
        # dynamic batch decoder
        batch = 2
        kv_head = 32        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 523
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [523, 512]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        batch_state = [1, 1]
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 1, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "isClamp" : is_clamp,
                    "clampMin" : clamp_min, "clampMax" : clamp_max, "isTriuMask": 0, "batchRunStatus": batch_state, "maskType": 0,"dataShapeType":1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch, batch_state = batch_state,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_MASK, is_mask = False, tor = tor)
        self.gen_out_tensor_bnsd()
        self.k = self.k.view(batch, max_seq, heads * embeddim)
        self.v = self.v.view(batch, max_seq, heads * embeddim)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.qbnsd.to(torch.float16))
        return self.execute([self.qbnsd, self.kbnsd, self.vbnsd, self.layer_id, torch.tensor([], dtype=data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=data_type)])

    @op_test.only_910b
    def test_flash_attention_case_clamp_encoder_longseq_bf16(self):
        # dynamic batch decoder
        batch = 2
        kv_head = 32        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 523
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [523, 512]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        batch_state = [1, 1]
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 4, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "isClamp" : is_clamp,
                    "clampMin" : clamp_min, "clampMax" : clamp_max, "isTriuMask": 1, "batchRunStatus": batch_state}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch, batch_state = batch_state,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, long_seq = True, tor = tor)
        self.gen_out_tensor()

        self.k = self.k.view(batch, max_seq, heads * embeddim)
        self.v = self.v.view(batch, max_seq, heads * embeddim)

        self.mask = self.mask.view(batch, 128, 128)[0, :, :]

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=data_type)])

    @op_test.only_910b
    def test_flash_attention_case_fp32_encoder1(self):
        # unpad encoder
        batch = 1
        kv_head = 8       # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 256
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [256]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2001, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float32

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
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

        attention_out = np.zeros_like(self.q)
        return self.execute([self.q.to(torch.float16), self.k.to(torch.float16), self.v.to(torch.float16), self.layer_id.to(torch.int), self.mask.to(torch.float16), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case_fp32_encoder_nocache(self):
        # unpad encoder
        batch = 2
        kv_head = 8       # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 512
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [114, 512]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2001, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "isClamp" : is_clamp,
                    "clampMin" : clamp_min, "clampMax" : clamp_max}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float32

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
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

        attention_out = np.zeros_like(self.q)
        return self.execute([self.q.to(torch.float16), self.k.to(torch.float16), self.v.to(torch.float16), self.layer_id.to(torch.int),
                             self.mask.to(torch.float16), torch.tensor([], dtype=torch.float16),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case_dynamic_batch_bf16_encoder_template(self):
        # unpad dynamic batch encoder
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 1          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 512
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False

        kv_seqLen = [129]

        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 4, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16
        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, tor = tor)
        self.gen_out_tensor()

        self.k = self.k.view(batch, max_seq, heads * embeddim)
        self.v = self.v.view(batch, max_seq, heads * embeddim)
        self.mask = self.mask.view(batch, max_seq, max_seq)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q.to(torch.bfloat16), self.k.to(torch.bfloat16),
                             self.v.to(torch.bfloat16), self.layer_id, self.mask.to(torch.bfloat16), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=data_type)])

    @op_test.only_910b
    def test_flash_attention_case_fp32_decoder(self):
        # dynamic batch decoder
        batch = 2
        kv_head = 32        # kv_head num
        isdecoder = 1       # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 512
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [114, 512]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2001, "qSeqLen": [1] * batch, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float32

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD_DECODER, tor = tor)
        self.gen_out_tensor()

        self.mask = np.reshape(self.mask, (batch, 1, max_seq))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([self.q.to(torch.float16), self.k.to(torch.float16), self.v.to(torch.float16), self.layer_id.to(torch.int), self.mask.to(torch.float16), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_nocache_nomask_template(self):
        # [b,ms,ms]
        batch = 2
        kv_head = 32        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32
        embeddim = 128
        max_seq = 128
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [114, 105]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "isClamp" : is_clamp,
                    "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_MASK, no_cache = True, tor = tor)
        self.gen_out_tensor()

        attention_out = np.zeros_like(self.q)
        return self.execute([self.q, self.k, self.v, torch.tensor([], dtype=torch.int), torch.tensor([], dtype=torch.half), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_nocache_mask1_temp(self):
        # [b,ms,ms]
        batch = 8
        kv_head = 32        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32
        embeddim = 128
        max_seq = 707
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        batchRunStatus = [1, 1, 0, 1, 0, 1, 0, 1]
        kv_seqLen = [114, 707, 0, 5, 0, 19, 0, 1]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = True
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "isClamp" : is_clamp,
                    "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 1, "isTriuMask": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch, batch_state = batchRunStatus,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD,
                             no_cache = True, is_triu_mask = True , tor = tor)
        self.gen_out_tensor()
        attention_out = np.zeros_like(self.q)
        return self.execute([self.q, self.k, self.v, torch.tensor([], dtype=torch.int), self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_nocache_mask1_longseq(self):
        # [b,ms,ms]
        batch = 2
        kv_head = 32        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32
        embeddim = 128
        max_seq = 512
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        batchRunStatus = [1]
        kv_seqLen = [512, 399]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "isClamp" : is_clamp,
                    "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 1, "isTriuMask": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch, batch_state = batchRunStatus,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD,
                             no_cache = True, long_seq= True, tor = tor)
        self.gen_out_tensor()
        self.mask = self.mask.view(batch, 128, 128)[0, :, :]
        attention_out = np.zeros_like(self.q)
        return self.execute([self.q, self.k, self.v, torch.tensor([], dtype=torch.int), self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_nocache_mask1_dim96(self):
        # [b,ms,ms]
        batch = 3
        kv_head = 32        # kv_head num
        heads = 32
        embeddim = 96
        max_seq = 512
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [114, 105, 500]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "isClamp" : is_clamp,
                    "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, no_cache = True, tor = tor)
        self.gen_out_tensor()

        attention_out = np.zeros_like(self.q)
        return self.execute([self.q, self.k, self.v, torch.tensor([], dtype=torch.int), self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_nocache_alibimask(self):
        # [b, headdim,ms,ms]
        batch = 2
        kv_head = 32        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 707
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [114, 707]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "isClamp" : is_clamp,
                    "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 2}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_BATCH,
                             no_cache = True, tor = tor)
        self.gen_out_tensor()

        attention_out = np.zeros_like(self.q)
        return self.execute([self.q, self.k, self.v, torch.tensor([], dtype=torch.int), self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_nocache_bf16(self):
        # [b,ms,ms]
        batch = 2
        kv_head = 32        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32
        embeddim = 128
        max_seq = 200
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [115, 15]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])

        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, no_cache = True, tor = tor)
        self.gen_out_tensor()
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, torch.tensor([], dtype=torch.int), self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_nocache_bf16_clamp(self):
        # [b,ms,ms]
        batch = 2
        kv_head = 32        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32
        embeddim = 128
        max_seq = 200
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [115, 15]
        is_clamp = 1
        clamp_min = -0.71
        clamp_max = 0.99
        dynamic_batch = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])

        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, no_cache = True, tor = tor)
        self.gen_out_tensor()
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, torch.tensor([], dtype=torch.int), self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_nocache_bf16_q_dim3(self):
        # [b,ms,ms]
        #batch = 2
        batch = 1
        kv_head = 32        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32
        embeddim = 128
        max_seq = 200
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [115]
        #kv_seqLen = [115, 15]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])

        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH, no_cache = True, tor = tor)
        self.gen_out_tensor()
        attention_out = np.zeros_like(self.q.to(torch.float16))
        self.q = self.q.view(sum(kv_seqLen), heads, embeddim)
        self.k = self.k.view(sum(kv_seqLen), heads, embeddim)
        self.v = self.v.view(sum(kv_seqLen), heads, embeddim)
        return self.execute([self.q, self.k, self.v, torch.tensor([], dtype=torch.int), self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_nocache_bf16_long_seq(self):
        # [b,ms,ms]
        batch = 2
        kv_head = 32        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32
        embeddim = 128
        max_seq = 523
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [523, 512]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 1, "isTriuMask": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])

        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, no_cache = True,
                             long_seq = True, tor = tor)
        self.gen_out_tensor()
        self.mask = self.mask.view(batch, 128, 128)[0, :, :]
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, torch.tensor([], dtype=torch.int), self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_nocache_alibimask_dim3(self):
        # [b, headdim,ms,ms]
        batch = 2
        kv_head = 32        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 800
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [114, 707]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "isClamp" : is_clamp,
                    "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 2}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_NO_BATCH, no_cache = True, tor = tor)
        self.gen_out_tensor()
        attention_out = np.zeros_like(self.q)
        return self.execute([self.q, self.k, self.v, torch.tensor([], dtype=torch.int), self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case_bf16_encoder(self):
        # unpad encoder
        batch = 2
        kv_head = 32        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 512
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [129, 320]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 1, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH, tor = tor)
        self.gen_out_tensor()

        self.mask = self.mask.view(max_seq, max_seq).to(data_type)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask, torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_bf16_encoder_multi_layer(self):
        # unpad encoder
        batch = 5
        kv_head = 4        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 512
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [18, 180, 181, 18, 18]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 1, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "kvHead": kv_head, "tor": tor, "isTriuMask": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH,
                             long_seq = True, is_multi_layer = True, tor = tor)
        self.gen_out_tensor()

        self.mask = self.mask.view(128, 128).to(data_type)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask, torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_bf16_decoder(self):
        # unpad encoder
        batch = 8
        kv_head = 32        # kv_head num
        isdecoder = 1      # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 1024
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [32, 1024] * 4
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 12, "qSeqLen":[1] * batch, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD_DECODER, tor = tor)
        self.gen_out_tensor()
        self.mask = self.mask.view(batch, 1, max_seq).to(data_type)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask, torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_bf16_decoder_nomask(self):
        # unpad encoder
        batch = 8
        kv_head = 4        # kv_head num
        isdecoder = 1      # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 1024
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [32, 1024] * 4
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 12, "qSeqLen":[1] * batch, "kvSeqLen": kv_seqLen, "headSize": heads,"kvHead":kv_head,"tor": tor, "maskType": 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False, is_mask = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD_DECODER, tor = tor)
        self.gen_out_tensor()
        self.mask = self.mask.view(batch, 1, max_seq).to(data_type)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, self.layer_id, torch.tensor([], dtype=torch.bfloat16), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_alibi_case(self):
        # unpad encoder
        batch = 28
        kv_head = 32        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 512
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [1, 2, 127, 505] * 7
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 1, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "maskType": 2}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_BATCH, tor = tor)
        self.gen_out_tensor()

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])


    @op_test.only_910b
    def test_flash_attention_hf_alibi_case(self):
        # dynamic batch decoder
        batch = 8
        kv_head = 8        # kv_head num
        isdecoder = 1       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = True
        batchRunStatus = [1, 1, 0, 1, 0, 1, 0, 1]
        kv_seqLen = [114, 707, 0, 5, 0, 2047, 0, 1]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 13, "qSeqLen": [1] * batch, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "batchRunStatus": batchRunStatus, "maskType": 2}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch, batch_state= batchRunStatus,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_BATCH, tor = tor)
        self.gen_out_tensor()

        self.k = np.reshape(self.k, (batch, max_seq, heads * embeddim))
        self.v = np.reshape(self.v, (batch, max_seq, heads * embeddim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_nocache_bf16_alibi(self):
        # [b,ms,ms]
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 12
        embeddim = 128
        max_seq = 256
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [256]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 2, "kvHead": kv_head}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_BATCH, no_cache = True, tor = tor)
        self.gen_out_tensor()

        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, torch.tensor([], dtype=torch.int), self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_fp32_encoder_alibi(self):
        # unpad encoder
        batch = 1
        kv_head = 8       # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 256
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [256]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2001, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 2}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float32

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_NO_BATCH, tor = tor)
        self.gen_out_tensor()
        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([self.q.to(torch.float16), self.k.to(torch.float16), self.v.to(torch.float16),
                             self.layer_id.to(torch.int), self.mask.to(torch.float16), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case1_alibi(self):
        # unpad dynamic batch encoder
        batch = 12
        kv_head = 32        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 1024
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = True
        batch_state = [1] * batch
        kv_seqLen = [1, 41, 88, 144, 323, 544] * 2

        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 4, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "batchRunStatus": batch_state, "maskType": 2}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16
        self.set_data_params(dynamic_batch = dynamic_batch, batch_state = batch_state,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_BATCH, tor = tor)
        self.gen_out_tensor()

        self.k = np.reshape(self.k, (batch, max_seq, heads * embeddim))
        self.v = np.reshape(self.v, (batch, max_seq, heads * embeddim))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q)
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case_dynamic_batch_bf16_encoder(self):
        # unpad dynamic batch encoder
        batch = 8
        kv_head = 32        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 1024
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = True
        batch_state = [1, 1, 0, 1, 0, 1, 0, 1]
        kv_seqLen = [114, 707, 0, 1023, 0, 1, 0, 1]

        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 4, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "batchRunStatus": batch_state}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16
        self.set_data_params(dynamic_batch = dynamic_batch, batch_state = batch_state,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, tor = tor)
        self.gen_out_tensor()

        self.k = self.k.view(batch, max_seq, heads * embeddim)
        self.v = self.v.view(batch, max_seq, heads * embeddim)
        self.mask = self.mask.view(batch, max_seq, max_seq).to(data_type)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask, torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=data_type)])

    @op_test.only_910b
    def test_flash_attention_case_dynamic_batch_bf16_decoder(self):
        # dynamic batch decoder
        batch = 8
        kv_head = 8        # kv_head num
        isdecoder = 1       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = True
        batch_state = [1, 1, 0, 1, 0, 1, 0, 1]
        kv_seqLen = [114, 707, 0, 5, 0, 2047, 0, 1]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        is_mask = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 4, "qSeqLen": [1] * batch, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "batchRunStatus": batch_state}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch, batch_state = batch_state, is_mask = True,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, tor = tor)
        self.gen_out_tensor()

        self.k = self.k.view(batch, max_seq, heads * embeddim)
        self.v = self.v.view(batch, max_seq, heads * embeddim)
        self.mask = self.mask.view(batch, max_seq, max_seq).to(data_type)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask, torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_dynamic_batch_bf16_decoder_alibi(self):
        # dynamic batch decoder
        batch = 8
        kv_head = 8        # kv_head num
        is_decoder = 1       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = True
        batch_state = [1, 1, 0, 1, 0, 1, 0, 1]
        kv_seqLen = [114, 707, 0, 5, 0, 2047, 0, 1]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        is_alibi = True
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 4, "qSeqLen": [1] * batch, "kvSeqLen": kv_seqLen, "headSize": heads,
                    "tor": tor, "batchRunStatus": batch_state, "maskType": 2}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch, batch_state = batch_state, is_mask = True,
                             is_decoder = is_decoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = is_alibi,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_BATCH, tor = tor)
        self.gen_out_tensor()

        self.k = self.k.view(batch, max_seq, heads * embeddim)
        self.v = self.v.view(batch, max_seq, heads * embeddim)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_dynamic_batch_bf16_hp_decoder(self):
        # dynamic batch decoder
        batch = 8
        kv_head = 8        # kv_head num
        is_decoder = 1       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = True
        batch_state = [1, 1, 0, 1, 0, 1, 0, 1]
        kv_seqLen = [114, 707, 0, 5, 0, 2047, 0, 1]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 13, "qSeqLen": [1] * batch, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "batchRunStatus": batch_state}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch, batch_state = batch_state, is_mask = True,
                             is_decoder = is_decoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min, data_type = data_type,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, tor = tor)
        self.gen_out_tensor()
        self.k = self.k.view(batch, max_seq, heads * embeddim)
        self.v = self.v.view(batch, max_seq, heads * embeddim)
        self.mask = self.mask.view(batch, max_seq, max_seq).to(data_type)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask, torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=data_type)])

    @op_test.only_910b
    def test_flash_attention_case_dynamic_batch_bf16_hp_decoder_gqa_opt(self):
        # dynamic batch decoder
        batch = 8
        kv_head = 4        # kv_head num
        is_decoder = 1       # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        batch_state = []
        kv_seqLen = [114, 707, 1, 5, 1, 2047, 1, 1]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 13, "qSeqLen": [1] * batch, "kvSeqLen": kv_seqLen, "headSize": heads, "kvHead": kv_head, "tor": tor, "batchRunStatus": batch_state}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch, batch_state = batch_state, is_mask = True,
                             is_decoder = is_decoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min, data_type = data_type,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, tor = tor)
        self.gen_out_tensor()
        self.k = self.k.view(batch, max_seq, kv_head * embeddim)
        self.v = self.v.view(batch, max_seq, kv_head * embeddim)
        self.mask = self.mask.view(batch, max_seq, max_seq).to(data_type)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask, torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=data_type)])

    @op_test.only_910b
    def test_flash_attention_case_dynamic_batch_bf16_hp_decoder_clamp(self):
        # dynamic batch decoder
        batch = 8
        kv_head = 8        # kv_head num
        is_decoder = 1       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = True
        batch_state = [1, 1, 0, 1, 0, 1, 0, 1]
        kv_seqLen = [114, 707, 0, 5, 0, 2047, 0, 1]
        is_clamp = 1
        clamp_min = -0.71
        clamp_max = 0.99
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 13, "qSeqLen": [1] * batch, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "batchRunStatus": batch_state, "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch, batch_state = batch_state, is_mask = True,
                             is_decoder = is_decoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min, data_type = data_type,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, tor = tor)
        self.gen_out_tensor()
        self.k = self.k.view(batch, max_seq, heads * embeddim)
        self.v = self.v.view(batch, max_seq, heads * embeddim)
        self.mask = self.mask.view(batch, max_seq, max_seq).to(data_type)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask, torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=data_type)])

    @op_test.only_910b
    def test_flash_attention_encoder_half_big_batch(self):
        # unpad encoder
        batch = 128
        kv_head = 32        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 512
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [1, 2, 127, 505, 510, 120, 340, 512] * 16
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 1, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
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

        attention_out = np.zeros_like(self.q)
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case_dynamic_batch_bf16_hp_decoder_big_batch(self):
        # dynamic batch decoder
        batch = 128
        kv_head = 8        # kv_head num
        is_decoder = 1       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = True
        batch_state = [1, 1, 0, 1, 0, 1, 0, 1] * 16
        kv_seqLen = [114, 707, 0, 5, 0, 2047, 0, 1] * 16
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 13, "qSeqLen": [1] * batch, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "batchRunStatus": batch_state}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch, batch_state = batch_state, is_mask = True,
                             is_decoder = is_decoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min, data_type = data_type,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, tor = tor)
        self.gen_out_tensor()
        self.k = self.k.view(batch, max_seq, heads * embeddim)
        self.v = self.v.view(batch, max_seq, heads * embeddim)
        self.mask = self.mask.view(batch, max_seq, max_seq).to(data_type)

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask, torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=data_type)])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_nocache_bf16_alibi_long_seq(self):
        # [b,ms,ms]
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 12
        embeddim = 128
        max_seq = 512
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [512]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 2, "kvHead": kv_head,
                    "isTriuMask" : 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_BATCH,
                             no_cache = True, tor = tor)
        self.gen_out_tensor()

        self.alibi_slopes *= -1
        mask = np.ones((256,256)) * 60000
        mask = np.triu(mask, 1)
        self.mask = self.bias[:256, :256] * -1 + mask
        self.mask = self.mask.to(torch.bfloat16)

        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, torch.tensor([], dtype=torch.int), self.mask, self.alibi_slopes.to(torch.float32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_nocache_bf16_sqrt_alibi_long_seq(self):
        # [b,ms,ms]
        batch = 2
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 12
        embeddim = 128
        max_seq = 4096
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [4090, 1000]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 2, "kvHead": kv_head,
                    "isTriuMask" : 1, "isAlibiMaskSqrt": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_BATCH,
                             no_cache = True, is_sqrt = True, tor = tor)
        self.gen_out_tensor()

        self.alibi_slopes *= -1
        mask = np.ones((256,256)) * 60000
        mask = np.triu(mask, 1)
        self.mask = self.bias[:256, :256] * -1 + mask
        self.mask = self.mask.to(torch.bfloat16)

        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, torch.tensor([], dtype=torch.int), self.mask, self.alibi_slopes.to(torch.float32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_nocache_bf16_sqrt_alibi_long_seq_dim3(self):
        # [b,ms,ms]
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 12
        embeddim = 128
        max_seq = 4096
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [4096]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 2, "kvHead": kv_head,
                    "isTriuMask" : 1, "isAlibiMaskSqrt": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_BATCH,
                             no_cache = True, is_sqrt = True, tor = tor)
        self.gen_out_tensor()
        self.mask = self.mask[0, :, :, :128].to(torch.bfloat16)
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, torch.tensor([], dtype=torch.int), self.mask, torch.tensor([], dtype=torch.float32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_nocache_bf16_left_align_alibi_long_seq(self):
        # [b,ms,ms]
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 12
        embeddim = 128
        max_seq = 4096
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [4096]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 2, "kvHead": kv_head,
                    "isTriuMask" : 1, "alibiLeftAlign": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_BATCH,
                             no_cache = True, left_align = True, tor = tor)
        self.gen_out_tensor()
        mask = np.ones((256,256)) * -(float("inf"))
        mask = np.triu(mask, 1)
        self.mask = self.bias[0, :256, :256] + mask
        self.mask = self.mask.to(torch.bfloat16)
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, torch.tensor([], dtype=torch.int), self.mask, self.alibi_slopes.to(torch.float32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_nocache_bf16_triu_small_head_dim(self):
        # [b,ms,ms]
        batch = 2
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 1
        embeddim = 71
        max_seq = 523
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [523, 512]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 1, "isTriuMask": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])

        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, no_cache = True, tor = tor)
        self.gen_out_tensor()
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, torch.tensor([], dtype=torch.int), self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    # @op_test.only_910b
    # def test_flash_attention_case_decoder_dynamic_batch_mla(self):
    #     # dynamic batch decoder
    #     batch = 64
    #     kv_head = 8        # kv_head num
    #     is_decoder = 1       # prefill or decoder
    #     heads = 8          # llama7b  hidden_size 4096
    #     embeddim = 576
    #     embeddimv = 512
    #     max_seq = 2048
    #     tor = 1.0 / math.sqrt(1.0 * embeddim)
    #     dynamic_batch = True
    #     batch_state = [1, 1, 0, 1, 0, 1, 0, 1] * 8
    #     kv_seqLen = [114, 707, 0, 5, 0, 2047, 0, 1] * 8
    #     is_clamp = 0
    #     clamp_min = 0
    #     clamp_max = 0
    #     OP_NAME = "UnpadFlashAttentionOperation"
    #     OP_PARAM = {"type": 13, "qSeqLen": [1] * batch, "kvSeqLen": kv_seqLen, "headDimV": embeddimv, "headSize": heads, "tor": tor, "batchRunStatus": batch_state}
    #     self.set_param(OP_NAME, OP_PARAM)
    #     self.set_input_formats([self.format_nd] * 12)
    #     self.set_output_formats([self.format_nd])
    #     data_type = torch.bfloat16

    #     self.set_data_params(dynamic_batch = dynamic_batch, batch_state = batch_state, is_mask = True,
    #                          is_decoder = is_decoder, batch = batch, kv_head = kv_head, heads = heads,
    #                          embeddim = embeddim,embeddimv = embeddimv, max_seq = max_seq, kv_seqLen = kv_seqLen,
    #                          is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min, data_type = data_type,
    #                          op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, tor = tor)
    #     self.gen_out_tensor()
    #     self.k = self.k.view(batch, max_seq, heads * embeddim)
    #     self.v = self.v.view(batch, max_seq, heads * embeddimv)
    #     self.mask = self.mask.view(batch, max_seq, max_seq).to(data_type)

    #     logging.debug("**********input shape***********")
    #     logging.debug(f"q shape: {self.q.shape}")
    #     logging.debug(f"k shape: {self.k.shape}")
    #     logging.debug(f"v shape: {self.v.shape}")
    #     logging.debug(f"layer_id shape: {self.layer_id.shape}")
    #     logging.debug(f"mask shape: {self.mask.shape}")

    #     attention_out = np.zeros_like(self.golden_out.to(torch.float16))
    #     return self.execute([self.q, self.k, self.v, self.layer_id, self.mask, torch.tensor([], dtype=torch.float),
    #                          torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
    #                          torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
    #                          torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
    #                         [torch.tensor(attention_out, dtype=data_type)])

    @op_test.only_910b
    def test_flash_attention_mla_fp16(self):
        # unpad encoder
        batch = 16
        kv_head = 8        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
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
        OP_PARAM = {"type": 1, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV,"headSize": heads, "tor": tor}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
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
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=data_type)])

    @op_test.only_910b
    def test_flash_attention_mla_bf16(self):
        # unpad encoder
        batch = 16
        kv_head = 8        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
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
        OP_PARAM = {"type": 1, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV,"headSize": heads, "tor": tor}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

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

        attention_out = np.zeros_like(self.golden_out)
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_nocache_bf16_case_gemma(self):
        # [b,ms,ms]
        batch = 1
        kv_head = 4        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 4
        embeddim = 256
        max_seq = 999
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [999]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "kvHead":kv_head,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 1, "isTriuMask": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])

        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, no_cache = True, long_seq = True, tor = tor)
        self.gen_out_tensor()
        self.mask = self.mask.view(batch, 128, 128)[0, :, :]

        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, torch.tensor([], dtype=torch.int), self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_nocache_bf16_case_ceval(self):
        # [b,ms,ms]
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8
        embeddim = 128
        max_seq = 999
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [999]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "kvHead":kv_head,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 1, "isTriuMask": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])

        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, no_cache = True, long_seq = True, tor = tor)
        self.gen_out_tensor()
        self.mask = self.mask.view(batch, 128, 128)[0, :, :]

        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, torch.tensor([], dtype=torch.int), self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_case_gemma(self):
        # [b,ms,ms]
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 1
        embeddim = 256
        max_seq = 999
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        batchRunStatus = [1]
        kv_seqLen = [999]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "isClamp" : is_clamp,
                    "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 1, "isTriuMask": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch, batch_state = batchRunStatus,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, no_cache = True, long_seq= True, tor = tor)
        self.gen_out_tensor()
        self.mask = self.mask.view(batch, 128, 128)[0, :, :]

        attention_out = np.zeros_like(self.golden_out)
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_mla_fp32(self):
        self.set_support_910b_only()
        # unpad encoder
        batch = 16
        kv_head = 8       # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 512
        max_seq = 512
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [512] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2001, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "headDimV": embeddimV, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 2}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float32

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, embeddimv=embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_NO_BATCH, tor = tor)
        self.gen_out_tensor()
        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.golden_out)
        return self.execute([self.q.to(torch.float16), self.k.to(torch.float16), self.v.to(torch.float16),
                             self.layer_id.to(torch.int), self.mask.to(torch.float16), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case_decoder_bf16(self):
        # dynamic batch decoder
        batch = 16
        kv_head = 1        # kv_head num
        isdecoder = 1       # prefill or decoder
        heads = 16          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimv = 512
        max_seq = 1024
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [1024] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 12, "qSeqLen": [1] * batch, "kvSeqLen": kv_seqLen, "headSize": heads, "headDimV": embeddimv, "kvHead": kv_head, "tor": tor, "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                            is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                            embeddim = embeddim, embeddimv = embeddimv, max_seq = max_seq, kv_seqLen = kv_seqLen,
                            is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                            data_type = data_type, is_alibi = False,
                            op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, tor = tor)
        self.gen_out_tensor()

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.golden_out.to(torch.float16))
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_decoder_fp16(self):
        # dynamic batch decoder
        batch = 16
        kv_head = 16        # kv_head num
        isdecoder = 1       # prefill or decoder
        heads = 16          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimv = 512
        max_seq = 1024
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [1024] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 12, "qSeqLen": [1] * batch, "kvSeqLen": kv_seqLen, "headSize": heads, "headDimV": embeddimv, "kvHead": kv_head, "tor": tor, "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                            is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                            embeddim = embeddim, embeddimv = embeddimv, max_seq = max_seq, kv_seqLen = kv_seqLen,
                            is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                            data_type = data_type, is_alibi = False,
                            op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, tor = tor)
        self.gen_out_tensor()

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.golden_out)
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type),torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_nocache_bf16_fav3_int8_online(self):
        # [b,ms,ms]
        batch = 1
        kv_head = 20        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 20
        embeddim = 128
        max_seq = 128
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [max_seq] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        fav3 = True
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "kvHead": kv_head, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 1, "isTriuMask": 0,
                    "quantType": 3 if fav3 else 0, "outDataType": 27}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])

        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, no_cache = True, fav3 = fav3, is_mask = True, tor = tor)
        self.gen_out_tensor()
        attention_out = np.zeros_like(self.q.to(torch.float16))
        if not self.fav3:
            return self.execute([self.q, self.k, self.v, torch.tensor([], dtype=torch.int), self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                            torch.tensor([], dtype=torch.float),torch.tensor([], dtype=torch.int),
                            torch.tensor([], dtype=torch.float),torch.tensor([], dtype=torch.int),torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])
        else:
            return self.execute([self.q_int8, self.k_int8, self.v_int8, torch.tensor([], dtype=torch.int), self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                            self.q_scale,
                            torch.tensor([], dtype=torch.int32),
                            self.v_scale,
                            torch.tensor([], dtype=torch.int32),
                            torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.float32)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_nocache_bf16_fav3_int8_offline(self):
        # [b,ms,ms]
        batch = 5
        kv_head = 4        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 32
        embeddim = 128
        max_seq = 128
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [max_seq] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        fav3=True
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "kvHead": kv_head, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 1, "isTriuMask": 0,
                    "quantType": 2 if fav3 else 0, "outDataType": 27}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])

        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, no_cache = True, fav3 = fav3, is_mask = True, tor = tor)

        self.gen_out_tensor()
        attention_out = np.zeros_like(self.q.to(torch.float16))
        if not self.fav3:
            return self.execute([self.q, self.k, self.v, torch.tensor([], dtype=torch.int), self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                            torch.tensor([], dtype=torch.float),torch.tensor([], dtype=torch.int),
                            torch.tensor([], dtype=torch.float),torch.tensor([], dtype=torch.int),torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])
        else:
            return self.execute([self.q_int8, self.k_int8, self.v_int8, torch.tensor([], dtype=torch.int), self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                            self.q_scale,
                            torch.tensor([], dtype=torch.int32),
                            self.v_scale,
                            torch.tensor([], dtype=torch.int32),
                            self.offline_scale, torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_nocache_fp16_fav3_int8_online(self):
        # [b,ms,ms]
        batch = 1
        kv_head = 2        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 2
        embeddim = 128
        max_seq = 257
        tor = 1.0
        kv_seqLen = [max_seq] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        fav3 = True
        online = True
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "kvHead": kv_head, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 1, "isTriuMask": 0,
                    "quantType": 3 if fav3 else 0}
        logging.info(f'input info: {batch}, {max_seq}, {heads}, {kv_head}, {embeddim}')
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])

        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                            is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                            embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                            is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                            data_type = data_type, is_alibi = False,
                            op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, no_cache = True, fav3 = fav3, is_mask = True, tor = tor)
        self.gen_out_tensor(online)
        attention_out = np.zeros_like(self.q.to(torch.float16))

        if not self.fav3:
            return self.execute([self.q, self.k, self.v, torch.tensor([], dtype=torch.int), self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                            torch.tensor([], dtype=torch.float),torch.tensor([], dtype=torch.int),
                            torch.tensor([], dtype=torch.float),torch.tensor([], dtype=torch.int),torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.float16)])
        else:
            return self.execute([self.q_int8, self.k_int8, self.v_int8, torch.tensor([], dtype=torch.int), self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                            self.q_scale,
                            torch.tensor([], dtype=torch.int32),
                            self.v_scale,
                            torch.tensor([], dtype=torch.int32),
                            torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.float16)])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_nocache_fp16_fav3_int8_offline(self):
        # [b,ms,ms]
        batch = 1
        kv_head = 2        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 2
        embeddim = 128
        max_seq = 777
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [max_seq] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        fav3 = True
        online = False
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "kvHead": kv_head, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 1, "isTriuMask": 0,
                    "quantType": 2 if fav3 else 0}
        logging.info(f'input info: {batch}, {max_seq}, {heads}, {kv_head}, {embeddim}')
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])

        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                            is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                            embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                            is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                            data_type = data_type, is_alibi = False,
                            op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, no_cache = True, fav3 = fav3, is_mask = True, tor = tor)

        self.gen_out_tensor()
        attention_out = np.zeros_like(self.q.to(torch.float16))

        if not self.fav3:
            return self.execute([self.q, self.k, self.v, torch.tensor([], dtype=torch.int), self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                            torch.tensor([], dtype=torch.float),torch.tensor([], dtype=torch.int),
                            torch.tensor([], dtype=torch.float),torch.tensor([], dtype=torch.int),torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.float16)])
        else:
            return self.execute([self.q_int8, self.k_int8, self.v_int8, torch.tensor([], dtype=torch.int), self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                            self.q_scale,
                            torch.tensor([], dtype=torch.int32),
                            self.v_scale,
                            torch.tensor([], dtype=torch.int32),
                            self.offline_scale, torch.tensor([], dtype=torch.float32)],
                            [torch.tensor(attention_out, dtype=torch.float16)])

    @op_test.only_910b
    def test_flash_attention_case_encoder_bf16_tor_nomask(self):
        batch = 1
        kv_head = 3        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 3
        embeddim = 128
        max_seq = 2077
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        kv_seqLen = [max_seq]*batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        scaleType = ScaleType.SCALE_TOR.value
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 10, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor, "kvHead":kv_head,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 0,"scaleType":scaleType}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])

        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch, no_cache=True,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_MASK, long_seq = True, scaleType = scaleType, tor = tor)
        self.gen_out_tensor()
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v, self.layer_id, torch.tensor([], dtype=torch.bfloat16), torch.tensor([], dtype=torch.float),
                            torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                             [torch.tensor(attention_out, dtype=torch.bfloat16)])
    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_withcache_fp16_alibi(self):
        # [b,ms,ms]
        batch = 10
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 12
        embeddim = 128
        max_seq = 256*3
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = [128] * batch
        kv_seqLen = [128*4] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2012, "qSeqLen": q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 2,"kvHead": kv_head}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_NO_BATCH_WITH_PREFIX,
                             tor = tor, q_seqlens = q_seqlens,
                             num_blocks = num_blocks, block_size = block_size)
        self.gen_out_tensor()
        self.mask = self.mask.to(torch.float16)
        attention_out = np.zeros_like(self.q.to(torch.float16))
        logging.info(f"self.mask: {self.mask}")
        logging.info(f"self.alibi_slopes: {self.alibi_slopes},self.block_tables.shape:{self.block_tables.shape},{self.block_tables}")
        return self.execute([self.q, self.k_cache, self.v_cache, self.block_tables,
                             self.mask, torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.float16)])
    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_withcache_bf16_alibi_sqrt_single_batch_without_tail_block(self):
        # [b,ms,ms]
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 12
        embeddim = 128
        max_seq = 4096
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = [128]
        kv_seqLen = [512]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2012, "qSeqLen": q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 7, "kvHead": kv_head,
                    "isTriuMask" : 1, "alibiLeftAlign": 0, "isAlibiMaskSqrt": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_PREFIX_BATCH,
                             no_cache = True, is_sqrt = True, tor = tor, q_seqlens = q_seqlens,
                             num_blocks = num_blocks, block_size = block_size)
        self.gen_out_tensor()
        self.alibi_slopes *= -1
        mask = np.ones((256,256)) * float("inf")
        mask = np.triu(mask, 1)
        self.mask = self.bias[:256, :256] * -1 + mask
        self.mask = self.mask.to(torch.bfloat16)
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k_cache, self.v_cache, self.block_tables,
                             self.mask, self.alibi_slopes.to(torch.float32)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_withcache_bf16_alibi_sqrt_single_batch_with_tail_block(self):
        # [b,ms,ms]
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 12
        embeddim = 128
        max_seq = 8192
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = [8062]
        kv_seqLen = [8190]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2012, "qSeqLen": q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 7, "kvHead": kv_head,
                    "isTriuMask" : 1, "alibiLeftAlign": 0, "isAlibiMaskSqrt": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_PREFIX_BATCH,
                             no_cache = True, is_sqrt = True, tor = tor, q_seqlens = q_seqlens,
                             num_blocks = num_blocks, block_size = block_size)
        self.gen_out_tensor()
        self.alibi_slopes *= -1
        mask = np.ones((256,256)) * float("inf")
        mask = np.triu(mask, 1)
        self.mask = self.bias[:256, :256] * -1 + mask
        self.mask = self.mask.to(torch.bfloat16)
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k_cache, self.v_cache, self.block_tables,
                             self.mask, self.alibi_slopes.to(torch.float32)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_withcache_bf16_alibi_sqrt_multi_batch_without_tail_block(self):
        # [b,ms,ms]
        batch = 10
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 12
        embeddim = 128
        max_seq = 4096
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = [128, 128, 128, 128, 128, 256, 256, 256, 256, 256]
        kv_seqLen = [256, 256, 256, 256, 256, 512, 512, 512, 512, 512]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2012, "qSeqLen": q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 7, "kvHead": kv_head,
                    "isTriuMask" : 1, "alibiLeftAlign": 0, "isAlibiMaskSqrt": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_PREFIX_BATCH,
                             no_cache = True, is_sqrt = True, tor = tor, q_seqlens = q_seqlens,
                             num_blocks = num_blocks, block_size = block_size)
        self.gen_out_tensor()
        self.alibi_slopes *= -1
        mask = np.ones((256,256)) * float("inf")
        mask = np.triu(mask, 1)
        self.mask = self.bias[:256, :256] * -1 + mask
        self.mask = self.mask.to(torch.bfloat16)
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k_cache, self.v_cache, self.block_tables,
                             self.mask, self.alibi_slopes.to(torch.float32)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])
    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_withcache_fp16_nomask(self):
        # [b,ms,ms]
        batch = 10
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 12
        embeddim = 128
        max_seq = 128*10
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = [128*3] * batch
        kv_seqLen = [128*10] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2012, "qSeqLen": q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 0, "kvHead": kv_head}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_MASK,
                             no_cache = True, tor = tor, q_seqlens = q_seqlens,
                             num_blocks = num_blocks, block_size = block_size)
        self.gen_out_tensor()
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k_cache, self.v_cache, self.block_tables,
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.float16)])
    
    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_withcache_bf16_nomask(self):
        # [b,ms,ms]
        batch = 10
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 12
        embeddim = 128
        max_seq = 128*10
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = [128*3] * batch
        kv_seqLen = [128*10] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2012, "qSeqLen": q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 0, "kvHead": kv_head}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_MASK,
                             no_cache = True, tor = tor, q_seqlens = q_seqlens,
                             num_blocks = num_blocks, block_size = block_size)
        self.gen_out_tensor()
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k_cache, self.v_cache, self.block_tables,
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])
    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_withcache_fp16_left_alibi(self):
        # [b,ms,ms]
        batch = 10
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 12
        embeddim = 128
        max_seq = 128*10
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = [128*3] * batch
        kv_seqLen = [128*10] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2012, "qSeqLen": q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 8, "kvHead": kv_head,
                    "isTriuMask" : 1, "alibiLeftAlign": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_PREFIX_BATCH,
                             no_cache = True, left_align = True, tor = tor, q_seqlens = q_seqlens,
                             num_blocks = num_blocks, block_size = block_size)
        self.gen_out_tensor()
        mask = np.ones((256,256)) * -(float("inf"))
        mask = np.triu(mask, 1)
        self.mask = self.bias[0, :256, :256] + mask
        self.mask = self.mask.to(torch.float16)
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k_cache, self.v_cache, self.block_tables,
                             self.mask, self.alibi_slopes.to(torch.float32)],
                            [torch.tensor(attention_out, dtype=torch.float16)])
    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_withcache_bf16_left_alibi(self):
        # [b,ms,ms]
        batch = 10
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 12
        embeddim = 128
        max_seq = 128*10
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = list(range(172, 172 + batch))
        kv_seqLen = list(range(300, 300 + batch))
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2012, "qSeqLen": q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 8, "kvHead": kv_head,
                    "isTriuMask" : 1, "alibiLeftAlign": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_PREFIX_BATCH,
                             no_cache = True, left_align = True, tor = tor, q_seqlens = q_seqlens,
                             num_blocks = num_blocks, block_size = block_size)
        self.gen_out_tensor()
        mask = np.ones((256,256)) * -(float("inf"))
        mask = np.triu(mask, 1)
        self.mask = self.bias[0, :256, :256] + mask
        self.mask = self.mask.to(torch.bfloat16)
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k_cache, self.v_cache, self.block_tables,
                             self.mask, self.alibi_slopes.to(torch.float32)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])
    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_withcache_alibi(self):
        # alibi
        batch = 4
        kv_head = 3        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 9
        embeddim = 128
        max_seq = 256
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = [128] * batch
        kv_seqLen = [256] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2012, "qSeqLen": q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 2,"kvHead": kv_head}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_PREFIX_BATCH,
                             tor = tor, q_seqlens = q_seqlens,
                             num_blocks = num_blocks, block_size = block_size)
        self.gen_out_tensor()
        self.mask = np.reshape(self.mask, (batch, heads, max_seq, max_seq))
        self.mask = self.mask.to(torch.bfloat16)
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k_cache, self.v_cache, self.block_tables,
                             self.mask, torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])
    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_withcache_bf16_128mask(self):
        # [b,ms,ms]
        batch = 1
        kv_head = 2      # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 10
        embeddim = 128
        max_seq = 256*10
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = list(range(172, 172 + batch))
        kv_seqLen = list(range(300, 300 + batch))
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2012, "qSeqLen": q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "kvHead": kv_head,
                    "isTriuMask" : 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, long_seq = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH_WITH_PREFIX,
                             no_cache = True, tor = tor, q_seqlens = q_seqlens,
                             num_blocks = num_blocks, block_size = block_size)
        self.gen_out_tensor()
        self.mask = self.mask.view(128, 128).to(data_type)
        attention_out = np.zeros_like(self.q.to(torch.float16))
        logging.info(f"self.q: {self.q}")
        logging.info(f"self.k_cache: {self.k_cache}")
        logging.info(f"self.v_cache: {self.v_cache}")
        logging.info(f"self.mask: {self.mask}")
        logging.info(f"self.block_tables.shape:{self.block_tables}")
        return self.execute([self.q, self.k_cache, self.v_cache, self.block_tables,
                             self.mask, torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])
    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_withcache_fp16_128mask(self):
        # [b,ms,ms]
        batch = 10
        kv_head = 1      # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 5
        embeddim = 128
        max_seq = 256*10
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = [256*3] * batch
        kv_seqLen = [256*10] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2012, "qSeqLen": q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "kvHead": kv_head,
                    "isTriuMask" : 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, long_seq = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH_WITH_PREFIX,
                             no_cache = True, tor = tor, q_seqlens = q_seqlens,
                             num_blocks = num_blocks, block_size = block_size)
        self.gen_out_tensor()
        self.mask = self.mask.view(128, 128).to(data_type)
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k_cache, self.v_cache, self.block_tables,
                             self.mask, torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])
    @op_test.only_910b
    def test_flash_attention_encoder_withcache_bf16(self):
        # unpad encoder
        batch = 1
        kv_head = 4        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 8192
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        q_seqlens = [8062]
        kv_seqLen = [8190]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        block_size = 128
        num_blocks = 1024
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2012, "qSeqLen":q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "kvHead": kv_head,"tor": tor}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16
        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH_WITH_PREFIX,
                             no_cache = True, tor = tor, q_seqlens = q_seqlens,
                             num_blocks = num_blocks, block_size = block_size)
        self.gen_out_tensor()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")
        attention_out = np.zeros_like(self.golden_out.to(torch.float16))
        return self.execute([self.q, self.k_cache, self.v_cache, self.block_tables,
                             self.mask.to(data_type), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])
    @op_test.only_910b
    def test_flash_attention_encoder_withcache_fp16(self):
        # unpad encoder
        batch = 1
        kv_head = 8        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 1280
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        q_seqlens = [128*3] * batch
        kv_seqLen = [128*10] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        block_size = 128
        num_blocks = 1024
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2012, "qSeqLen":q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16
        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH_WITH_PREFIX,
                             no_cache = True, tor = tor, q_seqlens = q_seqlens,
                             num_blocks = num_blocks, block_size = block_size)
        self.gen_out_tensor()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")
        attention_out = np.zeros_like(self.golden_out.to(torch.float16))
        return self.execute([self.q, self.k_cache, self.v_cache, self.block_tables,
                             self.mask.to(data_type), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.float16)])
    
    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_withcache_bf16_alibi_sqrt_multi_batch_with_tail_block(self):
        # [b,ms,ms]
        batch = 10
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 12
        embeddim = 128
        max_seq = 4096
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = list(range(172, 172 + batch))
        kv_seqLen = list(range(300, 300 + batch))
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2012, "qSeqLen": q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 7, "kvHead": kv_head,
                    "isTriuMask" : 1, "alibiLeftAlign": 0, "isAlibiMaskSqrt": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_PREFIX_BATCH,
                             no_cache = True, is_sqrt = True, tor = tor, q_seqlens = q_seqlens,
                             num_blocks = num_blocks, block_size = block_size)
        self.gen_out_tensor()
        self.alibi_slopes *= -1
        mask = np.ones((256,256)) * float("inf")
        mask = np.triu(mask, 1)
        self.mask = self.bias[:256, :256] * -1 + mask
        self.mask = self.mask.to(torch.bfloat16)
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k_cache, self.v_cache, self.block_tables,
                             self.mask, self.alibi_slopes.to(torch.float32)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_withcache_bf16_alibi_single_batch_without_tail_block(self):
        # [b,ms,ms]
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 12
        embeddim = 128
        max_seq = 4096
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = [128]
        kv_seqLen = [512]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2012, "qSeqLen": q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 6, "kvHead": kv_head,
                    "isTriuMask" : 1, "alibiLeftAlign": 0, "isAlibiMaskSqrt": 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_PREFIX_BATCH,
                             no_cache = True, is_sqrt = False, tor = tor, q_seqlens = q_seqlens,
                             num_blocks = num_blocks, block_size = block_size)
        self.gen_out_tensor()
        self.alibi_slopes *= -1
        mask = np.ones((256,256)) * float("inf")
        mask = np.triu(mask, 1)
        self.mask = self.bias[:256, :256] * -1 + mask
        self.mask = self.mask.to(torch.bfloat16)
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k_cache, self.v_cache, self.block_tables,
                             self.mask, self.alibi_slopes.to(torch.float32)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_withcache_bf16_alibi_single_batch_with_tail_block(self):
        # [b,ms,ms]
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 12
        embeddim = 128
        max_seq = 8192
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = [8062]
        kv_seqLen = [8190]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2012, "qSeqLen": q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 6, "kvHead": kv_head,
                    "isTriuMask" : 1, "alibiLeftAlign": 0, "isAlibiMaskSqrt": 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_PREFIX_BATCH,
                             no_cache = True, is_sqrt = False, tor = tor, q_seqlens = q_seqlens,
                             num_blocks = num_blocks, block_size = block_size)
        self.gen_out_tensor()
        self.alibi_slopes *= -1
        mask = np.ones((256,256)) * float("inf")
        mask = np.triu(mask, 1)
        self.mask = self.bias[:256, :256] * -1 + mask
        self.mask = self.mask.to(torch.bfloat16)
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k_cache, self.v_cache, self.block_tables,
                             self.mask, self.alibi_slopes.to(torch.float32)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_withcache_bf16_alibi_multi_batch_without_tail_block(self):
        # [b,ms,ms]
        batch = 10
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 12
        embeddim = 128
        max_seq = 4096
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = [128, 128, 128, 128, 128, 256, 256, 256, 256, 256]
        kv_seqLen = [256, 256, 256, 256, 256, 512, 512, 512, 512, 512]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2012, "qSeqLen": q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 6, "kvHead": kv_head,
                    "isTriuMask" : 1, "alibiLeftAlign": 0, "isAlibiMaskSqrt": 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_PREFIX_BATCH,
                             no_cache = True, is_sqrt = False, tor = tor, q_seqlens = q_seqlens,
                             num_blocks = num_blocks, block_size = block_size)
        self.gen_out_tensor()
        self.alibi_slopes *= -1
        mask = np.ones((256,256)) * float("inf")
        mask = np.triu(mask, 1)
        self.mask = self.bias[:256, :256] * -1 + mask
        self.mask = self.mask.to(torch.bfloat16)
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k_cache, self.v_cache, self.block_tables,
                             self.mask, self.alibi_slopes.to(torch.float32)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_withcache_bf16_alibi_multi_batch_with_tail_block(self):
        # [b,ms,ms]
        batch = 10
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 12
        embeddim = 128
        max_seq = 4096
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = list(range(172, 172 + batch))
        kv_seqLen = list(range(300, 300 + batch))
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2012, "qSeqLen": q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 6, "kvHead": kv_head,
                    "isTriuMask" : 1, "alibiLeftAlign": 0, "isAlibiMaskSqrt": 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_PREFIX_BATCH,
                             no_cache = True, is_sqrt = False, tor = tor, q_seqlens = q_seqlens,
                             num_blocks = num_blocks, block_size = block_size)
        self.gen_out_tensor()
        self.alibi_slopes *= -1
        mask = np.ones((256,256)) * float("inf")
        mask = np.triu(mask, 1)
        self.mask = self.bias[:256, :256] * -1 + mask
        self.mask = self.mask.to(torch.bfloat16)
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k_cache, self.v_cache, self.block_tables,
                             self.mask, self.alibi_slopes.to(torch.float32)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_withcache_bf16_alibi_compress_128_single_batch_without_tail_block(self):
        # [b,ms,ms]
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 12
        embeddim = 128
        max_seq = 4096
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = [128]
        kv_seqLen = [512]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2012, "qSeqLen": q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 9, "kvHead": kv_head,
                    "isTriuMask" : 1, "alibiLeftAlign": 0, "isAlibiMaskSqrt": 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_PREFIX_BATCH,
                             no_cache = True, is_sqrt = False, tor = tor, q_seqlens = q_seqlens,
                             num_blocks = num_blocks, block_size = block_size)
        self.gen_out_tensor()
        self.alibi_slopes *= -1
        self.mask = self.mask[0, :, :, :128].to(torch.bfloat16)
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k_cache, self.v_cache, self.block_tables,
                             self.mask, self.alibi_slopes.to(torch.float32)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_withcache_bf16_alibi_compress_128_single_batch_with_tail_block(self):
        # [b,ms,ms]
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 12
        embeddim = 128
        max_seq = 8192
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = [8062]
        kv_seqLen = [8190]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2012, "qSeqLen": q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 9, "kvHead": kv_head,
                    "isTriuMask" : 1, "alibiLeftAlign": 0, "isAlibiMaskSqrt": 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_PREFIX_BATCH,
                             no_cache = True, is_sqrt = False, tor = tor, q_seqlens = q_seqlens,
                             num_blocks = num_blocks, block_size = block_size)
        self.gen_out_tensor()
        self.alibi_slopes *= -1
        self.mask = self.mask[0, :, :, :128].to(torch.bfloat16)
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k_cache, self.v_cache, self.block_tables,
                             self.mask, self.alibi_slopes.to(torch.float32)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_withcache_bf16_alibi_compress_128_multi_batch_without_tail_block(self):
        # [b,ms,ms]
        batch = 10
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 12
        embeddim = 128
        max_seq = 4096
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = [128, 128, 128, 128, 128, 256, 256, 256, 256, 256]
        kv_seqLen = [256, 256, 256, 256, 256, 512, 512, 512, 512, 512]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2012, "qSeqLen": q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 9, "kvHead": kv_head,
                    "isTriuMask" : 1, "alibiLeftAlign": 0, "isAlibiMaskSqrt": 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_PREFIX_BATCH,
                             no_cache = True, is_sqrt = False, tor = tor, q_seqlens = q_seqlens,
                             num_blocks = num_blocks, block_size = block_size)
        self.gen_out_tensor()
        self.alibi_slopes *= -1
        self.mask = self.mask[0, :, :, :128].to(torch.bfloat16)
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k_cache, self.v_cache, self.block_tables,
                             self.mask, self.alibi_slopes.to(torch.float32)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_withcache_bf16_alibi_compress_128_multi_batch_with_tail_block(self):
        # [b,ms,ms]
        batch = 10
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 12
        embeddim = 128
        max_seq = 4096
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = list(range(172, 172 + batch))
        kv_seqLen = list(range(300, 300 + batch))
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2012, "qSeqLen": q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 9, "kvHead": kv_head,
                    "isTriuMask" : 1, "alibiLeftAlign": 0, "isAlibiMaskSqrt": 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_ALIBI_WITH_PREFIX_BATCH,
                             no_cache = True, is_sqrt = False, tor = tor, q_seqlens = q_seqlens,
                             num_blocks = num_blocks, block_size = block_size)
        self.gen_out_tensor()
        self.alibi_slopes *= -1
        self.mask = self.mask[0, :, :, :128].to(torch.bfloat16)
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k_cache, self.v_cache, self.block_tables,
                             self.mask, self.alibi_slopes.to(torch.float32)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_case_fa_razor_fusion_norm(self):
        # [b,ms,ms]
        batch = 2
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 1
        embeddim = 128
        razorLen = 512
        tor = 1.0
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        tileQ = 2
        tileKv = 3
        textQLen = 128
        textKvLen = 256

        is_razor_fusion = 1
        max_seq = max(tileQ * razorLen + textQLen, tileKv * razorLen + textKvLen)
        preTokens = 128
        nextTokens = 256
        q_seqLen = [tileQ * razorLen + textQLen] * batch
        kv_seqLen = [tileKv * razorLen + textKvLen] * batch
        
        dynamic_batch = False
        baseM = kv_seqLen[0] if kv_seqLen[0] <= 128 else 128
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2014, "qSeqLen": kv_seqLen, "kvSeqLen": kv_seqLen, "qSeqLen": q_seqLen,  "headSize": heads, "tor": tor,
                    "tileQ": tileQ,  "tileKv": tileKv,  "razorLen": razorLen,  "textQLen": textQLen,  "textKvLen": textKvLen,
                    "preTokens": preTokens, "nextTokens": nextTokens, 
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 0, "kvHead": kv_head,
                    "isTriuMask" : 0}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen, is_razor_fusion = is_razor_fusion,
                             baseM = baseM, razorLen = razorLen, textQLen = textQLen, textKvLen = textKvLen, 
                             tileQ = tileQ, tileKv = tileKv, preTokens = preTokens, nextTokens = nextTokens,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = True, is_mask = False, q_seqlens=q_seqLen,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_RAZOR_FUSION,
                             no_cache = True, tor = tor)
        self.gen_out_tensor()

        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k, self.v],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_flash_attention_mla_fp16_deepseek_test1(self):
        batch = 1
        kv_head = 128
        isdecoder = 0
        heads = 128
        embeddim = 192
        embeddimV = 128
        max_seq = 3072
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [3072] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 1, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV,"headSize": heads, "tor": tor, "maskType": 1, "isTriuMask": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_HEAD, long_seq= True, tor = tor)
        self.gen_out_tensor()
        self.mask = self.mask.view(batch, 128, 128)[0, :, :]
        self.q = self.q.reshape(3072, 128, 192)
        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.golden_out.to(torch.float16))
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=data_type)])
        
    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_withcache_fp16_maskfree(self):
        # [b,ms,ms]
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 1
        embeddim = 128
        max_seq = 128*100
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = [128*6+16] * batch
        kv_seqLen = [128*8+16] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2012, "qSeqLen": q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 10, "kvHead": kv_head,
                    "isTriuMask": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH_WITH_PREFIX,
                             no_cache = True, tor = tor, q_seqlens = q_seqlens,
                             num_blocks = num_blocks, block_size = block_size, is_triu_mask = True, is_mask=True)
        self.gen_out_tensor()
        attention_out = np.zeros_like(self.q.to(torch.float16))
        

        return self.execute([self.q, self.k_cache, self.v_cache, self.block_tables,
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.float16)])
        
    @op_test.only_910b
    def test_flash_attention_case_fa_encoder_withcache_bf16_maskfree(self):
        # [b,ms,ms]
        batch = 2
        kv_head = 2       # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 4
        embeddim = 128
        max_seq = 128*100
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        q_seqlens = [128*26+13] * batch
        kv_seqLen = [128*28+13] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        dynamic_batch = False
        block_size = 128
        num_blocks = 1024
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 2012, "qSeqLen": q_seqlens, "kvSeqLen": kv_seqLen, "headSize": heads, "tor": tor,
                    "isClamp" : is_clamp, "clampMin" : clamp_min, "clampMax" : clamp_max, "maskType": 10, "kvHead": kv_head,
                    "isTriuMask": 1}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH_WITH_PREFIX,
                             no_cache = True, tor = tor, q_seqlens = q_seqlens,
                             num_blocks = num_blocks, block_size = block_size, is_triu_mask = True, is_mask=True)
        self.gen_out_tensor()
        attention_out = np.zeros_like(self.q.to(torch.float16))
        return self.execute([self.q, self.k_cache, self.v_cache, self.block_tables,
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=torch.bfloat16)])
    
    @op_test.only_910b
    def test_flash_attention_mla_swa_bf16(self):
        # unpad encoder
        batch = 16
        kv_head = 8        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 512
        max_seq = 2048
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [500, 1024, 1500, 2048] * 4
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        window_size = 128 * 7 + 58
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 1, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV,"headSize": heads, "tor": tor,
                    "maskType": 4, "windowSize": window_size}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False, window_size = window_size,
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
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=data_type)])

    @op_test.only_910b
    def test_flash_attention_mla_swa_fp16(self):
        # unpad encoder
        batch = 16
        kv_head = 8        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 512
        max_seq = 2048
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [500, 1024, 1500, 2048] * 4
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        window_size = 128 * 7 + 58
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 1, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV,"headSize": heads, "tor": tor,
                    "maskType": 4, "windowSize": window_size}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False, window_size = window_size,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH, tor = tor)
        self.gen_out_tensor()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.golden_out)
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])
    
    @op_test.only_910b
    def test_flash_attention_mla_swa_cmp_bf16(self):
        # unpad encoder
        batch = 16
        kv_head = 8        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 512
        max_seq = 2048
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [500, 1024, 1500, 2048] * 4
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        window_size = 128 * 7 + 58
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 1, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV,"headSize": heads, "tor": tor,
                    "maskType": 5, "windowSize": window_size}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch, is_compress = True,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False, window_size = window_size,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH, tor = tor)
        self.gen_out_tensor()
        if self.is_compress:
            self.mask = self.gen_swa_cmp(max_seq, window_size)
            self.mask = np.reshape(self.mask, (512, 512))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.golden_out.to(torch.float16))
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out, dtype=data_type)])

    @op_test.only_910b
    def test_flash_attention_mla_swa_cmp_fp16(self):
        # unpad encoder
        batch = 16
        kv_head = 8        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 512
        max_seq = 2048
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [500, 1024, 1500, 2048] * 4
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        window_size = 128 * 7 + 58
        OP_NAME = "UnpadFlashAttentionOperation"
        OP_PARAM = {"type": 1, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV,"headSize": heads, "tor": tor,
                    "maskType": 5, "windowSize": window_size}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd] * 12)
        self.set_output_formats([self.format_nd])
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch, is_compress = True,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False, window_size = window_size,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH, tor = tor)
        self.gen_out_tensor()
        if self.is_compress:
            self.mask = self.gen_swa_cmp(max_seq, window_size)
            self.mask = np.reshape(self.mask, (512, 512))

        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

        attention_out = np.zeros_like(self.golden_out)
        return self.execute([self.q, self.k, self.v, self.layer_id, self.mask.to(data_type), torch.tensor([], dtype=torch.float),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.int32),
                             torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.float)],
                            [torch.tensor(attention_out).half()])

if __name__ == '__main__':
    unittest.main()