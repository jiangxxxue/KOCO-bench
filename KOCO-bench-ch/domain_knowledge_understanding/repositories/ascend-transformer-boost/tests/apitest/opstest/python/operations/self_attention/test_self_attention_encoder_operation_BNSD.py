# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import json
import random
import torch_npu
import logging
import unittest
import math
import numpy as np
import sys
import os
import torch
from enum import Enum
 
np.random.seed(43)
 
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

class ScaleType(Enum):
    SCALE_TOR = 0
    SCALE_LOGN = 1
    SCALE_LOGN_FP32 = 2
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
 
class TestFlashAttentionDataGenerator(operation_test.OperationTest):
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
                       is_mask=True, is_decoder=False, is_alibi=False, alibi_dim=4,
                       batch = 1, kv_head = 1, heads = 1, embeddim = 128, embeddimv = 0, max_seq = 2048,
                       kv_seqLen = [], is_clamp = 0, clamp_min = 0, is_splitm = False,
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
                self.k = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, batch, self.kv_max_seq, kv_head * self.embeddim))).to(data_type)
                self.v = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, batch, self.kv_max_seq, kv_head * self.embeddimv))).to(data_type)
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
        
    def gen_seq_len(self, batch, seq_len):
        ntokens = sum(seq_len)
        return seq_len, ntokens
 
 
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
        self.mask = torch.from_numpy(mask).to(torch.float32)
        self.post_mask_coff = post_mask_coff
        self.pre_mask_coff = pre_mask_coff
             
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
        maxQSeqlen = max(q_seqlen)
        obsnd = torch.zeros(batch, maxQSeqlen, heads, embedv)
        out_true_bnsd = torch.zeros(batch, maxQSeqlen, heads, embedv)
        # import pdb;pdb.set_trace()
        maxKvSeqlen = max(kv_seqlen)
        # kbsnd=k.view(layer_id+1,batch,kv_seqlen[0],kv_head,embed)
        # vbsnd=v.view(layer_id+1,batch,kv_seqlen[0],kv_head,embedv)
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
            # print(k.shape, kv_s)
            # kv_s = max_seq
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

            if is_mask:
                score = score + self.mask_info[1](self.mask, idx, q_s, kv_s) * self.post_mask_coff
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
            print(q_s)
            print(obsnd.shape)
            print(o.shape)
            for i in range(0,q_s):
                obsnd[idx][i] = o[i]
                out_true_bnsd[idx][i]=out_true[i]
            q_offset += q_s
            k_offset += kv_s
            v_offset += kv_s
        obnsd = torch.permute(obsnd, (0, 2, 1,3))
        out_true_bnsd = torch.permute(out_true_bnsd, (0, 2, 1,3))
        # qbsnd = torch.zeros(batch, q_seqlen[0], heads, embed)
        # b heads q-sel,emd
        self.qbnsd = torch.permute(qbsnd, (0, 2, 1, 3)).to(self.data_type)
        # kbsnd=k.view(layer_id+1,batch,kv_seqlen[0],kv_head,embed)
        # layer,
        self.kbnsd = torch.permute(kbsnd, (0, 1, 3, 2, 4)).to(self.data_type)
        # vbsnd=v.view(layer_id+1,batch,kv_seqlen[0],kv_head,embedv)
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
 
    def group_mm_torch(self, heads, group_num, A, B):
        group_head = heads // group_num
        score = None
        for i in range(group_num):
            group_score = torch.matmul(A[i * group_head: (i + 1) * group_head, :, :].to(torch.float32), B[i:(i + 1), :, :].to(torch.float32))
            if score is None:
                score = group_score
            else:
                score = torch.cat((score, group_score), 0)
        return score
 
    def golden_calc(self, in_tensors):
        golden_out = torch.tensor(self.golden_out).half().npu()
        return [golden_out]
 
    def golden_compare(self, out_tensors, golden_tensors):
        # print("out_tensors:", out_tensors)
        # print("golden_tensors:", golden_tensors)
        return self.compare_output_data(out_tensors.half(), golden_tensors.half(), [0.001, 0.001, 0.003, 0.003, 0.005, 0.005])

    def test_flash_attention_encoder_bnsd_bf16(self):
        # unpad encoder
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        batch = 2
        kv_head = 20       # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 20        # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 4096
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        q_seqLen = [255, 255]
        kv_seqLen = [1024, 1024]
        # kv_seqLen=[1] *batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "SelfAttentionOperation"
        
        data_type = torch.bfloat16
        scaleType = ScaleType.SCALE_TOR.value
        
        
        self.set_data_params(dynamic_batch = dynamic_batch, no_cache=True,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, q_seqlens = q_seqLen,kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,scaleType = scaleType, is_splitm = True,
                             op_type = 10, mask_type = MASK_TYPE_NO_MASK, is_mask = False, tor = tor)
        self.gen_out_tensor_bnsd()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))
        
        param_seqlen = np.concatenate((self.q_seqlen, self.kv_seqlen), 0).tolist()
        self.kv_seqlen = torch.from_numpy(np.array(self.kv_seqlen)).to(torch.int32)
        self.q_seqlen = torch.from_numpy(np.array(self.q_seqlen)).to(torch.int32)
        seqlen = torch.stack([self.q_seqlen, self.kv_seqlen])
        PARAM = json.dumps({"headNum": heads, "kvHeadNum":kv_head, "qkScale": 1.0 / math.sqrt(1.0 * embeddim),
                            "maskType": 0, "inputLayout":1, "calcType":3})
        RUN_PARAM = json.dumps({"seqLen": param_seqlen, "maskType": 0})
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [
                self.qbnsd.npu(), self.kbnsd[0].npu(), self.vbnsd[0].npu(), seqlen.npu()
            ])
        
    def test_flash_attention_encoder_bnsd_same_seqlen_bf16(self):
        # unpad encoder
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        batch = 2
        kv_head = 20       # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 20        # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 4096
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        q_seqLen = [255, 255]
        kv_seqLen = [255, 255]
        # kv_seqLen=[1] *batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "SelfAttentionOperation"
        
        data_type = torch.bfloat16
        scaleType = ScaleType.SCALE_TOR.value
        
        
        self.set_data_params(dynamic_batch = dynamic_batch, no_cache=True,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, q_seqlens = q_seqLen,kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,scaleType = scaleType, is_splitm = True,
                             op_type = 10, mask_type = MASK_TYPE_NO_MASK, is_mask = False, tor = tor)
        self.gen_out_tensor_bnsd()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))
        
        param_seqlen = self.q_seqlen
        seqlen = torch.from_numpy(np.array(self.q_seqlen)).to(torch.int32)
        PARAM = json.dumps({"headNum": heads, "kvHeadNum":kv_head, "qkScale": 1.0 / math.sqrt(1.0 * embeddim),
                            "maskType": 0, "inputLayout":1, "calcType":3})
        RUN_PARAM = json.dumps({"seqLen": param_seqlen, "maskType": 0})
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [
                self.qbnsd.npu(), self.kbnsd[0].npu(), self.vbnsd[0].npu(), seqlen.npu()
            ])
        
    def test_flash_attention_encoder_bnsd_same_seqlen2_bf16(self):
        # unpad encoder
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        batch = 2
        kv_head = 20       # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 20        # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 4096
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        q_seqLen = [255, 255]
        kv_seqLen = [255, 255]
        # kv_seqLen=[1] *batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "SelfAttentionOperation"
        
        data_type = torch.bfloat16
        scaleType = ScaleType.SCALE_TOR.value
        
        
        self.set_data_params(dynamic_batch = dynamic_batch, no_cache=True,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, q_seqlens = q_seqLen,kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,scaleType = scaleType, is_splitm = True,
                             op_type = 10, mask_type = MASK_TYPE_NO_MASK, is_mask = False, tor = tor)
        self.gen_out_tensor_bnsd()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))
        
        param_seqlen = np.concatenate((self.q_seqlen, self.kv_seqlen), 0).tolist()
        self.kv_seqlen = torch.from_numpy(np.array(self.kv_seqlen)).to(torch.int32)
        self.q_seqlen = torch.from_numpy(np.array(self.q_seqlen)).to(torch.int32)
        seqlen = torch.stack([self.q_seqlen, self.kv_seqlen])
        PARAM = json.dumps({"headNum": heads, "kvHeadNum":kv_head, "qkScale": 1.0 / math.sqrt(1.0 * embeddim),
                            "maskType": 0, "inputLayout":1, "calcType":3})
        RUN_PARAM = json.dumps({"seqLen": param_seqlen, "maskType": 0})
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [
                self.qbnsd.npu(), self.kbnsd[0].npu(), self.vbnsd[0].npu(), seqlen.npu()
            ])

if __name__ == '__main__':
    unittest.main()