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
sys.path.append('../')
sys.path.append('../..')
# sys.path.append(f"{os.environ['ATB_HOME_PATH']}/tests/atbopstest/pythontest")
# import op_test
import torch
from enum import Enum
import random
# from precision_calcu import *
 
import json
import random
import torch_npu
 
np.random.seed(0)
 
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402
import relay_attention.golden_compare_cv as golden_compare_cv
 
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
 
class TestFlashAttention():
 
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
                       kv_seqLen = [], is_clamp = 0, clamp_min = 0,
                       clamp_max = 0, data_type = torch.float16, op_type = 0, mask_type = 0,
                       no_cache = False, long_seq = False, is_triu_mask = False, is_multi_layer = False,
                       is_sqrt = False, left_align = False, scaleType = ScaleType.SCALE_TOR.value, fav3 = False,
                       tor = 1, bnsd = False, is_compress = False, share_len = 0):
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
        self.fav3 = False
        if self.embeddimv == 0:
            self.embeddimv = self.embeddim
        if is_decoder:
            self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, [1] * batch)
        else:
            self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, kv_seqLen)
        self.kv_seqlen, self.kv_ntokens = self.gen_seq_len(batch, kv_seqLen)
        # gen intensor for fa kernel
        self.share_seqlen = []
        for i in range(share_len):
            self.share_seqlen.append(np.random.randint(1, self.max_seq - 1))
        # share_seqlen = np.array(share_seqlen, dtype=np.int32)
        self.share_index = []
        for i in range(batch):
            if(i % 3 == 0):
                self.share_index.append(-1)
            else:
                self.share_index.append(np.random.randint(0, share_len))
        self.q_max_seq = np.max(self.q_seqlen)
        self.kv_max_seq = np.max(self.kv_seqlen)
        q = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.q_ntokens, heads * self.embeddim)))
        
 
        self.q = q.to(data_type)
 
        self.k_list = []
        self.v_list = []
        self.kShare_list = []
        self.vShare_list = []
        self.sharedGroupsCount = share_len
        for i in range(self.batch):
            self.k_list.append(np.random.uniform(-1.0, 1.0, size=(self.max_seq, kv_head * self.embeddim)))
            self.v_list.append(np.random.uniform(-1.0, 1.0, size=(self.max_seq, kv_head * self.embeddimv)))
        for i in range(self.sharedGroupsCount):
            self.kShare_list.append(np.random.uniform(-1.0, 1.0, size=(self.max_seq, kv_head * self.embeddim)))
            self.vShare_list.append(np.random.uniform(-1.0, 1.0, size=(self.max_seq, kv_head * self.embeddimv)))
        self.k = torch.Tensor(self.k_list).to(data_type).cpu()
        self.v = torch.Tensor(self.v_list).to(data_type).cpu()
        self.kshare = torch.Tensor(self.kShare_list).to(data_type).cpu()
        self.vshare = torch.Tensor(self.vShare_list).to(data_type).cpu()
        # self.k = torch.cat(self.k_list, 1).cpu()
        # self.v = torch.cat(self.v_list, 1).cpu()
        # self.kshare = torch.cat(self.kShare_list, 1).cpu()
        # self.vshare = torch.cat(self.vShare_list, 1).cpu()
        for i in range(self.batch):
            self.k_list[i] = torch.from_numpy(self.k_list[i]).to(data_type).squeeze().npu()
            self.v_list[i] = torch.from_numpy(self.v_list[i]).to(data_type).squeeze().npu()
        for i in range(self.sharedGroupsCount):
            self.kShare_list[i] = torch.from_numpy(self.kShare_list[i]).to(data_type).squeeze().npu()
            self.vShare_list[i] = torch.from_numpy(self.vShare_list[i]).to(data_type).squeeze().npu()
        # self.k = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(batch, self.max_seq, kv_head * self.embeddim))).to(data_type)
        # self.v = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(batch, self.max_seq, kv_head * self.embeddimv))).to(data_type)
        # self.kshare = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(share_len, self.max_seq, kv_head * self.embeddim))).to(data_type)
        # self.vshare = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(share_len, self.max_seq, kv_head * self.embeddimv))).to(data_type)
        self.gen_mask(batch, heads, data_type, mask_type, window_size, is_compress, cache_type)
        # logging.debug("**********data gen shape***********")
        # logging.debug(f"q shape: {self.q.shape}")
        # logging.debug(f"k shape: {self.k.shape}")
        # logging.debug(f"v shape: {self.v.shape}")
        # logging.debug(f"mask shape: {self.mask.shape}")
 
    def gen_mask(self, batch, heads, data_type, mask_type, window_size, is_compress, cache_type=0):
        import random
        q_max_seq = self.max_seq
        kv_max_seq = self.max_seq
        mask_type_dict = {
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
        if select_zero:
            mask.flat[zero_indice] = 0
        self.mask = torch.from_numpy(mask).to(torch.float32)
        self.post_mask_coff = post_mask_coff
        self.pre_mask_coff = pre_mask_coff
 
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
        k_share = self.kshare
        v_share = self.vshare
        if self.fav3:
            q = self.q_int8
            k = self.k_int8
            v = self.v_int8
        q_ntokens = self.q_ntokens
        kv_ntokens = self.kv_ntokens
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
            kv_s_share = 0
            if(self.share_index[idx] != -1):
                kv_s_share = self.share_seqlen[self.share_index[idx]]
            kv_s = kv_seqlen[idx] - kv_s_share
            if(kv_s == 0):
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
            q_slice = q[q_offset:q_offset + q_s][:]
            q_slice = q_slice.view(q_s, heads, embed)
            q_slice = torch.permute(q_slice, (1, 0, 2))
            k_slice = k[idx][:kv_s][:]
            k_slice_share = k_share[self.share_index[idx]][:kv_s_share][:]
            v_slice = v[idx][:kv_s][:]
            v_slice_share = v_share[self.share_index[idx]][:kv_s_share][:]
            k_slice = torch.cat((k_slice, k_slice_share),0)
            v_slice = torch.cat((v_slice, v_slice_share),0)
            kv_s = kv_seqlen[idx]
            k_slice = k_slice.view(kv_s, kv_head, embed)
            k_slice_t = torch.permute(k_slice, (1, 2, 0))
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
 
            if self.scaleType != ScaleType.SCALE_TOR.value:
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
            if is_mask:
                score = score + self.mask_info[1](self.mask, idx, q_s, kv_s) * self.post_mask_coff
 
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
        result_double = compare_cv(self.golden_out_true, golden_tensors[0], out_tensors[0])
        return (result_double or result_single)
 
    # @op_test.only_910b
    def test_relay_attention_case1(self):
        # decoder
        batch = 1
        kv_head = 8        # kv_head num
        isdecoder = 1       # prefill or decoder
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 256
        share_len = 5
        tor = 1.0 / math.sqrt(1.0 * embeddim)
        dynamic_batch = False
        kv_seqLen = [114] * batch
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        data_type = torch.float16
        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = 14, mask_type = MASK_TYPE_NO_MASK, tor = tor, share_len = share_len)
        # OP_NAME = "UnpadFlashAttentionOperation"
        # OP_PARAM = {"type": 14, "qSeqLen":[1] * batch, "kvSeqLen": kv_seqLen, "kvHead":kv_head, "headSize": heads, "tor": tor, "maskType": 0, "kvShareMap":self.share_index, "kvShareLen":self.share_seqlen}
        # self.set_param(OP_NAME, OP_PARAM)
        # self.set_input_formats([self.format_nd] * 7)
        # self.set_output_formats([self.format_nd])
        self.gen_out_tensor()
 
        # logging.debug("**********input shape***********")
        # logging.debug(f"q shape: {self.q.shape}")
        # logging.debug(f"k shape: {self.k.shape}")
        # logging.debug(f"v shape: {self.v.shape}")
        # logging.debug(f"k shape: {self.kshare.shape}")
        # logging.debug(f"v shape: {self.vshare.shape}")
        # logging.debug(f"mask shape: {self.mask.shape}")
        attention_out = np.zeros_like(self.q)
        # self.execute([self.q, self.k, self.v, self.kshare, self.vshare, torch.tensor([], dtype=data_type)],
        #                 [torch.tensor(attention_out).half()])
        self.mask = self.q
        ret = [self.q.npu(), self.k.npu(), self.v.npu(), self.kshare.npu(), self.vshare.npu(), self.mask.npu(), self.q_seqlen, self.kv_seqLen, self.share_index, self.share_seqlen, torch.tensor(self.golden_out, dtype=torch.bfloat16), self.k_list, self.v_list,self.kShare_list, self.vShare_list]
        return ret
 
data_generator = TestFlashAttention()
 
data_all = data_generator.test_relay_attention_case1()
data = data_all[:11]
data_tensorlist = data_all[-4:]
param_seqlen = data[6]
data[6] = torch.from_numpy(np.array(data[6]).astype(np.int32))
param_kvseqlen = data[7]
data[7] = torch.from_numpy(np.array(data[7]).astype(np.int32))
param_kvsharemap = data[8]
data[8] = torch.from_numpy(np.array(data[8]).astype(np.int32))
param_kvsharelen = data[9]
data[9] = torch.from_numpy(np.array(data[9]).astype(np.int32))
 
in_tensors = [tensor.npu().contiguous() for tensor in data]
tor = 1.0 / math.sqrt(1.0 * 128)
OP_NAME = "RelayAttentionOperation"
PARAM = json.dumps({"headNum": 32, "qkScale": tor, "kvHeadNum": 8, "maskType": 0})
RUN_PARAM = json.dumps({"tokenOffset": param_kvseqlen, "seqLen": param_seqlen, "kvShareMap": param_kvsharemap, "kvShareLen": param_kvsharelen})
 
class TestRelayAttention(operation_test.OperationTest):
    def golden_calc(self, input_tensors):
        return [in_tensors[10]]
 
    def golden_compare(self, out_tensor, golden_out_tensor):
        result_single = data_generator.compare_output_data(out_tensor[0].cpu(), golden_out_tensor[0].cpu(), [0.001, 0.001, 0.005, 0.005])
        result_double = golden_compare_cv.compare_cv(data_generator.golden_out_true, golden_out_tensor[0].cpu(), out_tensor[0].cpu())
        # result_double = golden_compare_cv.compare_cv(golden_out_tensor[1], golden_out_tensor[0], out_tensor[0].cpu())
        return (result_double or result_single)
        # if data_generator.is_int8_flag:
        #     result_double = compare_cv(data_generator.golden_out_true, golden_out_tensor.cpu(), out_tensor.cpu())
        #     return (result_double or result_single)
        # else:
        #     result_double = compare_cv(data_generator.golden_out_true, golden_out_tensor.cpu(), out_tensor.cpu())
        #     return (result_double or result_single)
 
    def test(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        print(PARAM)
        self.execute_with_param_and_tensor_list(OP_NAME, PARAM, RUN_PARAM,
                     [in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4], in_tensors[5], in_tensors[6], in_tensors[7], in_tensors[8], in_tensors[9]],
                     [data_tensorlist[0], data_tensorlist[1], data_tensorlist[2], data_tensorlist[3]], ["key", "value", "keyShare", "valueShare"])
        # self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [
        #     in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4], in_tensors[5], in_tensors[6], in_tensors[7], in_tensors[8], in_tensors[9]
        # ])
 
if __name__ == '__main__':
    unittest.main()