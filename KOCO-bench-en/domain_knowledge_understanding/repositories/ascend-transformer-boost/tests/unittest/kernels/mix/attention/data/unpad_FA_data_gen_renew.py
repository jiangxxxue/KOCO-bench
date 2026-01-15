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
import sys
import os
import torch
import random

MASK_TYPE_NO_MASK = 0
MASK_TYPE_NO_HEAD = 1
MASK_TYPE_NO_BATCH = 2
MASK_TYPE_ALIBI_WITH_BATCH = 3
MASK_TYPE_ALIBI_NO_BATCH = 4
MASK_TYPE_NO_HEAD_DECODER = 5

class FlashAttentionDataGen():
    def gen_seq_len_array(self, batch, max_seq, variate_seq):
        if variate_seq:
            num = max_seq // 16
            seqlen_aligned_arange = np.arange(1, min(batch + 1, num + 1)) * 16
            if batch > num:
                seqlen_aligned_remain = np.random.randint(1, max_seq, size=(batch - num))
                seqlen_aligned_remain[:] = ((seqlen_aligned_remain[:] + 15) // 16) * 16
                seqlen_aligned = np.concatenate((seqlen_aligned_arange, seqlen_aligned_remain), 0)
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

        return seqlen
    
    def gen_batch_state(self, batch):
        batch_state = np.ones(batch, dtype = np.int32)
        zero_indice = random.choices(range(batch), k = batch // 3)
        batch_state[zero_indice] = 0
        batch_state[0] = 1
        return batch_state

    def set_data_params(self, dynamic_batch=False,
                       is_mask=True, is_decoder=False, is_alibi=False, alibi_dim=4, 
                       batch = 1, kv_head = 1, heads = 1, embeddim = 128, max_seq = 2048,
                       kv_seqlen = 0, is_clamp = 0, clamp_min = 0,
                       clamp_max = 0, data_type = np.float16, mask_type = 0, 
                       no_cache = False, long_seq = False, is_triu_mask = False,
                       is_sqrt = False, left_align = False, variate_seq=False,
                       relayAttention = False, shareIdx = None, shareLen = None):
        self.batch = batch
        self.dynamic_batch = dynamic_batch
        if dynamic_batch:
            self.batch_state = self.gen_batch_state(batch)
        else:
            self.batch_state = np.ones(batch, dtype = np.int32)
        self.is_mask = is_mask
        self.is_decoder = is_decoder
        self.is_alibi = is_alibi
        self.alibi_dim = alibi_dim 
        self.variate_seq = variate_seq
        self.max_seq = max_seq
        self.kv_head = kv_head
        self.heads = heads
        self.embeddim = embeddim    
        kv_seqLen = self.gen_seq_len_array(batch, kv_seqlen, variate_seq)
        self.is_clamp = is_clamp
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.data_type = data_type
        self.no_cache = no_cache
        self.long_seq = long_seq
        self.mask_type = mask_type
        self.is_triu_mask = is_triu_mask
        self.is_sqrt = is_sqrt
        self.left_align = left_align
        self.relayAttention = relayAttention
        self.shareIdx = shareIdx
        self.shareLen = shareLen
        if is_decoder:
            self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, [1] * batch)
        else:
            self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, kv_seqLen)
        self.kv_seqlen, self.kv_ntokens = self.gen_seq_len(batch, kv_seqLen)
        # gen intensor for fa kernel
        self.layer_id = np.array([0], dtype=np.int32)
        self.q_max_seq = np.max(self.q_seqlen)
        self.kv_max_seq = np.max(self.kv_seqlen)
        q = np.random.uniform(-1.0, 1.0, size=(self.q_ntokens, heads * self.embeddim)).astype(data_type)
        tor = np.float32(1.0 / math.sqrt(1.0 * self.embeddim))
        self.q = (q * tor).astype(data_type)
        self.k = np.random.uniform(-1.0, 1.0, size=(1, batch, self.max_seq, kv_head * self.embeddim)).astype(data_type)
        self.v = np.random.uniform(-1.0, 1.0, size=(1, batch, self.max_seq, kv_head * self.embeddim)).astype(data_type)
        if self.relayAttention:
            self.kv_share_max_seq = np.max(self.shareLen)
            self.kshare = np.random.uniform(-1.0, 1.0, size=(len(self.shareLen), self.kv_share_max_seq, kv_head * self.embeddim)).astype(data_type)
            self.vshare = np.random.uniform(-1.0, 1.0, size=(len(self.shareLen), self.kv_share_max_seq, kv_head * self.embeddim)).astype(data_type)
            logging.debug(f"kshare shape: {self.kshare.shape}")
            logging.debug(f"vshare shape: {self.vshare.shape}")
        self.gen_mask(batch, heads, data_type, mask_type)

        logging.debug("**********data gen shape***********")
        logging.debug(f"q shape: {self.q.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")
        logging.debug(f"layer_id shape: {self.layer_id.shape}")
        logging.debug(f"mask shape: {self.mask.shape}")

    def get_alibi_slopes(self, n_heads):
        n = 2 ** math.floor(math.log2(n_heads))
        m0 = 2.0 ** (-8.0 / n)
        slopes = np.power(m0, np.arange(1, n + 1))
        if n < n_heads:
            m1 = 2.0 ** ( -4.0 / n)
            mm = np.power(m1, np.arange(1, 1 + 2 * (n_heads - n), 2))
            slopes = np.concatenate([slopes, mm])
        # slopes = torch.ones(n_heads)
        return slopes
    
    def get_alibi_bias(self, n_heads, max_seqlen):
        if not self.left_align:
            self.bias = np.arange(max_seqlen)
            self.bias = self.bias[None, :] - self.bias[:, None]
            if (self.is_sqrt):
                self.bias = np.sqrt(np.abs(self.bias)) * np.sign(self.bias)
            bias = np.empty(
                (n_heads,
                max_seqlen,
                max_seqlen)
            )
            bias[:, :max_seqlen, :max_seqlen] = self.bias
            self.alibi_slopes = self.get_alibi_slopes(n_heads)
        else:
            self.bias = np.arange(max_seqlen, dtype=np.float32).unsqueeze(0).unsqueeze(0).expand(n_heads, max_seqlen, -1)
            self.alibi_slopes = np.array(self.get_interleave(n_heads))
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
        if data_type == np.float16:
            post_mask_coff = 1
            pre_mask_coff = -10000.0
        # elif data_type == bfloat16.bfloat16 and self.is_alibi:
        #     post_mask_coff = 1
        #     pre_mask_coff = -float("inf")
        elif data_type == np.float32 and self.is_alibi:
            post_mask_coff = 1
            pre_mask_coff = 1
        else:
            post_mask_coff = -3e38
            pre_mask_coff = 1
        if data_type == np.float16:
            if self.is_alibi or self.long_seq:
                select_zero = False
            else:
                select_zero = True
        # elif data_type == bfloat16.bfloat16:
        #     if self.is_alibi:
        #         select_zero = False
        #     elif self.dynamic_batch or self.is_decoder:
        #         select_zero = True
        #     else:
        #         select_zero = False
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
            mask += self.alibi_bias
        if select_zero:
            mask.flat[zero_indice] = 0
        self.mask = mask.astype(np.float32)
        self.post_mask_coff = post_mask_coff
        self.pre_mask_coff = pre_mask_coff

    def gen_out_tensor(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        q_offset = 0
        k_offset = 0
        v_offset = 0
        batch = self.batch
        dynamic_batch = self.dynamic_batch
        batch_state = self.batch_state
        heads = self.heads
        is_decoder = self.is_decoder
        embed = self.embeddim
        max_seq = self.max_seq
        q_seqlen = self.q_seqlen
        kv_seqlen = self.kv_seqlen
        kv_head = self.kv_head
        mask = self.mask
        is_mask = self.is_mask
        q = self.q
        k = self.k
        v = self.v
        q_ntokens = self.q_ntokens
        kv_ntokens = self.kv_ntokens
        s = None
        s_high = None
        _p = None
        _p_high = None
        out = None
        out_high = None

        q_heads = np.array([1])
        q_heads[0] = heads
        kv_heads = np.array([1])
        kv_heads[0] = kv_head
        is_batch_dynamic = np.array([1])
        is_batch_dynamic[0] = dynamic_batch
        batch_num = np.array([1])
        batch_num[0] = batch
        embd_dim = np.array([1])
        embd_dim[0] = embed
        max_seqlen = np.array([1])
        max_seqlen[0] = max_seq
        q_heads.astype(np.uint32).tofile(os.path.join(cur_dir, "q_heads.bin"))
        kv_heads.astype(np.uint32).tofile(os.path.join(cur_dir, "kv_heads.bin"))
        is_batch_dynamic.astype(np.uint32).tofile(os.path.join(cur_dir, "is_batch_dynamic.bin"))
        batch_num.astype(np.uint32).tofile(os.path.join(cur_dir, "batch_num.bin"))
        embd_dim.astype(np.uint32).tofile(os.path.join(cur_dir, "embd_dim.bin"))
        max_seqlen.astype(np.uint32).tofile(os.path.join(cur_dir, "max_seqlen.bin"))
        batch_state.astype(np.uint32).tofile(os.path.join(cur_dir, "batchRunStatus.bin"))
        if self.relayAttention:
            for idx in range(len(self.shareLen)):
                kSepfilename = f"kvBatches/share_input2_{idx}.bin"
                self.kshare[idx].tofile(os.path.join(cur_dir, kSepfilename))
                vSepfilename = f"kvBatches/share_input3_{idx}.bin"
                self.vshare[idx].tofile(os.path.join(cur_dir, vSepfilename))
        for idx in range(batch):        
            q_s = q_seqlen[idx]
            q_slice = q[q_offset:q_offset + q_s][:]
            q_slice = q_slice.reshape(q_s, heads, embed)
            q_slice = np.transpose(q_slice, (1, 0, 2))
            k_slice = None
            v_slice = None
            if self.relayAttention:
                kv_s_share = 0
                if(self.shareIdx[idx] != -1):
                    kv_s_share = self.shareLen[self.shareIdx[idx]]
                kv_s = kv_seqlen[idx] - kv_s_share
                k_slice = k[0][idx][:kv_s][:]
                k_slice_share = self.kshare[self.shareIdx[idx]][:kv_s_share][:]
                v_slice = v[0][idx][:kv_s][:]
                v_slice_share = self.vshare[self.shareIdx[idx]][:kv_s_share][:]
                k_slice = np.concatenate((k_slice_share, k_slice), 0)
                v_slice = np.concatenate((v_slice_share, v_slice), 0)
            else:
                kv_s = kv_seqlen[idx]
                k_slice = k[0][idx][:kv_s][:]
                v_slice = v[0][idx][:kv_s][:]
            kv_s = kv_seqlen[idx]
            k_slice = k_slice.reshape(kv_s, kv_head, embed)
            k_slice_t = np.transpose(k_slice, (1, 2, 0))   # get K^T
            v_slice = v_slice.reshape(kv_s, kv_head, embed)
            v_slice = np.transpose(v_slice, (1, 0, 2))
            kfilename = f"kvBatches/input2_{idx}.bin"
            k[0][idx].tofile(os.path.join(cur_dir, kfilename))
            vfilename = f"kvBatches/input3_{idx}.bin"
            v[0][idx].tofile(os.path.join(cur_dir, vfilename))

            if dynamic_batch and batch_state[idx] == 0 and not is_decoder:
                continue

            if dynamic_batch and batch_state[idx] == 0:
                output = np.zeros([heads, q_s, embed])
                output = np.transpose(output, (1, 0, 2))
                output_high = output.astype(np.float32)
                if out is None:
                    out = output
                    out_high = output_high
                else:
                    out = np.concatenate((out, output), 0)
                    out_high = np.concatenate((out_high, output_high), 0)
                q_offset += q_s
                k_offset += max_seq
                v_offset += max_seq
                continue

            score = self.group_matmul(heads, kv_head, q_slice, k_slice_t)
            score_high = score.astype(np.float32)


            if s is None:
                s = score.reshape([-1, ])
                s_high = score_high.reshape([-1, ])
            else:
                s = np.concatenate((s, score.reshape([-1, ])), 0)
                s_high = np.concatenate((s_high, score_high.reshape([-1, ])), 0)

            scale = 1
            score = score * scale
            score_high = score_high * scale

            if self.is_clamp == 1:
                clamp_min_brc = np.ones((score.shape)) * self.clamp_min
                clamp_max_brc = np.ones((score.shape)) * self.clamp_max
                score = np.float16(np.maximum(score, clamp_min_brc))
                score = np.float16(np.minimum(score, clamp_max_brc))
            if is_mask:
                score = score + self.mask_info[1](self.mask, idx, q_s, kv_s) * self.post_mask_coff
                score_high = score_high + self.mask_info[1](self.mask, idx, q_s, kv_s).astype(np.float32) * self.post_mask_coff
            score = score.astype(np.float32)
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
            # p = p_high.astype(bfloat16.bfloat16)
            p = p_high.astype(np.float16)
            o = self.group_matmul(heads, kv_head, p, v_slice)
            o_high = self.group_matmul(heads, kv_head, p_high, v_slice)
            o = o.reshape(heads, q_s, embed)
            o_high = o_high.reshape(heads, q_s, embed)
            o = np.transpose(o, (1, 0, 2))
            o_high = np.transpose(o_high, (1, 0, 2))
            if out is None:
                out = o
                out_high = o_high
            else:
                out = np.concatenate((out, o), 0)
                out_high = np.concatenate((out_high, o_high), 0)

            q_offset += q_s
            k_offset += max_seq
            v_offset += max_seq
        
        # golden data
        out = out.reshape(q_ntokens, heads * embed)
        out_high = out_high.reshape(q_ntokens, heads * embed)
        self.golden_out = out.astype(self.data_type)
        self.golden_out_high = out_high.astype(np.float32)

        self.q.tofile(os.path.join(cur_dir, "input1.bin"))
        self.k.tofile(os.path.join(cur_dir, "input2.bin"))
        self.v.tofile(os.path.join(cur_dir, "input3.bin"))
        self.mask.astype(self.data_type).tofile(os.path.join(cur_dir, "input4.bin"))
        self.golden_out.tofile(os.path.join(cur_dir, "expect.bin"))
        self.golden_out_high.tofile(os.path.join(cur_dir, "expect_high.bin"))
        np.array(self.q_seqlen).astype(np.uint32).tofile(os.path.join(cur_dir, "q_seqlen.bin"))
        np.array(self.kv_seqlen).astype(np.uint32).tofile(os.path.join(cur_dir, "kv_seqlen.bin"))
        self.layer_id.tofile(os.path.join(cur_dir, "layerId.bin"))
        if self.long_seq:
            self.max_seq = 128
            self.gen_mask(self.batch, self.heads, self.data_type, self.mask_type)

    def gen_seq_len(self, batch, seq_len):
        ntokens = sum(seq_len)
        return seq_len, ntokens

    def group_matmul(self, heads, group_num, A, B):
        group_head = heads // group_num
        score = None
        for i in range(group_num):
            group_score = np.matmul(A[i * group_head: (i + 1) * group_head, :, :].astype(np.float32), B[i:(i + 1), :, :].astype(np.float32))
            if score is None:
                score = group_score
            else:
                score = np.concatenate((score, group_score), 0)
        return score

def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

if __name__ == '__main__':
    folder_for_kvbatches = "kvBatches"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(cur_dir, folder_for_kvbatches)
    os.makedirs(folder_path, exist_ok = True)

    batch = int(sys.argv[1])
    kv_seq = int(sys.argv[2])
    kvhead = int(sys.argv[3])
    isdecoder = int(sys.argv[4])
    heads = int(sys.argv[5])
    embeddim = int(sys.argv[6])
    max_seq = int(sys.argv[7])

    variate_seq = int(sys.argv[8])
    dynamic_batch = int(sys.argv[9])
    mask_type = int(sys.argv[10])
    dtype = int(sys.argv[11])
    is_sqrt = int(sys.argv[12])
    relayAttention = 0
    shareIdx = np.array([])
    shareLen = np.array([])
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
    if len(sys.argv) - 1 > 12:
        relayAttention = int(sys.argv[13])
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        shareIdx = np.fromfile(cur_dir + "/share_idx.bin",dtype=np.int32)
        shareLen = np.fromfile(cur_dir + "/share_len.bin",dtype=np.int32)
        logging.debug(f"shareIdx: {shareIdx}")
        logging.debug(f"shareLen: {shareLen}")

    is_alibi = False
    data_type = None
    if dtype == 1:
        data_type = np.float16
    if mask_type == 3 or mask_type == 4:
        is_alibi = True
    else:
        is_alibi = False
    delete_files_in_folder(folder_path)
    test_data_gen = FlashAttentionDataGen()
    test_data_gen.set_data_params(dynamic_batch = dynamic_batch, is_mask = not relayAttention,
                                  is_decoder = isdecoder, batch = batch, kv_head = kvhead, heads = heads, 
                                  embeddim = embeddim, max_seq = max_seq, kv_seqlen = kv_seq,
                                  data_type = data_type, is_alibi = is_alibi,
                                  mask_type = mask_type, is_sqrt = is_sqrt, variate_seq = variate_seq,
                                  relayAttention = relayAttention, shareIdx = shareIdx, shareLen = shareLen)
    test_data_gen.gen_out_tensor()
