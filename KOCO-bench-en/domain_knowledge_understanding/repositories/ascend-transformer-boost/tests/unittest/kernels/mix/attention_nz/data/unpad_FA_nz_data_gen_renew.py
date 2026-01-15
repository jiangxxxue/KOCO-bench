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
import sys
import unittest
import math
import numpy as np
import torch
import random
import os
import shutil

class FlashAttentionNzDataGen():
    def set_data_params(self, is_mask=True, is_batch_mask=False, is_decoder=False, variate_seq=False, is_alibi=False, is_alibi_128=False, is_alibi_256=False, left_align=False, is_sqrt=False, is_BNSD=False):
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

    def get_data_params(self):
        ret = (self.is_mask, self.is_batch_mask,
               self.is_decoder, self.variate_seq, self.is_alibi, self.is_alibi_128, self.is_alibi_256, self.is_BNSD)
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
        return x.reshape(x_shape[:-4] + [m1, m0, n1, n0]).transpose(*array_trans)

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
            group_score = np.matmul(mat_a[i * group_heads: (i + 1) * group_heads].astype(np.float32),
                                    mat_b[i:(i + 1), :, :].astype(np.float32))
            if score is None:
                score = group_score
            else:
                score = np.concatenate((score, group_score), 0)
        return score

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
        bias = bias * self.alibi_slopes[:, None, None]
        return bias
    
    def calc_data(self, shape: tuple):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        layer = 1
        batch, seqlen, heads, kv_head, embed, max_seq, mask_dim = shape
        self.embed, self.max_seq = embed, max_seq
        is_mask, is_batch_mask, is_decoder, variate_seq, is_alibi, is_alibi_128, is_alibi_256, is_BNSD = self.get_data_params()
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
        layer_id = np.array([layer-1], dtype=np.int32)
        q = np.random.uniform(-1.0, 1.0, size=(q_ntokens,
                              heads * embed)).astype(np.float16)
        k = np.random.uniform(-1.0, 1.0, size=(batch *
                              seqlen, kv_head * embed)).astype(np.float16)
        v = np.random.uniform(-1.0, 1.0, size=(batch *
                              seqlen, kv_head * embed)).astype(np.float16)
        if is_BNSD and not is_decoder:
            q = np.random.uniform(-1.0, 1.0, size=(batch * seqlen, heads * embed)).astype(np.float16)
            q_max = np.zeros(shape=(batch * max_seq, heads * embed)).astype(np.float16)
            for i in range(batch):
                q_max[max_seq * i:(max_seq * i) + seqlen] = q[seqlen * i:(seqlen * i) + seqlen]
            q = q_max
        k_max = np.zeros(shape=(layer, batch * max_seq,
                         kv_head * embed)).astype(np.float16)
        v_max = np.zeros(shape=(layer, batch * max_seq,
                         kv_head * embed)).astype(np.float16)
        k_max[layer_id[0]][:batch * seqlen] = k
        v_max[layer_id[0]][:batch * seqlen] = v
        if is_mask:
            if is_alibi:
                bsz_heads = batch * heads if mask_dim == 4 else heads
                mask = np.ones(shape=(bsz_heads, max_seq,max_seq))
                mask *= -60000
                mask = np.triu(mask, 1)
                mask = mask.astype(np.float16)
                self.alibi_bias = self.get_alibi_bias(heads, max_seq)
                mask = mask.reshape(
                    (batch, heads, max_seq, max_seq)) if mask_dim == 4 else mask
                mask += self.alibi_bias.numpy()
            elif mask_dim == 2:
                mask = np.ones(shape=(1, max_seq, max_seq)).astype(np.float16)
                mask = np.triu(mask, 1)
                mask *= -10000.0
            else:
                mask = np.ones(shape=(batch, max_seq, max_seq)
                               ).astype(np.float16)
                mask = np.triu(mask, 1)
                mask *= -10000.0
        elif is_decoder:
            if is_alibi:
                bsz_heads = batch * heads if mask_dim == 4 else heads
                mask = np.ones(shape=(bsz_heads, 16, max_seq)
                               ).astype(np.float16)
                mask = np.triu(mask, 1)
                mask *= -10000
                for i in range(bsz_heads):
                    mask[i] += np.random.uniform(-2, 2,
                                                 size=(16, max_seq)).astype(np.float16)
                mask = mask.reshape((batch, heads, 16, max_seq)
                                    ) if mask_dim == 4 else mask
            elif mask_dim == 2:
                mask = np.zeros(shape=(1, 16, max_seq)).astype(np.float16)
                mask[0, :1, :2] = -10000
            else:
                mask = np.zeros(shape=(batch, 16, max_seq)).astype(np.float16)
                for i in range(batch):
                    mask[i, :1, :i] = -10000
        q_offset = 0
        k_offset = 0
        v_offset = 0
        input1_nd = None
        input2 = None
        input3 = None
        out = None
        out_high = None

        q_heads = np.array([1])
        q_heads[0] = heads
        kv_heads = np.array([1])
        kv_heads[0] = kv_head
        batch_num = np.array([1])
        batch_num[0] = batch
        embd_dim = np.array([1])
        embd_dim[0] = embed
        max_seqlen = np.array([1])
        max_seqlen[0] = max_seq
        q_heads.astype(np.uint32).tofile(os.path.join(cur_dir, "q_heads.bin"))
        kv_heads.astype(np.uint32).tofile(os.path.join(cur_dir, "kv_heads.bin"))
        batch_num.astype(np.uint32).tofile(os.path.join(cur_dir, "batch_num.bin"))
        embd_dim.astype(np.uint32).tofile(os.path.join(cur_dir, "embd_dim.bin"))
        max_seqlen.astype(np.uint32).tofile(os.path.join(cur_dir, "max_seqlen.bin"))

        for idx in range(batch):
            s_align = q_seqlen_aligned[idx]
            kv_align = kv_seqlen_aligned[idx]
            s = q_seqlen[idx]
            kv = kv_seqlen[idx]

            q_slice = q[q_offset:q_offset + s][:]
            q_slice = q_slice.reshape(s, heads, embed)
            q_slice = np.transpose(q_slice, (1, 0, 2))  # (head, token, emd)

            # 计算
            k_slice = k_max[layer_id[0]][k_offset:k_offset + kv][:]
            k_slice = k_slice.reshape(kv, kv_head, embed)
            # heads*max_seqlen*embed
            k_slice = np.transpose(k_slice, (1, 0, 2))

            # 输入
            k_slice_max = k_max[layer_id[0]][k_offset:k_offset + max_seq][:]
            k_slice_max = k_slice_max.reshape(max_seq, kv_head, embed)
            # heads*max_seqlen*embed
            k_slice_max = np.transpose(k_slice_max, (1, 0, 2))
            k_slice_t = np.transpose(k_slice, (0, 2, 1))   # get K^T

            # 计算
            v_slice = v_max[layer_id[0]][v_offset:v_offset + kv][:]
            v_slice = v_slice.reshape(kv, kv_head, embed)
            v_slice = np.transpose(v_slice, (1, 0, 2))

            # 输入
            v_slice_max = v_max[layer_id[0]][v_offset:v_offset + max_seq][:]
            v_slice_max = v_slice_max.reshape(max_seq, kv_head, embed)
            v_slice_max = np.transpose(v_slice_max, (1, 0, 2))

            score = self.group_matmul(heads, kv_head, q_slice, k_slice_t)
            score_high = score.astype(np.float32)
            tor = np.float16(math.sqrt(1.0 * embed))
            score = score / tor
            if mask_dim != 0:
                if is_alibi:
                    if mask_dim == 4:
                        score += mask[idx][:, :s, :kv]
                    else:
                        score += mask[:, :s, :kv]
                elif mask_dim == 2:
                    score += mask[0][:s, :kv]
                else:
                    score = score + mask[idx, :s, :kv]

            score = score.astype(np.float32)
            score_max = np.max(score, axis=-1)
            score = score - score_max.reshape((heads, s, 1))
            score_exp = np.exp(score.astype(np.float32))
            score_sum = np.sum(score_exp, axis=-1)
            p_high = score_exp / score_sum.reshape((heads, s, 1))
            p = p_high.astype(np.float16)
            o_mat = self.group_matmul(heads, kv_head, p,
                                      v_slice)
            o_mat_high = self.group_matmul(heads, kv_head, p_high,
                                      v_slice)
            o_mat = o_mat.reshape(heads, s, embed)
            o_mat_high = o_mat_high.reshape(heads, s, embed)
            if is_BNSD and not is_decoder:
                q_pad = np.zeros((heads, max_seq, embed))
                o_pad = np.zeros((heads, max_seq, embed))
                o_pad_high = np.zeros((heads, max_seq, embed))
            else:
                q_pad = np.zeros((heads, s_align, embed))
                o_pad = np.zeros((heads, s_align, embed))
                o_pad_high = np.zeros((heads, s_align, embed))
            q_pad[:, :s, :] = q_slice
            o_pad[:, :s, :] = o_mat
            o_pad_high[:, :s, :] = o_mat_high

            if input1_nd is None:
                input1_nd = q_pad
            else:
                input1_nd = np.concatenate((input1_nd, q_pad), 1) if not is_BNSD else np.concatenate((input1_nd, q_pad), 0)
            input2_slice = self.convert_nd_to_nz(k_slice_max)
            if input2 is None:
                input2 = input2_slice.reshape([-1, 16, 16])
            else:
                input2 = np.concatenate(
                    (input2, input2_slice.reshape([-1, 16, 16])), 0)
            input3_slice = self.convert_nd_to_nz(v_slice_max)
            if input3 is None:
                input3 = input3_slice.reshape([-1, 16, 16])
            else:
                input3 = np.concatenate(
                    (input3, input3_slice.reshape([-1, 16, 16])), 0)
            if out is None:
                out = o_pad
                out_high = o_pad_high
            else:
                out = np.concatenate((out, o_pad), 1) if not is_BNSD else np.concatenate((out, o_pad), 0)
                out_high = np.concatenate((out_high, o_pad_high), 1) if not is_BNSD else np.concatenate((out_high, o_pad_high), 0)

            if is_BNSD and not is_decoder:
                q_offset += max_seq
            else:
                q_offset += s
            k_offset += max_seq
            v_offset += max_seq
        # input data
        q = self.convert_nd_to_nz(input1_nd)
        if is_BNSD and not is_decoder:
            self.q = q.astype(np.float16).reshape(batch * heads, embed // 16, max_seq, 16)
        elif is_BNSD and is_decoder:
            self.q = q.astype(np.float16).reshape(batch * heads, embed // 16, 16, 16)
        else:
            self.q = q.astype(np.float16).reshape(1, heads * embed // 16, q_ntokens, 16)
        input2_shape0, input2_shape1, input2_shape2 = input2.shape
        k = np.zeros(shape=(layer, input2_shape0, input2_shape1,
                     input2_shape2)).astype(np.float16)
        k[layer_id] = input2
        self.k = k.reshape(layer, batch, kv_head * embed // 16, max_seq, 16) if not is_BNSD else k.reshape(layer, batch * kv_head, embed // 16, max_seq, 16)
        input3_shape0, input3_shape1, input3_shape2 = input3.shape
        v = np.zeros(shape=(layer, input3_shape0, input3_shape1,
                     input3_shape2)).astype(np.float16)
        v[layer_id] = input3
        self.v = v.reshape(layer, batch, kv_head * embed // 16, max_seq, 16) if not is_BNSD else v.reshape(layer, batch * kv_head, embed // 16, max_seq, 16)
        if mask_dim == 4 and is_alibi_128:
            mask = mask[0, :, :, :128]
        mask = self.convert_nd_to_nz(mask).astype(np.float16)
        if is_mask:
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
            self.alibi_slopes *= -1
            mask = np.ones(shape=(256, 256)) * 60000
            mask = np.triu(mask, 1)
            mask = mask.astype(np.float16)
            if max_seq < 256:
                self.bias = torch.tensor(np.pad(self.bias, ((0, 256 - max_seq), (0, 256 - max_seq)), "constant"))
            mask = self.bias[:256, :256] * -1 + mask
            mask = mask.numpy().astype(np.float16)
            mask = self.convert_nd_to_nz(mask).astype(np.float16)
            self.mask = mask.reshape(1, 16, 256, 16)
        else:
            self.alibi_slopes = []
        self.layer_id = layer_id

        for idx in range(batch):
            kfilename = f"kvBatches/input2_{idx}.bin"
            self.k[0][idx].astype(np.float16).tofile(os.path.join(cur_dir, kfilename))
            vfilename = f"kvBatches/input3_{idx}.bin"
            self.v[0][idx].astype(np.float16).tofile(os.path.join(cur_dir, vfilename))
        
        # golden data
        if is_BNSD and not is_decoder:
            out_nz = self.convert_nd_to_nz(out).astype(np.float16).reshape(batch * heads, embed // 16, max_seq, 16)
            out_nz_high = self.convert_nd_to_nz(out_high).astype(np.float32).reshape(batch * heads, embed // 16, max_seq, 16)
        elif is_BNSD and is_decoder:
            out_nz = self.convert_nd_to_nz(out).astype(np.float16).reshape(batch * heads, embed // 16, 16, 16)
            out_nz_high = self.convert_nd_to_nz(out_high).astype(np.float32).reshape(batch * heads, embed // 16, 16, 16)
        else:
            out_nz = self.convert_nd_to_nz(out).astype(np.float16).reshape(1, heads * embed // 16, q_ntokens, 16)
            out_nz_high = self.convert_nd_to_nz(out_high).astype(np.float32).reshape(1, heads * embed // 16, q_ntokens, 16)
        self.golden_out = out_nz
        self.golden_out_high = out_nz_high
        self.q_seqlen = q_seqlen
        self.q_ntokens = q_ntokens
        self.kv_seqlen = kv_seqlen
        self.kv_ntokens = kv_ntokens

        self.q.astype(np.float16).tofile(os.path.join(cur_dir, "input1.bin"))
        self.k.astype(np.float16).tofile(os.path.join(cur_dir, "input2.bin"))
        self.v.astype(np.float16).tofile(os.path.join(cur_dir, "input3.bin"))
        self.mask.astype(np.float16).tofile(os.path.join(cur_dir, "input4.bin"))
        self.golden_out.tofile(os.path.join(cur_dir, "expect.bin"))
        self.golden_out_high.tofile(os.path.join(cur_dir, "expect_high.bin"))
        np.array(self.q_seqlen).astype(np.uint32).tofile(os.path.join(cur_dir, "q_seqlen.bin"))
        np.array(self.kv_seqlen).astype(np.uint32).tofile(os.path.join(cur_dir, "kv_seqlen.bin"))
        self.layer_id.astype(np.uint32).tofile(os.path.join(cur_dir, "layerId.bin"))

        logging.debug("**********data gen shape***********")
        logging.debug(f"q shape: {q.shape}")
        logging.debug(f"k shape: {k.shape}")
        logging.debug(f"v shape: {v.shape}")
        logging.debug(f"layer_id shape: {layer_id.shape}")
        logging.debug(f"mask shape: {mask.shape}")

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
    kv_head = int(sys.argv[3])
    is_decoder = int(sys.argv[4])
    heads = int(sys.argv[5])
    embeddim = int(sys.argv[6])
    max_seq = int(sys.argv[7])

    variate_seq = int(sys.argv[8])
    is_alibi = int(sys.argv[9])
    mask_dim = int(sys.argv[10])
    is_sqrt = int(sys.argv[11])
    is_BNSD = int(sys.argv[12])
    delete_files_in_folder(folder_path)
    test_data_gen = FlashAttentionNzDataGen()
    if mask_dim == 0:
        test_data_gen.set_data_params(is_mask=False, is_batch_mask=False, 
                                  is_decoder=is_decoder, variate_seq=variate_seq, 
                                  is_alibi=is_alibi, is_alibi_128=False, 
                                  is_alibi_256=False, left_align=False, is_sqrt=is_sqrt, is_BNSD=is_BNSD)
    else:
        test_data_gen.set_data_params(is_mask=True, is_batch_mask=False, 
                                  is_decoder=is_decoder, variate_seq=variate_seq, 
                                  is_alibi=is_alibi, is_alibi_128=False, 
                                  is_alibi_256=False, left_align=False, is_sqrt=is_sqrt, is_BNSD=is_BNSD)
    test_data_gen.calc_data((batch, kv_seq, heads, kv_head, embeddim, max_seq, mask_dim))

