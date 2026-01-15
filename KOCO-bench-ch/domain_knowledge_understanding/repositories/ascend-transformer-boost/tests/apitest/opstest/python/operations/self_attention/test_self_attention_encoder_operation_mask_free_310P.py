#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import json
import math
import os
import sys
import unittest
import random
import numpy as np
import torch
import torch_npu
import logging

np.random.seed(0)

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402


class AttentionDataGenerator():
    def set_data_params(self, is_mask=True, is_batch_mask=False, is_decoder=False, variate_seq=False, is_alibi=False, is_alibi_128=False, is_alibi_256=False, left_align=False, is_sqrt=False):
        self.is_mask = is_mask
        self.is_batch_mask = is_batch_mask
        self.is_decoder = is_decoder
        self.variate_seq = variate_seq
        self.is_alibi = is_alibi
        self.is_alibi_128 = is_alibi_128
        self.is_alibi_256 = is_alibi_256
        self.left_align = left_align
        self.is_sqrt = is_sqrt

    def get_data_params(self):
        ret = (self.is_mask, self.is_batch_mask,
               self.is_decoder, self.variate_seq, self.is_alibi, self.is_alibi_128, self.is_alibi_256)
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
                                    mat_b[i:(i + 1), :, :].astype(np.float32)).astype(np.float16)
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
    
    def calc_data_pa(self, shape: tuple):
        batch, seqlen, heads, kv_head, embed, max_seq, mask_dim = shape
        is_mask, is_batch_mask, is_decoder, variate_seq, is_alibi, is_alibi_128, is_alibi_256 = self.get_data_params()
        if is_decoder:
            q_seqlen, q_seqlen_aligned, q_ntokens = self.gen_seq_len(
                batch, 1, variate_seq)
        else:
            q_seqlen, q_seqlen_aligned, q_ntokens = self.gen_seq_len(
                batch, seqlen, variate_seq)
 
        q_ntokens = q_seqlen.sum()
        q_ntokens_aligned = (q_ntokens + 15) // 16 * 16
        q = np.random.uniform(-1.0, 1.0, size=(q_ntokens_aligned,
                              heads * embed)).astype(np.float16)
        k = np.random.uniform(-1.0, 1.0, size=(q_ntokens_aligned,
                              kv_head * embed)).astype(np.float16)
        v = np.random.uniform(-1.0, 1.0, size=(q_ntokens_aligned,
                              kv_head * embed)).astype(np.float16)
        max_seq = (seqlen + 15) // 16 * 16
        if is_alibi:
            bsz_heads = batch * heads if mask_dim == 4 else heads
            mask = np.ones(shape=(bsz_heads, max_seq,
                            max_seq)).astype(np.float16)
            mask = np.triu(mask, 1)
            mask *= -10000
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
            q_slice = np.transpose(q_slice, (1, 0, 2))  # (head, token, emd)
 
            # 计算
            k_slice = k[k_offset:k_offset + s][:]
            k_slice = k_slice.reshape(s, kv_head, embed)
            # heads*max_seqlen*embed
            k_slice = np.transpose(k_slice, (1, 0, 2))
 
            # 输入
            k_slice_t = np.transpose(k_slice, (0, 2, 1))   # get K^T
 
            # 计算
            v_slice = v[v_offset:v_offset + s][:]
            v_slice = v_slice.reshape(s, kv_head, embed)
            # heads*max_seqlen*embed
            v_slice = np.transpose(v_slice, (1, 0, 2))
            score = self.group_matmul(heads, kv_head, q_slice, k_slice_t)
            tor = np.float16(math.sqrt(1.0 * embed))
            score = score / tor
            if mask_dim != 0:
                if is_alibi:
                    if mask_dim == 4:
                        score += mask[idx][:, :s, :s]
                    else:
                        score += mask[:, :s, :s]
                else:
                    score = score + mask[0][:s, :s]
 
            score_max = np.max(score, axis=-1)
            score = score - score_max.reshape((heads, s, 1))
            score_exp = np.exp(score.astype(np.float32))
            score_sum = np.sum(score_exp, axis=-1)
            p = score_exp / score_sum.reshape((heads, s, 1))
            p = p.astype(np.float16)
            o_mat = self.group_matmul(heads, kv_head, p.astype(np.float32),
                                      v_slice.astype(np.float32)).astype(np.float16)
            o_mat = o_mat.reshape(heads, s, embed)
            if input1_nd is None:
                input1_nd = q_slice
            else:
                input1_nd = np.concatenate((input1_nd, q_slice), 1)
 
            if input2_nd is None:
                input2_nd = k_slice
            else:
                input2_nd = np.concatenate((input2_nd, k_slice), 1)
 
            if input3_nd is None:
                input3_nd = v_slice
            else:
                input3_nd = np.concatenate((input3_nd, v_slice), 1)
 
            if out is None:
                out = o_mat
            else:
                out = np.concatenate((out, o_mat), 1)
 
            q_offset += s
            k_offset += s
            v_offset += s
        # input data
        q_nd = np.zeros((heads, q_ntokens_aligned, embed)).astype(np.float16)
        q_nd[:, :q_ntokens, :] = input1_nd
 
        k_nd = np.zeros((kv_head, q_ntokens_aligned, embed)).astype(np.float16)
        k_nd[:, :q_ntokens, :] = input2_nd
 
        v_nd = np.zeros((kv_head, q_ntokens_aligned, embed)).astype(np.float16)
        v_nd[:, :q_ntokens, :] = input3_nd
 
        o_nd = np.zeros((heads, q_ntokens_aligned, embed)).astype(np.float16)
        o_nd[:, :q_ntokens, :] = out
        self.q_nd = q_nd
        self.k_nd = k_nd
        self.v_nd = v_nd
        self.o_nd = o_nd
        q = self.convert_nd_to_nz(q_nd)
        self.q = q.astype(np.float16).reshape(
            1, heads * embed // 16, q_ntokens_aligned, 16)
        k = self.convert_nd_to_nz(k_nd)
        self.k = k.reshape(1, kv_head * embed // 16, q_ntokens_aligned, 16)
        v = self.convert_nd_to_nz(v_nd)
        self.v = v.reshape(1, kv_head * embed // 16, q_ntokens_aligned, 16)
        if mask_dim == 4 and is_alibi_128:
            mask = mask[0, :, :, :128]
        mask = self.convert_nd_to_nz(mask).astype(np.float16)
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
            mask = np.ones(shape=(256, 256)).astype(np.float16) * 60000
            mask = np.triu(mask, 1)
            if max_seq < 256:
                self.bias = torch.tensor(np.pad(self.bias, ((0, 256 - max_seq), (0, 256 - max_seq)), "constant"))
            mask = self.bias[:256, :256] * -1 + mask
            mask = mask.numpy().astype(np.float16)
            self.mask_nd = mask
            mask = self.convert_nd_to_nz(mask).astype(np.float16)
            self.mask = mask.reshape(1, 16, 256, 16)
        else:
            self.alibi_slopes = []
        # golden data
        out_nz = self.convert_nd_to_nz(o_nd).astype(np.float16).reshape(
            1, heads * embed // 16, q_ntokens_aligned, 16)
        self.golden_out = out_nz
        self.q_seqlen = q_seqlen
        self.q_ntokens = q_ntokens
        self.kv_seqlen = q_seqlen
        self.kv_ntokens = q_ntokens
        self.layer_id = torch.tensor([], dtype=torch.int32)

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
        param_lists["is_cache"] = False
        param_lists["type"] = 2005
        param_lists["is_alibi"] = True
        param_lists["is_alibi_128"] = False
        param_lists["is_alibi_256"] = True
        param_lists["is_sqrt"] = False
        logging.info("======================test flash attention encoder alibi_256 no cache test1======================")


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
            is_sqrt
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
            param_lists["is_sqrt"]
        )


        self.set_data_params(is_alibi=is_alibi, is_alibi_128=is_alibi_128, is_alibi_256=is_alibi_256, is_sqrt=is_sqrt)
        self.calc_data_pa((batch, qkv_seq, heads, kv_head, embeddim, max_seq, mask_dim))

        print("**********input shape***********")
        print(f"q shape: {self.q.shape}")
        print(f"k shape: {self.k.shape}")
        print(f"v shape: {self.v.shape}")
        print(f"layer_id shape: {self.layer_id.shape}")
        print(f"mask shape: {self.mask.shape}")
        attention_out = np.zeros_like(self.q)
data_generator = AttentionDataGenerator()
data_generator.test_flash_attention_encoder_alibi_test1()

(query, key, value, mask, seqLen, slopes, golden) = (
        data_generator.q_nd,
        data_generator.k_nd,
        data_generator.v_nd,
        data_generator.mask,
        data_generator.q_seqlen,
        data_generator.alibi_slopes,
        data_generator.o_nd
    )

data = [query, key, value, mask, seqLen, slopes, golden]
param_seqlen = data[4].tolist()

data[0] = torch.from_numpy(data[0]).permute(1, 0, 2)
data[1] = torch.from_numpy(data[1]).permute(1, 0, 2)
data[2] = torch.from_numpy(data[2]).permute(1, 0, 2)
data[3] = torch.from_numpy(data[3])
data[4] = torch.from_numpy(data[4])

data[6] = torch.from_numpy(data[6]).permute(1, 0, 2)
in_tensors = [tensor.npu().contiguous() for tensor in data]
in_tensors[3] = torch_npu.npu_format_cast(in_tensors[3], 29)
in_tensors[5] = in_tensors[5].half()



OP_NAME = "SelfAttentionOperation"
PARAM = json.dumps({"headNum": 17, "qkScale": (1 / float(math.sqrt(128))), "kvHeadNum": 17,
                    "calcType": 3, "maskType": 4, "isTriuMask": 1})
RUN_PARAM = json.dumps({"seqLen": param_seqlen})

class TestFlashAttentionEncoderOperationMaskFree910A(operation_test.OperationTest):
    def golden_calc(self, input_tensors):
        return [in_tensors[6]]

    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.allclose(out_tensor, golden_out_tensor, rtol=0.001, atol=0.001)

    def test(self):
        if operation_test.get_soc_version() != 'Ascend310P':
            print("this testcase only supports Ascend310P")
            return
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [
            in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4], in_tensors[5]
        ])


if __name__ == '__main__':
    unittest.main()
