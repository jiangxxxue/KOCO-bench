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

np.random.seed(43)

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
import operation_test  # NOQA: E402

class TestFlashAttentionNzDataGenerator():
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
        layer = 2
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
            tor = float(1/math.sqrt(1.0 * embed))
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
        self.q_atb = input1_nd.reshape(batch, heads, max_seq, embed)
        q = self.convert_nd_to_nz(input1_nd)
    
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
        self.golden_out = out
 
        self.golden_outhigh_pre = outhigh_pre
        self.q_seqlen = q_seqlen
        self.q_ntokens = q_ntokens
        self.kv_seqlen = kv_seqlen
        self.kv_ntokens = kv_ntokens
 
        logging.debug("**********data gen shape***********")
        logging.debug(f"q shape: {self.q_atb.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")

    def test_flash_attention_case_encoder_norm_mask_dim3_BNSD(self):
        batch = 2
        qkv_seq = 1040
        kv_head = 4        # kv_head num
        heads = 32          # llama7b  hidden_size 4096
        embeddim = 128
        max_seq = 2048
        tor = math.sqrt(1 / embeddim)
        mask_dim = 3

        self.set_data_params(is_BNSD=True)
        self.calc_data((batch, qkv_seq, heads, kv_head,
                       embeddim, max_seq, mask_dim))
 
        logging.debug("**********input shape***********")
        logging.debug(f"q shape: {self.q_atb.shape}")
        logging.debug(f"k shape: {self.k.shape}")
        logging.debug(f"v shape: {self.v.shape}")

        return [self.q_atb, self.k, self.v, self.mask, self.kv_seqlen, self.q_seqlen, self.layer_id, self.golden_out]


class TestFlashAttentionEncoderOperationBNSD910A(operation_test.OperationTest):
    def golden_calc(self, input_tensors):
        return [self.in_tensors[-1]]

    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.allclose(out_tensor, golden_out_tensor.reshape(out_tensor.shape), rtol=0.001, atol=0.001)

    def test(self):
        if operation_test.get_soc_version() != 'Ascend310P':
            print("this testcase only supports Ascend310P")
            return
        dataGenerator = TestFlashAttentionNzDataGenerator()
        data = dataGenerator.test_flash_attention_case_encoder_norm_mask_dim3_BNSD()

        param_seqlen = data[5]
        data[5] = torch.from_numpy(data[5])
        param_token_offset = data[4]
        data[4] = torch.from_numpy(data[4])

        self.in_tensors = [tensor.npu() for tensor in data]
        self.in_tensors[1] = torch_npu.npu_format_cast(self.in_tensors[1].contiguous(), 29)
        self.in_tensors[2] = torch_npu.npu_format_cast(self.in_tensors[2].contiguous(), 29)
        self.in_tensors[3] = torch_npu.npu_format_cast(self.in_tensors[3].contiguous(), 29)

        OP_NAME = "SelfAttentionOperation"
        PARAM = json.dumps({"headNum": 32, "kvHeadNum":4, "qkScale": (1 / float(math.sqrt(128))),
                            "maskType": 1, "kvcacheCfg":1, "inputLayout":1})
        RUN_PARAM = json.dumps({"tokenOffset": param_token_offset.tolist(), "seqLen": param_seqlen.tolist()})
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [
                self.in_tensors[0], self.in_tensors[1], self.in_tensors[2], self.in_tensors[3], self.in_tensors[4],
                self.in_tensors[5], self.in_tensors[6]
            ])
if __name__ == '__main__':
    unittest.main()
