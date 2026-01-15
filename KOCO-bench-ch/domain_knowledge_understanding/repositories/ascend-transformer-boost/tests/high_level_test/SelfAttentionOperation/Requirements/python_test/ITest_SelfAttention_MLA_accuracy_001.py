# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.



import unittest
import math
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

import torch
import operation_test  # NOQA: E402
import json

MASK_TYPE_NO_MASK = 0
MASK_TYPE_NO_HEAD = 1
MASK_TYPE_NO_BATCH = 2
MASK_TYPE_ALIBI_WITH_BATCH = 3
MASK_TYPE_ALIBI_NO_BATCH = 4
MASK_TYPE_NO_HEAD_DECODER = 5

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


    def set_data_params(self, dynamic_batch=False, batch_state=None,
                       is_mask=True, is_decoder=False, is_alibi=False, alibi_dim=4,
                       batch = 1, kv_head = 1, heads = 1, embeddim = 128, embeddimv = 0, max_seq = 2048,
                       kv_seqLen = [], is_clamp = 0, clamp_min = 0,
                       clamp_max = 0, data_type = torch.float16, op_type = 0, mask_type = 0,
                       no_cache = False, long_seq = False, is_triu_mask = False, is_multi_layer = False,
                       is_sqrt = False, left_align = False):
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
        tor = np.float32(1.0 / math.sqrt(1.0 * self.embeddim))
        # tor = 1
        self.q = (q * tor).to(data_type)
        #self.k = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddim))).to(data_type)
        self.k = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(batch, self.max_seq, kv_head, self.embeddim))).to(data_type)
        # import pdb;pdb.set_trace()
        self.v = self.k
        # self.v = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, batch, self.max_seq, kv_head * self.embeddimv))).to(data_type)
        self.gen_mask(batch, heads, data_type, mask_type)


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

    def gen_out_tensor(self):
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
            #k_slice = k[layer_id][idx][:kv_s][:]
            k_slice = k[idx][:kv_s][:][:]
            k_slice = k_slice.view(kv_s, kv_head, embed)
            k_slice_t = torch.permute(k_slice, (1, 2, 0))   # get K^T
            #v_slice = v[layer_id][idx][:kv_s][:]
            v_slice = v[idx][:kv_s][:][:]
            v_slice = v_slice.view(kv_s, kv_head, embed)
            v_slice = v_slice[:,:,:embedv]
            v_slice = torch.permute(v_slice, (1, 0, 2))

            score = self.group_mm_torch(heads, kv_head, q_slice, k_slice_t)

            if s is None:
                s = score.view([-1, ])
            else:
                s = torch.cat((s, score.view([-1, ])), 0)

            scale = 1
            if not self.is_multi_layer:
                # 当前scale和tor保持一致，模型侧可能传入scale = np.float32(layer_id + 1)
                scale = np.float32(layer_id + 1)
            score = score * scale

            if self.is_clamp == 1:
                clamp_min_brc = np.ones((score.shape)) * self.clamp_min
                clamp_max_brc = np.ones((score.shape)) * self.clamp_max
                score = np.float16(np.maximum(score, clamp_min_brc))
                score = torch.from_numpy(np.float16(np.minimum(score, clamp_max_brc)))
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
        out_high = out_high.view(q_ntokens, heads * embedv)
        out = out.view(q_ntokens, heads * embedv)
        self.golden_out = out.to(self.data_type)
        self.golden_out_high = out_high.to(torch.float32)
        if self.no_cache:
            self.k = self.close_pack(self.k.to(torch.float32), kv_seqlen).to(self.data_type)
            self.v = self.close_pack(self.v.to(torch.float32), kv_seqlen).to(self.data_type)
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

        # len = out.shape[0] * out.shape[0]
        len = 20288 * 8192
        diff = torch.abs(golden - out)
        max_diff = diff.max().item()

        limit_error = torch.maximum(torch.abs(golden * ratios[0]), torch.tensor(ratios[1]))
        strict_limit_error = torch.maximum(torch.abs(golden * ratios[2]), torch.tensor(ratios[3]))
        error_count = torch.gt(diff, limit_error).sum().item()
        strict_error_count = torch.gt(diff, strict_limit_error).sum().item()
        # print("1/1000 Accuracy is %f",  1 - float(error_count) / len)
        # print("3/1000 Accuracy is %f",  1 - float(strict_error_count) / len)
        if self.data_type == torch.bfloat16 or not self.is_decoder:
            return (float(strict_error_count) / len) <= ratios[2]
        else:
            return (float(error_count) / len) <= ratios[0]

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
        # result_double = compare_cv(self.golden_out_high, golden_tensors[0], out_tensors[0])
        # result_old = self.compare_output_data(out_tensors[0].half(), golden_tensors[0].half(), [0.001, 0.001, 0.003, 0.003, 0.005, 0.005])
        # return (result_double or result_old)
        return self.compare_output_data(out_tensors[0].half(), golden_tensors[0].half(), [0.001, 0.001, 0.003, 0.003, 0.005, 0.005])
        
        #return torch.allclose(out_tensors, golden_tensors, rtol=0.001, atol=0.001)

    def test_flash_attention_mla_fp16(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        # unpad encoder
        batch = 16
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 16          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 512
        max_seq = 2048
        tor = 1
        dynamic_batch = False
        kv_seqLen = [500, 1024, 1500, 2048] * 4
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "SelfAttentionOperation"
        OP_PARAM = {"type": 2008, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV,"kvHead": kv_head,"headSize": heads, "tor": tor}
        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH)
        self.gen_out_tensor()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))

        PARAM = json.dumps({"headNum": 16, "qkScale": float(1.0 / math.sqrt(1.0 * self.embeddim)), "kvHeadNum": kv_head, \
            "calcType": 3, "maskType": 1, "mlaVHeadSize": embeddimV})
        RUN_PARAM = json.dumps({"seqLen": kv_seqLen})
        kv_seqLen = np.array(kv_seqLen)
        kv_seqLen = torch.from_numpy(kv_seqLen).to(torch.int32).npu()
        # attention_out = np.zeros_like(self.golden_out)
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [self.q.npu(), self.k.npu(), self.mask.to(data_type).npu(), kv_seqLen])
        # return self.execute([self.q, self.k, self.layer_id, self.mask.to(data_type)],
        #                     [torch.tensor(attention_out).half()])

    def test_flash_attention_mla_fp16_case2(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        # unpad encoder
        batch = 1
        kv_head = 16        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 16          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 512
        max_seq = 2048
        tor = 1
        dynamic_batch = False
        kv_seqLen = [500]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "SelfAttentionOperation"
        OP_PARAM = {"type": 2008, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV,"kvHead": kv_head,"headSize": heads, "tor": tor}

        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH)
        self.gen_out_tensor()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))

        attention_out = np.zeros_like(self.golden_out)
       
        PARAM = json.dumps({"headNum": heads, "qkScale": float(1.0 / math.sqrt(1.0 * self.embeddim)), "kvHeadNum": kv_head, \
            "calcType": 3, "maskType": 1, "mlaVHeadSize": embeddimV})
        RUN_PARAM = json.dumps({"seqLen": kv_seqLen})
        kv_seqLen = np.array(kv_seqLen)
        kv_seqLen = torch.from_numpy(kv_seqLen).to(torch.int32).npu()
       
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [self.q.npu(), self.k.npu(), self.mask.to(data_type).npu(), kv_seqLen])


    def test_flash_attention_mla_bf16(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        # unpad encoder
        batch = 16
        kv_head = 8        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 512
        max_seq = 2048
        tor = 1
        dynamic_batch = False
        kv_seqLen = [500, 1024, 1500, 2048] * 4
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "SelfAttentionOperation"
        OP_PARAM = {"type": 2008, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV, "kvHead": kv_head,"headSize": heads, "tor": tor}

        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH)
        self.gen_out_tensor()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))

        PARAM = json.dumps({"headNum": heads, "qkScale": float(1.0 / math.sqrt(1.0 * self.embeddim)), "kvHeadNum": kv_head, \
            "calcType": 3, "maskType": 1, "mlaVHeadSize": embeddimV})
        RUN_PARAM = json.dumps({"seqLen": kv_seqLen})
        kv_seqLen = np.array(kv_seqLen)
        kv_seqLen = torch.from_numpy(kv_seqLen).to(torch.int32).npu()

        attention_out = np.zeros_like(self.golden_out.to(torch.float16))
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [self.q.npu(), self.k.npu(), self.mask.to(data_type).npu(), kv_seqLen])


    def test_flash_attention_mla_bf16_case2(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        # unpad encoder
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 512
        max_seq = 2048
        tor = 1
        dynamic_batch = False
        kv_seqLen = [500]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "SelfAttentionOperation"
        OP_PARAM = {"type": 2008, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV, "kvHead": kv_head,"headSize": heads, "tor": tor}

        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH)
        self.gen_out_tensor()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))

        PARAM = json.dumps({"headNum": heads, "qkScale": float(1.0 / math.sqrt(1.0 * self.embeddim)), "kvHeadNum": kv_head, \
            "calcType": 3, "maskType": 1, "mlaVHeadSize": embeddimV})
        RUN_PARAM = json.dumps({"seqLen": kv_seqLen})
        kv_seqLen = np.array(kv_seqLen)
        kv_seqLen = torch.from_numpy(kv_seqLen).to(torch.int32).npu()


        attention_out = np.zeros_like(self.golden_out.to(torch.float16))
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [self.q.npu(), self.k.npu(), self.mask.to(data_type).npu(), kv_seqLen])
        
    def test_flash_attention_mla_high_precision(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        # unpad encoder
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 512
        max_seq = 2048
        tor = 1
        dynamic_batch = False
        kv_seqLen = [500]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "SelfAttentionOperation"
        OP_PARAM = {"type": 2008, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV, "kvHead": kv_head,"headSize": heads, "tor": tor}

        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH)
        self.gen_out_tensor()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))

        PARAM = json.dumps({"headNum": heads, "qkScale": float(1.0 / math.sqrt(1.0 * self.embeddim)), "kvHeadNum": kv_head, \
            "calcType": 3, "maskType": 1, "mlaVHeadSize": embeddimV, "kernelType": 1})
        RUN_PARAM = json.dumps({"seqLen": kv_seqLen})
        kv_seqLen = np.array(kv_seqLen)
        kv_seqLen = torch.from_numpy(kv_seqLen).to(torch.int32).npu()


        attention_out = np.zeros_like(self.golden_out.to(torch.float16))
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [self.q.npu(), self.k.npu(), self.mask.to(data_type).npu(), kv_seqLen])
        
    def test_flash_attention_mla_fp16_high_precision(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        # unpad encoder
        batch = 1
        kv_head = 16        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 16          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 512
        max_seq = 2048
        tor = 1
        dynamic_batch = False
        kv_seqLen = [500]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "SelfAttentionOperation"
        OP_PARAM = {"type": 2010, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV,"kvHead": kv_head,"headSize": heads, "tor": tor}

        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH)
        self.gen_out_tensor()
        self.mask = np.reshape(self.mask, (max_seq, max_seq))

        attention_out = np.zeros_like(self.golden_out)
       
        PARAM = json.dumps({"headNum": heads, "qkScale": float(1.0 / math.sqrt(1.0 * self.embeddim)), "kvHeadNum": kv_head, \
            "calcType": 3, "maskType": 1, "mlaVHeadSize": embeddimV, "kernelType": 1})
        RUN_PARAM = json.dumps({"seqLen": kv_seqLen})
        kv_seqLen = np.array(kv_seqLen)
        kv_seqLen = torch.from_numpy(kv_seqLen).to(torch.int32).npu()
       
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [self.q.npu(), self.k.npu(), self.mask.to(data_type).npu(), kv_seqLen])
        
    def test_flash_attention_mla_mask0_bf16(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        # unpad encoder
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 576
        max_seq = 2048
        tor = 1
        dynamic_batch = False
        kv_seqLen = [500]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "SelfAttentionOperation"
        OP_PARAM = {"type": 2008, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV, "kvHead": kv_head,"headSize": heads, "tor": tor}

        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False, is_mask = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH)
        self.gen_out_tensor()

        PARAM = json.dumps({"headNum": heads, "qkScale": float(1.0 / math.sqrt(1.0 * self.embeddim)), "kvHeadNum": kv_head, \
            "calcType": 3, "maskType": 0, "mlaVHeadSize": embeddimV, "kernelType": 0})
        RUN_PARAM = json.dumps({"seqLen": kv_seqLen})
        kv_seqLen = np.array(kv_seqLen)
        kv_seqLen = torch.from_numpy(kv_seqLen).to(torch.int32).npu()


        attention_out = np.zeros_like(self.golden_out.to(torch.float16))
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [self.q.npu(), self.k.npu(), kv_seqLen])
    
    def test_flash_attention_mla_mask0(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        # unpad encoder
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 576
        max_seq = 2048
        tor = 1
        dynamic_batch = False
        kv_seqLen = [500]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "SelfAttentionOperation"
        OP_PARAM = {"type": 2008, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV, "kvHead": kv_head,"headSize": heads, "tor": tor}

        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False, is_mask = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH)
        self.gen_out_tensor()

        PARAM = json.dumps({"headNum": heads, "qkScale": float(1.0 / math.sqrt(1.0 * self.embeddim)), "kvHeadNum": kv_head, \
            "calcType": 3, "maskType": 0, "mlaVHeadSize": embeddimV, "kernelType": 0})
        RUN_PARAM = json.dumps({"seqLen": kv_seqLen})
        kv_seqLen = np.array(kv_seqLen)
        kv_seqLen = torch.from_numpy(kv_seqLen).to(torch.int32).npu()


        attention_out = np.zeros_like(self.golden_out.to(torch.float16))
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [self.q.npu(), self.k.npu(), kv_seqLen])
        
    def test_flash_attention_mla_mask0_bf16_high_precision(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        # unpad encoder
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 576
        max_seq = 2048
        tor = 1
        dynamic_batch = False
        kv_seqLen = [500]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "SelfAttentionOperation"
        OP_PARAM = {"type": 2010, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV, "kvHead": kv_head,"headSize": heads, "tor": tor}

        data_type = torch.bfloat16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False, is_mask = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH)
        self.gen_out_tensor()

        PARAM = json.dumps({"headNum": heads, "qkScale": float(1.0 / math.sqrt(1.0 * self.embeddim)), "kvHeadNum": kv_head, \
            "calcType": 3, "maskType": 0, "mlaVHeadSize": embeddimV, "kernelType": 1})
        RUN_PARAM = json.dumps({"seqLen": kv_seqLen})
        kv_seqLen = np.array(kv_seqLen)
        kv_seqLen = torch.from_numpy(kv_seqLen).to(torch.int32).npu()


        attention_out = np.zeros_like(self.golden_out.to(torch.float16))
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [self.q.npu(), self.k.npu(), kv_seqLen])
    
    def test_flash_attention_mla_mask0_high_precision(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        # unpad encoder
        batch = 1
        kv_head = 1        # kv_head num
        isdecoder = 0       # prefill or decoder
        heads = 8          # llama7b  hidden_size 4096
        embeddim = 576
        embeddimV = 576
        max_seq = 2048
        tor = 1
        dynamic_batch = False
        kv_seqLen = [500]
        is_clamp = 0
        clamp_min = 0
        clamp_max = 0
        OP_NAME = "SelfAttentionOperation"
        OP_PARAM = {"type": 2010, "qSeqLen":kv_seqLen, "kvSeqLen": kv_seqLen, "headDimV": embeddimV, "kvHead": kv_head,"headSize": heads, "tor": tor}

        data_type = torch.float16

        self.set_data_params(dynamic_batch = dynamic_batch,
                             is_decoder = isdecoder, batch = batch, kv_head = kv_head, heads = heads,
                             embeddim = embeddim,embeddimv = embeddimV, max_seq = max_seq, kv_seqLen = kv_seqLen,
                             is_clamp = is_clamp, clamp_max = clamp_max, clamp_min = clamp_min,
                             data_type = data_type, is_alibi = False, is_mask = False,
                             op_type = OP_PARAM["type"], mask_type = MASK_TYPE_NO_BATCH)
        self.gen_out_tensor()

        PARAM = json.dumps({"headNum": heads, "qkScale": float(1.0 / math.sqrt(1.0 * self.embeddim)), "kvHeadNum": kv_head, \
            "calcType": 3, "maskType": 0, "mlaVHeadSize": embeddimV, "kernelType": 1})
        RUN_PARAM = json.dumps({"seqLen": kv_seqLen})
        kv_seqLen = np.array(kv_seqLen)
        kv_seqLen = torch.from_numpy(kv_seqLen).to(torch.int32).npu()


        attention_out = np.zeros_like(self.golden_out.to(torch.float16))
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [self.q.npu(), self.k.npu(), kv_seqLen])
   
if __name__ == '__main__':
    unittest.main()