#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import sys
import os
import unittest
import json
import math
import torch
import torch_npu
import numpy as np
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
import operation_test  # NOQA: E402
import pdb

OP_NAME = "SelfAttentionOperation"

MASK_TYPE_NO_HEAD_DECODER = 5
class TestUnpadSelfAttentionOperation(operation_test.OperationTest):
    def test_swa_decoder(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        mask_type = MASK_TYPE_NO_HEAD_DECODER
        self.data_type = torch.float16
        data_type = self.data_type
        self.batch = 8
        batch = self.batch
        self.kv_head = 32        # kv_head num
        kv_head = self.kv_head
        self.is_decoder = 1       # prefill or decoder
        self.heads = 32          # llama7b  hidden_size 4096
        self.embeddim = 128
        self.embeddim_v = self.embeddim
        self.max_seq = 256
        tor = 1
        self.dynamic_batch = False
        kv_seqLen = [114] * batch
        qSeqLen = [1] * batch
        self.window_size = 16
        self.cacheType = 0
        self.is_clamp = 0
        self.clamp_min = 0
        self.clamp_max = 0
        self.is_triu_mask = False
        self.long_seq = False
        self.is_alibi = False
        self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, [1] * batch)
        self.kv_seqlen, self.kv_ntokens = self.gen_seq_len(batch, kv_seqLen)
        self.layer_id = torch.from_numpy(np.array([0], dtype=np.int32)).to(torch.int32)
        self.q_max_seq = np.max(self.q_seqlen)
        self.kv_max_seq = np.max(self.kv_seqlen)
        q = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.q_ntokens, self.heads * self.embeddim)))
        tor = np.float32(1.0 / math.sqrt(1.0 * self.embeddim))
        #self.q = (q * tor).to(data_type)
        self.q = q.to(data_type)
        self.k = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, self.batch, self.max_seq, kv_head * self.embeddim))).to(data_type)
        self.v = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, self.batch, self.max_seq, kv_head * self.embeddim_v))).to(data_type)


        self.q_scale = 1
        self.qk_scale = tor
        param = json.dumps({"headNum": self.heads, "qScale": float(self.q_scale), "qkScale": float(self.qk_scale), "maskType": 7, "kvcacheCfg":1,"calcType":2,
                            "windowSize":self.window_size})
        self.param_seqlen = self.q_seqlen
        self.param_token_offset = self.kv_seqlen
        run_param = json.dumps({"tokenOffset": self.param_token_offset, "seqLen": self.param_seqlen, "maskType": 7})
        #pdb.set_trace()
        self.execute_with_param(OP_NAME, param, run_param,
                     [self.q.npu(), self.k.npu(), self.v.npu(),torch.tensor(self.kv_seqlen).to(torch.int32).npu(), torch.tensor(self.q_seqlen).to(torch.int32).npu(), self.layer_id.npu()])
    
    def gen_seq_len(self, batch, seq_len):
        ntokens = sum(seq_len)
        return seq_len, ntokens
    
    def gen_mask(self, batch, heads, data_type, mask_type):
        import random
        q_max_seq = self.max_seq
        kv_max_seq = self.max_seq
        mask_type_dict = {
            # 三维的alibi mask
            #MASK_TYPE_NO_HEAD : ((batch, q_max_seq, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),
            MASK_TYPE_NO_HEAD_DECODER : ((batch, 1, kv_max_seq), (lambda mask, idx, q_s, kv_s: (mask[idx, :q_s, :kv_s]))),

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
        # print("-------------------",self.mask_info[0])
        mask = np.ones(shape=self.mask_info[0]) * pre_mask_coff
        mask = np.triu(mask, 1)
        zero_indice = random.choices(range(self.max_seq), k = 300)
        if self.is_alibi:
            self.alibi_bias = self.get_alibi_bias(heads, self.max_seq)
            mask += self.alibi_bias.numpy()
        if select_zero:
            mask.flat[zero_indice] = 0
        self.mask = torch.from_numpy(mask).to(torch.float32)
        #self.mask[0]=self.mask[1]
        #self.mask = torch.zeros(self.mask.shape)
        self.post_mask_coff = post_mask_coff
        self.pre_mask_coff = pre_mask_coff

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

    def compare_output_data(self, out, golden, ratios):
        error_count = 0
        strict_error_count = 0
        alibi_error_count = 0
        fp16_min_normal = 1.0 / (1 << 14)
        len = out.shape[0] * out.shape[1]
        diff = torch.abs(golden - out)
        max_diff = diff.max().item()
        print("maxDiff:", max_diff)


        limit_error = torch.maximum(torch.abs(golden * ratios[0]), torch.tensor(ratios[1]))
        strict_limit_error = torch.maximum(torch.abs(golden * ratios[2]), torch.tensor(ratios[3]))
        error_count = torch.gt(diff, limit_error).sum().item()
        strict_error_count = torch.gt(diff, strict_limit_error).sum().item()
        print("1/1000 Accuracy is ",  1 - float(error_count) / len)
        print("3/1000 Accuracy is ",  1 - float(strict_error_count) / len)
        if self.data_type == torch.bfloat16 or not self.is_decoder:
            return (float(strict_error_count) / len) <= ratios[2]
        else:
            return (float(error_count) / len) <= ratios[0]

    def golden_calc(self, in_tensors):
        q_offset = 0
        k_offset = 0
        v_offset = 0
        isdecoder = 1 
        batch = self.batch
        heads = self.heads
        embed = self.embeddim
        embed_v = self.embeddim_v
        max_seq = self.max_seq
        q_seqlen = self.q_seqlen
        kv_seqlen = self.kv_seqlen
        kv_head = self.kv_head
        
        is_mask = True
        q = self.q
        k = self.k
        v = self.v
        q_ntokens = self.q_ntokens
        kv_ntokens = self.kv_ntokens
        layer_id = self.layer_id[0]
        self.is_multi_layer = True
        s = None
        _p = None
        out = None

        for idx in range(batch):
            q_s = q_seqlen[idx]
            kv_s = kv_seqlen[idx]
            q_slice = q[q_offset:q_offset + q_s][:]
            q_slice = q_slice.view(q_s, heads, embed)
            q_slice = torch.permute(q_slice, (1, 0, 2))
            k_slice = k[layer_id][idx][:kv_s][:]
            k_slice = k_slice.view(kv_s, kv_head, embed)
            k_slice_t = torch.permute(k_slice, (1, 2, 0))   # get K^T
            v_slice = v[layer_id][idx][:kv_s][:]
            v_slice = v_slice.view(kv_s, kv_head, embed_v)
            v_slice = torch.permute(v_slice, (1, 0, 2))

            score = self.group_mm_torch(heads, kv_head, q_slice, k_slice_t)

            if s is None:
                s = score.view([-1, ])
            else:
                s = torch.cat((s, score.view([-1, ])), 0)

            scale = 1
            tor = np.float32(1.0 / math.sqrt(1.0 * self.embeddim))
            if not self.is_multi_layer:
                # 当前scale和tor保持一致，模型侧可能传入scale = np.float32(layer_id + 1)
                scale = np.float32(layer_id + 1)
            score = score * tor

            if self.is_clamp == 1:
                clamp_min_brc = np.ones((score.shape)) * self.clamp_min
                clamp_max_brc = np.ones((score.shape)) * self.clamp_max
                score = np.float16(np.maximum(score, clamp_min_brc))
                score = torch.from_numpy(np.float16(np.minimum(score, clamp_max_brc)))
            if len(in_tensors) == 6:
                attention_mask = np.ones(shape=(1, kv_s)).astype(np.float16) * -10000.0 # 使用当前最大seqlen生成mask
                # attention_mask[:, :self.window_size] = 0
                if self.cacheType == 0:
                    attention_mask[:, kv_s - self.window_size: kv_s] = 0
                else:
                    attention_mask[:, :self.window_size] = 0
                attention_mask = torch.from_numpy(attention_mask)
            else:
                attention_mask = in_tensors[3].cpu()
                if attention_mask.shape[0] == 512 and attention_mask.shape[1] == 512:
                    mask = np.ones(shape=(kv_s, kv_s)).astype(np.float16)  # 使用当前最大seqlen生成mask
                    mask_u = np.triu(mask, 1)
                    mask_l = np.tril(mask, -self.window_size)
                    mask = mask_u + mask_l
                    if attention_mask.dtype == torch.float16:
                        mask *= -10000.0
                    else:
                        mask *= -3e38
                    attention_mask = torch.from_numpy(mask)
                    
                else:
                    if attention_mask.dtype == torch.bfloat16:
                        attention_mask *= -3e38
            
            score = score + attention_mask[:q_s, :kv_s]
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

            p = (score_exp / score_sum.reshape((heads, q_s, 1)))
            p = torch.from_numpy(p).to(torch.bfloat16)
            o = self.group_mm_torch(heads, kv_head, p, v_slice)
            o = o.view(heads, q_s, embed_v)
            o = torch.permute(o, (1, 0, 2)).contiguous()
            if out is None:
                out = o
            else:
                out = torch.cat((out, o), 0)

            q_offset += q_s
            k_offset += max_seq
            v_offset += max_seq
        
        # golden data
        out = out.view(q_ntokens, heads * embed_v)
        self.golden_out = out.to(self.data_type)
        return [self.golden_out]
    
    def golden_compare(self, out_tensor, golden_out_tensor):
        # print("out_tensor", out_tensor.cpu())
        # print("golden_out_tensor", golden_out_tensor.cpu())
        return self.compare_output_data(out_tensor.cpu(), golden_out_tensor, [0.001, 0.001, 0.003, 0.003, 0.005, 0.005])
        #return torch.allclose(out_tensor.cpu(), golden_out_tensor, rtol=0.001, atol=0.001)
    
    def test_swa_decoder_cache(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        mask_type = MASK_TYPE_NO_HEAD_DECODER    
        self.data_type = torch.bfloat16
        data_type = self.data_type
        self.batch = 8
        batch = self.batch
        self.kv_head = 32        # kv_head num
        kv_head = self.kv_head
        self.is_decoder = 1       # prefill or decoder
        self.heads = 32          # llama7b  hidden_size 4096
        self.embeddim = 128
        self.embeddim_v = self.embeddim
        self.max_seq = 1024
        tor = 1
        self.dynamic_batch = False
        kv_seqLen = [32, 1024] * 4
        qSeqLen = [1] * batch
        self.window_size = 64
        self.is_clamp = 0
        self.clamp_min = 0
        self.cacheType = 1
        self.clamp_max = 0
        self.is_triu_mask = False
        self.long_seq = False
        self.is_alibi = False
        self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, [1] * batch)
        self.kv_seqlen, self.kv_ntokens = self.gen_seq_len(batch, kv_seqLen)
        self.layer_id = torch.from_numpy(np.array([0], dtype=np.int32)).to(torch.int32)
        self.q_max_seq = np.max(self.q_seqlen)
        self.kv_max_seq = np.max(self.kv_seqlen)
        q = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.q_ntokens, self.heads * self.embeddim)))
        tor = np.float32(1.0 / math.sqrt(1.0 * self.embeddim))
        #self.q = (q * tor).to(data_type)
        self.q = q.to(data_type)
        self.k = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, self.batch, self.max_seq, kv_head * self.embeddim))).to(data_type)
        self.v = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, self.batch, self.max_seq, kv_head * self.embeddim_v))).to(data_type)
        self.gen_mask(self.batch, self.heads, data_type, mask_type)

        self.q_scale = 1
        self.qk_scale = tor
        param = json.dumps({"headNum": self.heads, "qScale": float(self.q_scale), "qkScale": float(self.qk_scale), "maskType": 7, "kvcacheCfg":1,"calcType":2,
                            "windowSize":self.window_size, "cacheType":1})
        self.param_seqlen = self.q_seqlen
        self.param_token_offset = self.kv_seqlen
        run_param = json.dumps({"tokenOffset": self.param_token_offset, "seqLen": self.param_seqlen, "maskType": 7})
        #pdb.set_trace()
        self.execute_with_param(OP_NAME, param, run_param,
                     [self.q.npu(), self.k.npu(), self.v.npu(),torch.tensor(self.kv_seqlen).to(torch.int32).npu(), torch.tensor(self.q_seqlen).to(torch.int32).npu(), self.layer_id.npu()])
    def gen_swa_mask(self, max_seq, window_size, pre_mask_coff, cache_type=0):
        swa_mask = np.ones(shape=(max_seq, max_seq)) * pre_mask_coff

        if window_size < max_seq or self.is_compress:
            triu_mask = np.triu(swa_mask, 1)
            tril_mask = np.tril(swa_mask, -window_size)
            swa_mask = triu_mask + tril_mask
        else:
            swa_mask = np.triu(swa_mask, 1)
        
        return swa_mask
    def gen_swa_cmp(self, window_size, embeddim, pre_mask_coff):
        swa_mask = np.ones(shape=(1, 512, 512)) * pre_mask_coff
        pp_n = 128 if embeddim <= 128 else 64
        # pp_n = 128
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
        swa_mask = torch.from_numpy(swa_mask).to(torch.float16)
        swa_mask = swa_mask.reshape(512,512)
        return swa_mask
    def test_swa_encoder_cache(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        mask_type = MASK_TYPE_NO_HEAD_DECODER    
        self.data_type = torch.float16
        data_type = self.data_type
        self.batch = 8
        batch = self.batch
        self.kv_head = 32        # kv_head num
        kv_head = self.kv_head
        self.is_decoder = 1       # prefill or decoder
        self.heads = 32          # llama7b  hidden_size 4096
        self.embeddim = 128
        self.embeddim_v = self.embeddim
        self.max_seq = 1024
        tor = 1
        self.dynamic_batch = False
        kv_seqLen = [32, 1024] * 4
        qSeqLen = kv_seqLen
        self.window_size = 16
        self.is_clamp = 0
        self.clamp_min = 0
        self.cacheType = 1
        self.clamp_max = 0
        self.is_triu_mask = False
        self.long_seq = False
        self.is_alibi = False
        self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, kv_seqLen)
        self.kv_seqlen, self.kv_ntokens = self.gen_seq_len(batch, kv_seqLen)
        self.layer_id = torch.from_numpy(np.array([0], dtype=np.int32)).to(torch.int32)
        self.q_max_seq = np.max(self.q_seqlen)
        self.kv_max_seq = np.max(self.kv_seqlen)
        q = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.q_ntokens, self.heads * self.embeddim)))
        tor = np.float32(1.0 / math.sqrt(1.0 * self.embeddim))
        #self.q = (q * tor).to(data_type)
        self.q = q.to(data_type)
        self.k = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, self.batch, self.max_seq, kv_head * self.embeddim))).to(data_type)
        self.v = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, self.batch, self.max_seq, kv_head * self.embeddim_v))).to(data_type)
        mask = np.ones(shape=(self.q_max_seq, self.kv_max_seq)).astype(np.float16)  # 使用当前最大seqlen生成mask
        mask_u = np.triu(mask, 1)
        mask_l = np.tril(mask, -self.window_size)
        mask = mask_u + mask_l
        # mask *= -3e38
        

        mask *= -10000.0
        # mask = self.gen_swa_mask(self.kv_max_seq, self.window_size, -10000.0, self.cacheType)
        # print(torch.from_numpy(mask).to(data_type))
        attention_mask = torch.from_numpy(mask).to(data_type).npu()
        

        self.q_scale = 1
        self.qk_scale = tor
        param = json.dumps({"headNum": self.heads, "qScale": float(self.q_scale), "qkScale": float(self.qk_scale), "maskType": 7, "kvcacheCfg":1,"calcType":1,
                            "windowSize":self.window_size, "cacheType":self.cacheType})
        self.param_seqlen = self.q_seqlen
        self.param_token_offset = self.kv_seqlen
        run_param = json.dumps({"tokenOffset": self.param_token_offset, "seqLen": self.param_seqlen, "maskType": 7})
        #pdb.set_trace()
        self.execute_with_param(OP_NAME, param, run_param,
                     [self.q.npu(), self.k.npu(), self.v.npu(),attention_mask,torch.tensor(self.kv_seqlen).to(torch.int32).npu(), torch.tensor(self.q_seqlen).to(torch.int32).npu(), self.layer_id.npu()])
    
    def test_swa_encoder(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        mask_type = MASK_TYPE_NO_HEAD_DECODER    
        self.data_type = torch.bfloat16
        data_type = self.data_type
        self.batch = 8
        batch = self.batch
        self.kv_head = 32        # kv_head num
        kv_head = self.kv_head
        self.is_decoder = 1       # prefill or decoder
        self.heads = 32          # llama7b  hidden_size 4096
        self.embeddim = 128
        self.embeddim_v = self.embeddim
        self.max_seq = 1024
        tor = 1
        self.dynamic_batch = False
        kv_seqLen = [32, 256] * 4
        qSeqLen = kv_seqLen
        self.window_size = 16
        self.is_clamp = 0
        self.clamp_min = 0
        self.cacheType = 0
        self.clamp_max = 0
        self.is_triu_mask = False
        self.long_seq = False
        self.is_alibi = False
        self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, kv_seqLen)
        self.kv_seqlen, self.kv_ntokens = self.gen_seq_len(batch, kv_seqLen)
        self.layer_id = torch.from_numpy(np.array([0], dtype=np.int32)).to(torch.int32)
        self.q_max_seq = np.max(self.q_seqlen)
        self.kv_max_seq = np.max(self.kv_seqlen)
        q = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.q_ntokens, self.heads * self.embeddim)))
        tor = np.float32(1.0 / math.sqrt(1.0 * self.embeddim))
        #self.q = (q * tor).to(data_type)
        self.q = q.to(data_type)
        self.k = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, self.batch, self.max_seq, kv_head * self.embeddim))).to(data_type)
        self.v = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, self.batch, self.max_seq, kv_head * self.embeddim_v))).to(data_type)
        # mask = np.ones(shape=(self.q_max_seq, self.kv_max_seq)).astype(np.float16)  # 使用当前最大seqlen生成mask
        # mask_u = np.triu(mask, 1)
        # mask_l = np.tril(mask, -self.window_size)
        # mask = mask_u + mask_l
        # mask *= -3e38
        
        
        mask = self.gen_swa_mask(self.kv_max_seq, self.window_size, 1, self.cacheType)
        # print(torch.from_numpy(mask).to(data_type))
        attention_mask = torch.from_numpy(mask).to(data_type).npu()

        self.q_scale = 1
        self.qk_scale = tor
        param = json.dumps({"headNum": self.heads, "qScale": float(self.q_scale), "qkScale": float(self.qk_scale), "maskType": 7, "kvcacheCfg":1,"calcType":1,
                            "windowSize":self.window_size, "cacheType":self.cacheType})
        self.param_seqlen = self.q_seqlen
        self.param_token_offset = self.kv_seqlen
        run_param = json.dumps({"tokenOffset": self.param_token_offset, "seqLen": self.param_seqlen, "maskType": 7})
        #pdb.set_trace()
        self.execute_with_param(OP_NAME, param, run_param,
                     [self.q.npu(), self.k.npu(), self.v.npu(),attention_mask,torch.tensor(self.kv_seqlen).to(torch.int32).npu(), torch.tensor(self.q_seqlen).to(torch.int32).npu(), self.layer_id.npu()])
        
    def test_swa_encoder_compress_mask(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        mask_type = MASK_TYPE_NO_HEAD_DECODER    
        self.data_type = torch.bfloat16
        data_type = self.data_type
        self.batch = 8
        batch = self.batch
        self.kv_head = 32        # kv_head num
        kv_head = self.kv_head
        self.is_decoder = 1       # prefill or decoder
        self.heads = 32          # llama7b  hidden_size 4096
        self.embeddim = 128
        self.embeddim_v = self.embeddim
        self.max_seq = 1024
        tor = 1
        self.dynamic_batch = False
        kv_seqLen = [32, 256] * 4
        qSeqLen = kv_seqLen
        self.window_size = 16
        self.is_clamp = 0
        self.clamp_min = 0
        self.cacheType = 0
        self.clamp_max = 0
        self.is_triu_mask = False
        self.long_seq = False
        self.is_alibi = False
        self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, kv_seqLen)
        self.kv_seqlen, self.kv_ntokens = self.gen_seq_len(batch, kv_seqLen)
        self.layer_id = torch.from_numpy(np.array([0], dtype=np.int32)).to(torch.int32)
        self.q_max_seq = np.max(self.q_seqlen)
        self.kv_max_seq = np.max(self.kv_seqlen)
        q = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.q_ntokens, self.heads * self.embeddim)))
        tor = np.float32(1.0 / math.sqrt(1.0 * self.embeddim))
        #self.q = (q * tor).to(data_type)
        self.q = q.to(data_type)
        self.k = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, self.batch, self.max_seq, kv_head * self.embeddim))).to(data_type)
        self.v = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, self.batch, self.max_seq, kv_head * self.embeddim_v))).to(data_type)
        # mask = np.ones(shape=(self.q_max_seq, self.kv_max_seq)).astype(np.float16)  # 使用当前最大seqlen生成mask
        # mask_u = np.triu(mask, 1)
        # mask_l = np.tril(mask, -self.window_size)
        # mask = mask_u + mask_l
        # mask *= -10000.0
        # mask *= -3e38
        
        pre_mask_coff = 1
        attention_mask = self.gen_swa_cmp(self.window_size, self.embeddim, pre_mask_coff).to(data_type).npu()
        # print(attention_mask)


        self.q_scale = 1
        self.qk_scale = tor
        param = json.dumps({"headNum": self.heads, "qScale": float(self.q_scale), "qkScale": float(self.qk_scale), "maskType": 8, "kvcacheCfg":1,"calcType":1,
                            "windowSize":self.window_size, "cacheType":self.cacheType})
        self.param_seqlen = self.q_seqlen
        self.param_token_offset = self.kv_seqlen
        run_param = json.dumps({"tokenOffset": self.param_token_offset, "seqLen": self.param_seqlen, "maskType": 7})
        #pdb.set_trace()
        self.execute_with_param(OP_NAME, param, run_param,
                     [self.q.npu(), self.k.npu(), self.v.npu(),attention_mask,torch.tensor(self.kv_seqlen).to(torch.int32).npu(), torch.tensor(self.q_seqlen).to(torch.int32).npu(), self.layer_id.npu()])
        
    def test_swa_encoder_compress_mask_cache(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        mask_type = MASK_TYPE_NO_HEAD_DECODER    
        self.data_type = torch.float16
        data_type = self.data_type
        self.batch = 8
        batch = self.batch
        self.kv_head = 32        # kv_head num
        kv_head = self.kv_head
        self.is_decoder = 1       # prefill or decoder
        self.heads = 32          # llama7b  hidden_size 4096
        self.embeddim = 128
        self.embeddim_v = self.embeddim
        self.max_seq = 1024
        tor = 1
        self.dynamic_batch = False
        kv_seqLen = [32, 1024] * 4
        qSeqLen = kv_seqLen
        self.window_size = 16
        self.is_clamp = 0
        self.clamp_min = 0
        self.cacheType = 1
        self.clamp_max = 0
        self.is_triu_mask = False
        self.long_seq = False
        self.is_alibi = False
        self.q_seqlen, self.q_ntokens = self.gen_seq_len(batch, kv_seqLen)
        self.kv_seqlen, self.kv_ntokens = self.gen_seq_len(batch, kv_seqLen)
        self.layer_id = torch.from_numpy(np.array([0], dtype=np.int32)).to(torch.int32)
        self.q_max_seq = np.max(self.q_seqlen)
        self.kv_max_seq = np.max(self.kv_seqlen)
        q = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.q_ntokens, self.heads * self.embeddim)))
        tor = np.float32(1.0 / math.sqrt(1.0 * self.embeddim))
        #self.q = (q * tor).to(data_type)
        self.q = q.to(data_type)
        self.k = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, self.batch, self.max_seq, kv_head * self.embeddim))).to(data_type)
        self.v = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(self.layer_id[0] + 1, self.batch, self.max_seq, kv_head * self.embeddim_v))).to(data_type)
        # mask = np.ones(shape=(self.q_max_seq, self.kv_max_seq)).astype(np.float16)  # 使用当前最大seqlen生成mask
        # mask_u = np.triu(mask, 1)
        # mask_l = np.tril(mask, -self.window_size)
        # mask = mask_u + mask_l
        # mask *= -10000.0
        # mask *= -3e38
        
        pre_mask_coff = -10000.0
        attention_mask = self.gen_swa_cmp(self.window_size, self.embeddim, pre_mask_coff).to(data_type).npu()
        # print(attention_mask)


        self.q_scale = 1
        self.qk_scale = tor
        param = json.dumps({"headNum": self.heads, "qScale": float(self.q_scale), "qkScale": float(self.qk_scale), "maskType": 8, "kvcacheCfg":1,"calcType":1,
                            "windowSize":self.window_size, "cacheType":self.cacheType})
        self.param_seqlen = self.q_seqlen
        self.param_token_offset = self.kv_seqlen
        run_param = json.dumps({"tokenOffset": self.param_token_offset, "seqLen": self.param_seqlen, "maskType": 7})
        #pdb.set_trace()
        self.execute_with_param(OP_NAME, param, run_param,
                     [self.q.npu(), self.k.npu(), self.v.npu(),attention_mask,torch.tensor(self.kv_seqlen).to(torch.int32).npu(), torch.tensor(self.q_seqlen).to(torch.int32).npu(), self.layer_id.npu()])

if __name__ == '__main__':
    unittest.main()
