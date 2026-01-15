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

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402


OP_NAME = "SelfAttentionOperation"

seed = 42
torch.manual_seed(seed)
class TestUnpadSelfAttentionOperation(operation_test.OperationTest):
    def test_swa_large_window(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        min_seqlen = 2
        max_seqlen = 5
        self.batch = 2
        self.layer = 4
        window_size = 2 ** 32 -1
        seqlen = torch.randint(min_seqlen, max_seqlen, (self.batch,), dtype=torch.int32).npu()
        min_token_offset_start = 0
        max_token_offset_start = 5
        token_offset_start = torch.randint(min_token_offset_start, max_token_offset_start, (self.batch,), dtype=torch.int32).npu()
        token_offset = token_offset_start + seqlen
        total_seqlen = max_token_offset_start + max_seqlen
        ntokens = int(seqlen.sum())
        self.head_num = 4
        self.head_size = 128
        self.head_size_v = 128
        hidden_size = self.head_num * self.head_size
        hidden_size_v = self.head_num * self.head_size_v
        mixed_q = torch.rand(ntokens, hidden_size, dtype=torch.float16).npu()
        mixed_k = torch.rand(ntokens, hidden_size, dtype=torch.float16).npu()
        mixed_v = torch.rand(ntokens, hidden_size_v, dtype=torch.float16).npu()
        cache_k = torch.rand(self.layer, self.batch, total_seqlen, hidden_size, dtype=torch.float16).npu()
        cache_v = torch.rand(self.layer, self.batch, total_seqlen, hidden_size_v, dtype=torch.float16).npu()
        
        # attention_mask = torch.zeros(self.batch, total_seqlen, total_seqlen, dtype=torch.float16).npu()
        mask = np.ones(shape=(total_seqlen, total_seqlen)).astype(np.float16)  # 使用当前最大seqlen生成mask
        mask_u = np.triu(mask, 1)
        mask_l = np.tril(mask, -total_seqlen)
        mask = mask_u + mask_l
        mask *= -10000.0
        attention_mask = torch.from_numpy(mask).to(torch.float16).npu()
        layerid = torch.randint(self.layer, (1,), dtype=torch.int32).npu()

        self.q_scale = 0.2
        self.qk_scale = 1
        param = json.dumps({"headNum": self.head_num, "qScale": float(self.q_scale), "qkScale": float(self.qk_scale), "maskType": 7,
                            "windowSize":window_size})
        self.param_seqlen = seqlen.tolist()
        self.param_token_offset = token_offset.tolist()
        run_param = json.dumps({"tokenOffset": self.param_token_offset, "seqLen": self.param_seqlen})

        self.execute_with_param(OP_NAME, param, run_param,
                     [mixed_q, mixed_k, mixed_v, cache_k, cache_v, attention_mask, token_offset, seqlen, layerid])
    
    

    def gen_swa_cmp(self, window_size, embeddim):
        swa_mask = np.ones(shape=(1, 512, 512)) * -10000.0
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
    
    def golden_calc(self, in_tensors):
        mixed_q = in_tensors[0]
        mixed_k = in_tensors[1]
        mixed_v = in_tensors[2]
        cache_k = in_tensors[3]
        cache_v = in_tensors[4]
        if len(in_tensors) == 9:
            attention_mask = in_tensors[5]
            layerid = int(in_tensors[8][0])
        elif len(in_tensors) == 8:
            layerid = int(in_tensors[7][0])
            # max_seq = max(cur_seqlen, cur_token_offset)
            attention_mask = np.ones(shape=(1, 5)).astype(np.float16) * -10000.0 # 使用当前最大seqlen生成mask
            kvseqlen = 5
            window_size = 3
            if self.cache_type == 1:
                attention_mask[:, :window_size] = 0
            else:
                attention_mask[:, kvseqlen - window_size: kvseqlen] = 0
            # print("mask: ", attention_mask)
            # mask_u = np.triu(attention_mask, 1)
            # mask_l = np.tril(attention_mask, -3)
            # attention_mask = mask_u + mask_l
            # attention_mask *= -10000.0
            attention_mask = torch.from_numpy(attention_mask).npu()
        offset = 0
        context_list = []
        for i, _ in enumerate(range(self.batch)):
            cur_seqlen = self.param_seqlen[i]
            cur_token_offset = self.param_token_offset[i]
            cur_token_offset_start = cur_token_offset - cur_seqlen
            next_offset = offset + cur_seqlen
            cur_q = mixed_q[offset:next_offset]
            cur_k = mixed_k[offset:next_offset]
            cur_v = mixed_v[offset:next_offset]
            if cur_token_offset_start > 0:
                past_k = cache_k[layerid, i, :cur_token_offset_start, :]
                past_v = cache_v[layerid, i, :cur_token_offset_start, :]
                cur_k = torch.concat([past_k, cur_k], dim=0)
                cur_v = torch.concat([past_v, cur_v], dim=0)
            cur_q = (cur_q * self.q_scale).view(cur_seqlen, self.head_num, self.head_size).transpose(0, 1)
            cur_k = cur_k.view(cur_token_offset, self.head_num, self.head_size).permute(1, 2, 0)
            cur_qk = torch.bmm(cur_q, cur_k) # [head_num, seqlen, token_offset]

            if attention_mask.ndim == 3: # masked_fill
                cur_qk = cur_qk + attention_mask[i, :cur_seqlen, :cur_token_offset]
            if attention_mask.ndim == 2:
                if attention_mask.shape[0] == 512 and attention_mask.shape[1] == 512:
                    window_size = 256
                    max_seq = max(cur_seqlen, cur_token_offset)
                    mask = np.ones(shape=(max_seq, max_seq)).astype(np.float16)  # 使用当前最大seqlen生成mask
                    mask_u = np.triu(mask, 1)
                    mask_l = np.tril(mask, -window_size)
                    mask = mask_u + mask_l
                    mask *= -10000.0
                    mask = torch.from_numpy(mask).npu()
                    cur_qk = cur_qk + mask[:cur_seqlen, :cur_token_offset]
                else:
                    cur_qk = cur_qk + attention_mask[:cur_seqlen, :cur_token_offset]
            cur_qk = cur_qk * self.qk_scale
            cur_qk = torch.nn.functional.softmax(cur_qk.type(torch.float32), dim=-1).type(torch.float16)

            cur_v = cur_v.view(cur_token_offset, self.head_num, self.head_size_v).transpose(0, 1)

            cur_context = torch.bmm(cur_qk, cur_v).transpose(0, 1).contiguous().view(cur_seqlen, self.head_num * self.head_size_v)
            context_list.append(cur_context)

            offset = next_offset

        context = torch.concat(context_list, dim=0)
        return [context]
    
    def golden_compare(self, out_tensor, golden_out_tensor):
        # print("out_tensor")
        # print(out_tensor.cpu())
        # print("golden_out_tensor")
        # print(golden_out_tensor.cpu())
        # return torch.allclose(out_tensor, golden_out_tensor, rtol=0.001, atol=0.001)
        print("run test_self_attention_operation_swa_large_window")
        out = out_tensor
        golden = golden_out_tensor
        ratios = [0.001, 0.001, 0.003, 0.003, 0.005, 0.005]
        embeddim = 128
        max_seq = 1280
        error_count = 0
        strict_error_count = 0
        alibi_error_count = 0
        fp16_min_normal = 1.0 / (1 << 14)
        out = out.flatten()
        golden = golden.flatten()
        out_len = out.shape[0]
        diff = torch.abs(golden - out)
        max_diff = diff.max().item()
        print("maxDiff: " , max_diff)
        golden = golden.to(torch.float32)
        out = out.to(torch.float32)


        limit_error = torch.maximum(
            torch.abs(golden * ratios[0]), torch.tensor(ratios[1]))
        strict_limit_error = torch.maximum(
            torch.abs(golden * ratios[2]), torch.tensor(ratios[3]))
        error_count = torch.gt(diff, limit_error).sum().item()
        strict_error_count = torch.gt(
            diff, strict_limit_error).sum().item()
        print("1/1000 Accuracy is: ",
                        1 - float(error_count) / out_len)
        print("3/1000 Accuracy is ",  1 -
                        float(strict_error_count) / out_len)
        print("accuracy is correct: ", (float(strict_error_count) / out_len) <= ratios[0])  
        # 新精度标准fp16 参考精度标准v0.3浮点计算单标杆
        # 计算次数 两个matmul + 一个softmax
        calc_times = embeddim * max_seq + 4
        # import pdb;pdb.set_trace()
        if calc_times < 2048:
            error = 2**(-8)
        else :
            error = 2**(-7)
        error_threshold = torch.clamp(torch.abs(golden), min = 1) * error
        return (diff <= error_threshold).all()

if __name__ == '__main__':
    unittest.main()
