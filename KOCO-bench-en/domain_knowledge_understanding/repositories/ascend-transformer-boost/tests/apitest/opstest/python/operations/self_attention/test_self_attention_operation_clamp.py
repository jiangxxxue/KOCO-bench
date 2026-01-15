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


class TestUnpadSelfAttentionOperation(operation_test.OperationTest):
    def test(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        min_seqlen = 1
        max_seqlen = 5
        self.batch = 3
        self.layer = 4
        seqlen = torch.randint(min_seqlen, max_seqlen, (self.batch,), dtype=torch.int32).npu()
        min_token_offset_start = 0
        max_token_offset_start = 5
        token_offset_start = torch.randint(min_token_offset_start, max_token_offset_start, (self.batch,), dtype=torch.int32).npu()
        token_offset = token_offset_start + seqlen
        total_seqlen = max_token_offset_start + max_seqlen
        ntokens = int(seqlen.sum())
        self.head_num = 4
        self.head_size = 128
        self.head_size_v = 16 * np.random.randint(1,8)
        hidden_size = self.head_num * self.head_size
        hidden_size_v = self.head_num * self.head_size_v
        mixed_q = torch.rand(ntokens, hidden_size, dtype=torch.float16).npu()
        mixed_k = torch.rand(ntokens, hidden_size, dtype=torch.float16).npu()
        mixed_v = torch.rand(ntokens, hidden_size_v, dtype=torch.float16).npu()
        cache_k = torch.rand(self.layer, self.batch, total_seqlen, hidden_size, dtype=torch.float16).npu()
        cache_v = torch.rand(self.layer, self.batch, total_seqlen, hidden_size_v, dtype=torch.float16).npu()
        attention_mask = torch.zeros(self.batch, total_seqlen, total_seqlen, dtype=torch.float16).npu()
        layerid = torch.randint(self.layer, (1,), dtype=torch.int32).npu()

        self.q_scale = 0.2
        self.qk_scale = 1
        self.clampMin = 0.3333
        self.clampMax = 0.5555
        self.isClamp = 1
        param = json.dumps({"headNum": self.head_num, "qScale": float(self.q_scale), "qkScale": float(self.qk_scale), "clampMin": float(self.clampMin), "clampMax": float(self.clampMax), "clampType": 1, "maskType": 1})
        self.param_seqlen = seqlen.tolist()
        self.param_token_offset = token_offset.tolist()
        run_param = json.dumps({"tokenOffset": self.param_token_offset, "seqLen": self.param_seqlen})

        self.execute_with_param(OP_NAME, param, run_param,
                     [mixed_q, mixed_k, mixed_v, cache_k, cache_v, attention_mask, token_offset, seqlen, layerid])

    def golden_calc(self, in_tensors):
        layerid = int(in_tensors[8][0])
        mixed_q = in_tensors[0]
        mixed_k = in_tensors[1]
        mixed_v = in_tensors[2]
        cache_k = in_tensors[3]
        cache_v = in_tensors[4]
        attention_mask = in_tensors[5]
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
            if (self.isClamp == 1):
                cur_qk = torch.clamp(cur_qk, self.clampMin, self.clampMax)
            if attention_mask.ndim == 3: # masked_fill
                cur_qk = cur_qk + attention_mask[i, :cur_seqlen, :cur_token_offset]
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
        return torch.allclose(out_tensor, golden_out_tensor, rtol=0.001, atol=0.001)

if __name__ == '__main__':
    unittest.main()
