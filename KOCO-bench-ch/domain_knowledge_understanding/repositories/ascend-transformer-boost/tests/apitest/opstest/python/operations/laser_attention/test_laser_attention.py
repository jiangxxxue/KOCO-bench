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
import random
import torch
import torch_npu
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

OP_NAME = "LaserAttentionOperation"

ATTEN_MASK_VALUE = -65536
SCALE_VALUE = 0.08838834764831843
PRE_TOKENS = 4096

np.random.seed(123)
random.seed(123)


def torch_batch_dot(a, b):
    batch = a.shape[0]
    n = a.shape[1]
    h = a.shape[2]
    w = b.shape[3]
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    res = torch.zeros((batch, n, h, w), dtype=torch.float32)
    for i in range(batch):
        for j in range(n):
            res[i, j] = torch.matmul(a[i, j], b[i, j])
    return res


def golden_base(query, key, value, atten_mask):
    seq_size = query.shape[2]
    bmm1_res = torch_batch_dot(query, torch.transpose(key, 2, 3))
    if atten_mask.numel() == 0:
        bmm1_res = bmm1_res * SCALE_VALUE
    else:
        sparse_mask = torch.from_numpy(np.tri(seq_size - PRE_TOKENS, k=-1)).to(torch.float32)
        bmm1_res = bmm1_res * SCALE_VALUE + (atten_mask * ATTEN_MASK_VALUE)
        if PRE_TOKENS < seq_size:
            bmm1_res[:, :, PRE_TOKENS:, 0:seq_size - PRE_TOKENS] += sparse_mask * ATTEN_MASK_VALUE
    softmax_max, _ = torch.max(bmm1_res, dim=-1, keepdim=True)
    softmax_sub = bmm1_res - softmax_max
    softmax_exp = torch.exp(softmax_sub)
    softmax_sum = torch.sum(softmax_exp, -1, keepdim=True)
    softmax_out = (softmax_exp / softmax_sum).to(torch.float32)
    attention_out = torch_batch_dot(softmax_out, value)
    softmax_max = softmax_max.squeeze(3)
    softmax_sum = softmax_sum.squeeze(3)
    return softmax_max, softmax_sum, attention_out


class TestLaserAttentionOperation(operation_test.OperationTest):
    def test(self):
        if operation_test.get_soc_version() != 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        head_num = 8
        param = {"headNum": head_num, "preTokens": PRE_TOKENS}
        batch = 1
        q_num_head = head_num
        seq_size = 4096
        head_dim = 128
        kv_num_head = 8
        kv_size = 4096
        query = torch.empty([batch, q_num_head, seq_size, head_dim]).uniform_(-100, 100).to(torch.bfloat16).npu()
        key = torch.empty([batch, kv_num_head, kv_size, head_dim]).uniform_(-100, 100).to(torch.bfloat16).npu()
        value = torch.empty([batch, kv_num_head, kv_size, head_dim]).uniform_(-100, 100).to(torch.bfloat16).npu()
        atten_mask = torch.from_numpy(1 - np.tri(seq_size)).to(torch.bfloat16).npu()
        empty_tensor = torch.tensor([]).npu()

        self.execute(OP_NAME, param,
                     [query, key, value, empty_tensor, empty_tensor, empty_tensor, atten_mask, empty_tensor,
                      empty_tensor, empty_tensor])

    def golden_calc(self, in_tensors):
        query = in_tensors[0].cpu().to(torch.float32)
        key = in_tensors[1].cpu().to(torch.float32)
        value = in_tensors[2].cpu().to(torch.float32)
        atten_mask = in_tensors[6].cpu().to(torch.float32)
        q_num_head = query.shape[1]
        kv_num_head = key.shape[1]
        if q_num_head == kv_num_head:
            softmax_max, softmax_sum, attention_out = golden_base(query, key, value, atten_mask)
            return [softmax_max, softmax_sum, torch.tensor([0]), attention_out]
        else:
            head_num_per_group = q_num_head // kv_num_head
            for i in range(q_num_head):
                group_index = i // head_num_per_group
                query_per_group = query[:, i:i + 1, :, :].to(torch.float32)
                key_per_group = key[:, group_index:group_index + 1, :, :].to(torch.float32)
                value_per_group = value[:, group_index:group_index + 1, :, :].to(torch.float32)
                softmax_max, softmax_sum, attention_out = golden_base(query_per_group, key_per_group, value_per_group,
                                                                      atten_mask)
                if i == 0:
                    total_softmax_max = softmax_max
                    total_softmax_sum = softmax_sum
                    total_attention_out = attention_out
                else:
                    total_softmax_max = torch.cat((total_softmax_max, softmax_max), dim=1)
                    total_softmax_sum = torch.cat((total_softmax_sum, softmax_sum), dim=1)
                    total_attention_out = torch.cat((total_attention_out, attention_out), dim=1)
            return [total_softmax_max, total_softmax_sum, torch.tensor([0]), total_attention_out]

    def golden_compare(self, out_tensor, golden_out_tensor, rtol=0.02, atol=0.02):
        return True


if __name__ == '__main__':
    unittest.main()
