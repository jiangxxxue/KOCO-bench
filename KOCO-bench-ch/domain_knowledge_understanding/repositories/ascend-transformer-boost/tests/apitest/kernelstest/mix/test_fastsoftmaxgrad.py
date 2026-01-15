# 
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# 
import unittest
import numpy as np
import torch
import torch.nn.functional as F
import sys, os
import op_test


OP_NAME = "FastSoftMaxGradOperation"
OP_PARAM = {"qSeqLen": None, "headNum": 0}

def gen_softmax_grad(head_num, seq_len):
    x = torch.randn([head_num * seq_len, seq_len]).npu().to(torch.float16)
    x.requires_grad = True
    y = torch.softmax(x, dim=-1)
    y.retain_grad()
    w = torch.randn_like(x)
    loss = (w * y).sum()
    loss.backward()
    return (y.detach().cpu(), y.grad.detach().cpu(), x.grad.detach().cpu())

def calc_fast_softmax_grad_golden(y_input, y_grad, seq_len, head_num):
    y_input = y_input.to(torch.float32)
    y_grad = y_grad.to(torch.float32)
    result = torch.empty_like(y_grad)
    start = 0
    for l in seq_len:
        end = start + head_num * l * l
        cur_y_input = y_input[start:end]
        cur_y_grad = y_grad[start:end]
        inner_start = 0
        for j in range(l * head_num):
            inner_end = inner_start + l
            y_input_line = cur_y_input[inner_start:inner_end]
            y_grad_line = cur_y_grad[inner_start:inner_end]
            temp = torch.sum(y_input_line * y_grad_line, dim=0, keepdims=True)
            result[start + inner_start:start + inner_end] = (y_grad_line - temp) * y_input_line
            inner_start = inner_end
        start = end
    return result.to(torch.float16)

class TestFastSoftMaxGrad(op_test.OpTest):
    def golden_calc(self, in_tensors):
        return [self.golden]

    def golden_compare(self, out_tensors, golden_out_tensors):
        return torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)

    @op_test.only_910b
    def test_fastsoftmaxgrad1(self):
        batch_size_imm = 32
        head_num_imm = 8
        head_num = torch.Tensor([head_num_imm,]).to(torch.int64)
        seq_len = torch.randint(1, 100, [batch_size_imm,]).to(torch.int32)

        y_input_list = []
        y_grad_list = []
        golden_list = []
        for i in range(seq_len.shape[0]):
            yi, yg, gd = gen_softmax_grad(head_num_imm, seq_len[i])
            y_input_list.append(yi.reshape(-1))
            y_grad_list.append(yg.reshape(-1))
            golden_list.append(gd.reshape(-1))
        y_input = torch.cat(y_input_list)
        y_grad = torch.cat(y_grad_list)
        self.golden = torch.cat(golden_list)
        x_grad = torch.zeros_like(y_input)

        in_tensors = [y_input, y_grad]
        out_tensors = [x_grad]

        OP_PARAM['headNum'] = head_num_imm
        OP_PARAM['qSeqLen'] = seq_len.tolist()
        self.set_param(OP_NAME, OP_PARAM)
        self.execute(in_tensors, out_tensors)

    @op_test.only_910b
    def test_fastsoftmaxgrad2(self):
        batch_size_imm = 32
        head_num_imm = 8
        head_num = torch.Tensor([head_num_imm,]).to(torch.int64)
        seq_len = torch.randint(100, 1000, [batch_size_imm,]).to(torch.int32)

        y_input_list = []
        y_grad_list = []
        golden_list = []
        for i in range(seq_len.shape[0]):
            yi, yg, gd = gen_softmax_grad(head_num_imm, seq_len[i])
            y_input_list.append(yi.reshape(-1))
            y_grad_list.append(yg.reshape(-1))
            golden_list.append(gd.reshape(-1))
        y_input = torch.cat(y_input_list)
        y_grad = torch.cat(y_grad_list)
        self.golden = torch.cat(golden_list)
        x_grad = torch.zeros_like(y_input)

        in_tensors = [y_input, y_grad]
        out_tensors = [x_grad]

        OP_PARAM['headNum'] = head_num_imm
        OP_PARAM['qSeqLen'] = seq_len.tolist()
        self.set_param(OP_NAME, OP_PARAM)
        self.execute(in_tensors, out_tensors)

    @op_test.only_910b
    def test_fastsoftmaxgrad3(self):
        batch_size_imm = 32
        head_num_imm = 8
        head_num = torch.Tensor([head_num_imm,]).to(torch.int64)
        seq_len = torch.randint(1000, 4000, [batch_size_imm,]).to(torch.int32)
        while seq_len.square().sum() * head_num_imm > 2 ** 31:
            seq_len = torch.randint(1000, 4000, [batch_size_imm,]).to(torch.int32)
        y_input_list = []
        y_grad_list = []
        golden_list = []
        for i in range(seq_len.shape[0]):
            yi, yg, gd = gen_softmax_grad(head_num_imm, seq_len[i])
            y_input_list.append(yi.reshape(-1))
            y_grad_list.append(yg.reshape(-1))
            golden_list.append(gd.reshape(-1))
        y_input = torch.cat(y_input_list)
        y_grad = torch.cat(y_grad_list)
        self.golden = torch.cat(golden_list)
        x_grad = torch.zeros_like(y_input)

        in_tensors = [y_input, y_grad]
        out_tensors = [x_grad]

        OP_PARAM['headNum'] = head_num_imm
        OP_PARAM['qSeqLen'] = seq_len.tolist()
        self.set_param(OP_NAME, OP_PARAM)
        self.execute(in_tensors, out_tensors)


if __name__ == '__main__':
    unittest.main()
