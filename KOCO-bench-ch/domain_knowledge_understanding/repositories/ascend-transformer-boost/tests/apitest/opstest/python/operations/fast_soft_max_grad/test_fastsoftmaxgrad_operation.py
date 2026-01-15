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
import torch
import torch_npu
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test

OP_NAME = "FastSoftMaxGradOperation"
OP_PARAM = {"headNum": 0, "qSeqLen": None}

def gen_softmax_grad(head_num, seq_len):
    x = torch.randn([head_num * seq_len, seq_len]).to(torch.float32)
    x.requires_grad = True
    y = torch.softmax(x.to(torch.float32), dim=-1).to(torch.float32)
    y.retain_grad()
    w = torch.randn_like(x).to(torch.float32)
    loss = (w * y).sum()
    loss.backward()
    return (y.detach().to(torch.float16), y.grad.detach().to(torch.float16), x.grad.detach().to(torch.float16))

class TestFastSoftMaxGradOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        return [self.golden]

    def get_test_data(self, batch_size_imm, head_num_imm):
        head_num = torch.Tensor([head_num_imm,]).to(torch.int64)
        seq_len = torch.randint(100, 300, [batch_size_imm,]).to(torch.int32)
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
        OP_PARAM['qSeqLen'] = seq_len.tolist()
        OP_PARAM['headNum'] = head_num_imm
        return [y_input.npu(), y_grad.npu()]

    def test_fastsoftmaxgrad(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        batch_size_imm = 4
        head_num_imm = 8
        in_tensors = self.get_test_data(batch_size_imm, head_num_imm)
        self.execute(OP_NAME, OP_PARAM, in_tensors)
        batch_size_imm = 6
        in_tensors = self.get_test_data(batch_size_imm, head_num_imm)
        self.execute_update_param(OP_NAME, OP_PARAM, in_tensors)

if __name__ == '__main__':
    unittest.main()