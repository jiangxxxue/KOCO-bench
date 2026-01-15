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


MIX_UNPAD_FASTSOFTMAX = 1301
OP_NAME = "FastSoftMaxOperation"
OP_PARAM = {"qSeqLen": None, "headNum": 0}

class TestFastSoftMax(op_test.OpTest):
    def golden_calc(self, in_tensors):
        data_input = in_tensors[0].npu()
        seq_len_list = OP_PARAM['qSeqLen']
        head_num_imm = OP_PARAM['headNum']
        golden = torch.empty_like(data_input)

        start = 0
        for i in range(len(seq_len_list)):
            end = start + head_num_imm * seq_len_list[i] * seq_len_list[i]
            cur_data_input = data_input[start:end].reshape(-1, seq_len_list[i]).npu()
            cur_golden = torch.softmax(cur_data_input, dim=-1)
            golden[start:end] = cur_golden.reshape(-1).cpu()
            start = end

        return [golden.cpu()]

    def golden_compare(self, out_tensors, golden_out_tensors):
        return torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)

    @op_test.only_910b
    def test_fastsoftmax1(self):
        batch_size_imm = 32
        head_num_imm = 8
        head_num = torch.Tensor([head_num_imm,]).to(torch.int32)
        seq_len = torch.randint(1, 100, [batch_size_imm,]).to(torch.int32)
        data_input_list = [torch.randn(head_num_imm * l * l).to(torch.float16) for l in seq_len.tolist()]

        data_input = torch.cat(data_input_list)
        data_output = torch.zeros_like(data_input)

        in_tensors = [data_input]
        out_tensors = [data_output]

        OP_PARAM['qSeqLen'] = seq_len.tolist()
        OP_PARAM['headNum'] = head_num_imm
        self.set_param(OP_NAME, OP_PARAM)
        self.execute(in_tensors, out_tensors)

    @op_test.only_910b
    def test_fastsoftmax2(self):
        batch_size_imm = 32
        head_num_imm = 8
        head_num = torch.Tensor([head_num_imm,]).to(torch.int32)
        seq_len = torch.randint(100, 1000, [batch_size_imm,]).to(torch.int32)
        data_input_list = [torch.randn(head_num_imm * l * l).to(torch.float16) for l in seq_len.tolist()]

        data_input = torch.cat(data_input_list)
        data_output = torch.zeros_like(data_input)

        in_tensors = [data_input]
        out_tensors = [data_output]

        OP_PARAM['qSeqLen'] = seq_len.tolist()
        OP_PARAM['headNum'] = head_num_imm
        self.set_param(OP_NAME, OP_PARAM)
        self.execute(in_tensors, out_tensors)

    # @op_test.only_910b
    # def test_fastsoftmax3(self):
    #     batch_size_imm = 32
    #     head_num_imm = 8
    #     head_num = torch.Tensor([head_num_imm,]).to(torch.int32)
    #     seq_len = torch.randint(1000, 4000, [batch_size_imm,]).to(torch.int32)
    #     while seq_len.square().sum() * head_num_imm > 2 ** 31:
    #         seq_len = torch.randint(1000, 4000, [batch_size_imm,]).to(torch.int32)
    #     data_input_list = [torch.randn(head_num_imm * l * l).to(torch.float16) for l in seq_len.tolist()]

    #     data_input = torch.cat(data_input_list)
    #     data_output = torch.zeros_like(data_input)

    #     in_tensors = [data_input]
    #     out_tensors = [data_output]

    #     OP_PARAM['qSeqLen'] = seq_len.tolist()
    #     OP_PARAM['headNum'] = head_num_imm
    #     self.set_param(OP_NAME, OP_PARAM)
    #     self.execute(in_tensors, out_tensors)


if __name__ == '__main__':
    unittest.main()
