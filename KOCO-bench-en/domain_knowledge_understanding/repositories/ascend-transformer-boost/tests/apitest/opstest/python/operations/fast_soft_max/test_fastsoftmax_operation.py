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

OP_NAME = "FastSoftMaxOperation"
OP_PARAM = {"headNum": 0, "qSeqLen": None}

class TestFastSoftMaxOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        data_input = in_tensors[0]
        seq_len_list = OP_PARAM['qSeqLen']
        head_num_imm = OP_PARAM['headNum']
        golden = torch.empty_like(data_input)

        start = 0
        for i in range(len(seq_len_list)):
            end = start + head_num_imm * seq_len_list[i] * seq_len_list[i]
            cur_data_input = data_input[start:end].reshape(-1, seq_len_list[i])
            cur_golden = torch.softmax(cur_data_input.to(torch.float32), dim=-1).to(torch.float16)
            golden[start:end] = cur_golden.reshape(-1)
            start = end
        return [golden]

    def get_test_data(self, batch_size_imm, head_num_imm):
        # 设置固定的 batch_size_imm 和 head_num_imm
        head_num = torch.tensor([head_num_imm]).to(torch.int64)
        seq_len = torch.randint(100, 300, [batch_size_imm]).to(torch.int32)

        # 生成 data_input_list
        data_input_list = [torch.randn(head_num_imm * seq_len[i] * seq_len[i]).to(torch.float16) for i in range(batch_size_imm)]
        data_input = torch.cat(data_input_list)
        in_tensors = [data_input.npu()]  # 假设这里的 npu() 是你的自定义操作

        # 构建 OP_PARAM
        OP_PARAM['qSeqLen'] = seq_len.tolist()
        OP_PARAM['headNum'] = head_num_imm
        print("qSeqLen", seq_len.tolist())
        return in_tensors

    def test_fastsoftmax(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        batch_size_imm = 4
        head_num_imm = 8

        in_tensors = self.get_test_data(batch_size_imm, head_num_imm)
        self.execute(OP_NAME, OP_PARAM, in_tensors)

        print("-----update_param")
        batch_size_imm = 6
        in_tensors = self.get_test_data(batch_size_imm, head_num_imm)
        self.execute_update_param(OP_NAME, OP_PARAM, in_tensors)

if __name__ == '__main__':
    unittest.main()