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

OP_NAME = "UnpadWithHiddenStateOperation"
OP_PARAM = {"qSeqLen": None, "maxSeqLen": 0}

class TestUnpadWithHiddenStateOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        data_input = in_tensors[0]
        seq_len_list = OP_PARAM['qSeqLen']
        max_seq_len_imm = OP_PARAM['maxSeqLen']
        hidden_size_imm = data_input.shape[-1]

        golden = torch.empty(size=[sum(seq_len_list), hidden_size_imm], dtype=torch.float16)
        start = 0
        for i in range(len(seq_len_list)):
            golden[start:start + seq_len_list[i]] = data_input[i][:seq_len_list[i]]
            start = start + seq_len_list[i]

        return [golden]

    def get_test_data(self, batch_size_imm, max_seq_len_imm, hidden_size_imm):
        seq_len = torch.randint(low=100, high=300, size=[batch_size_imm,], dtype=torch.int32)
        data_input = torch.randn(size=[batch_size_imm, max_seq_len_imm, hidden_size_imm], dtype=torch.float16)
        OP_PARAM['qSeqLen'] = seq_len.tolist()
        OP_PARAM['maxSeqLen'] = max_seq_len_imm
        return [data_input.npu()]

    def test_unpad_with_hidden_state(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        batch_size_imm = 4
        max_seq_len_imm = 4096
        hidden_size_imm = 4096
        input_tensor = self.get_test_data(batch_size_imm, max_seq_len_imm, hidden_size_imm)
        self.execute(OP_NAME, OP_PARAM, input_tensor)
        batch_size_imm = 8
        input_tensor = self.get_test_data(batch_size_imm, max_seq_len_imm, hidden_size_imm)
        self.execute_update_param(OP_NAME, OP_PARAM, input_tensor)

if __name__ == '__main__':
    unittest.main()