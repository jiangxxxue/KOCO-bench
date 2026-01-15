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

OP_NAME = "PadWithHiddenStateOperation"
OP_PARAM = {"qSeqLen": None, "maxSeqLen": 0}

class TestPadWithHiddenState(op_test.OpTest):
    def golden_calc(self, in_tensors):
        data_input = in_tensors[0]
        seq_len_list = OP_PARAM['qSeqLen']
        max_seq_len_imm = OP_PARAM['maxSeqLen']
        hidden_size_imm = data_input.shape[-1]

        golden = torch.zeros(size=[len(seq_len_list), max_seq_len_imm, hidden_size_imm], dtype=torch.float16)
        start = 0
        for i in range(len(seq_len_list)):
            golden[i][:seq_len_list[i]] = data_input[start:start + seq_len_list[i]]
            golden[i][seq_len_list[i]:] = 0
            start = start + seq_len_list[i]

        return [golden]

    def golden_compare(self, out_tensors, golden_out_tensors):
        return torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)

    @op_test.only_910b
    def test_pad_with_hidden_state1(self):
        batch_size_imm = 4
        max_seq_len_imm = 4096
        hidden_size_imm = 4096

        seq_len = torch.randint(low=100, high=300, size=[batch_size_imm,], dtype=torch.int32)
        data_input = torch.randn(size=[seq_len.sum(), hidden_size_imm], dtype=torch.float16)
        data_output = torch.zeros(size=[batch_size_imm, max_seq_len_imm, hidden_size_imm], dtype=torch.float16)

        OP_PARAM['qSeqLen'] = seq_len.tolist()
        OP_PARAM['maxSeqLen'] = max_seq_len_imm

        self.set_param(OP_NAME, OP_PARAM)
        self.execute([data_input], [data_output])

    @op_test.only_910b
    def test_pad_with_hidden_state2(self):
        batch_size_imm = 32
        max_seq_len_imm = 4096
        hidden_size_imm = 4096

        seq_len = torch.randint(low=100, high=300, size=[batch_size_imm,], dtype=torch.int32)
        data_input = torch.randn(size=[seq_len.sum(), hidden_size_imm], dtype=torch.float16)
        data_output = torch.zeros(size=[batch_size_imm, max_seq_len_imm, hidden_size_imm], dtype=torch.float16)

        OP_PARAM['qSeqLen'] = seq_len.tolist()
        OP_PARAM['maxSeqLen'] = max_seq_len_imm

        self.set_param(OP_NAME, OP_PARAM)
        self.execute([data_input], [data_output])


if __name__ == '__main__':
    unittest.main()
