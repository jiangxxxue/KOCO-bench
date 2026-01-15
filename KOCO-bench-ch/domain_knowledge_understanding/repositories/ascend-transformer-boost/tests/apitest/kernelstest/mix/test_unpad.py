# 
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# 
import os
import unittest
import numpy as np
import torch
import sys
import logging

import op_test
from tensor_file import read_tensor


OP_NAME = "UnpadOperation"
OP_PARAM0 = {}

batch = 4
total_length_imm = 9
shape = (batch, total_length_imm)
input_ids = np.zeros(shape)
seq_len = np.random.randint(total_length_imm / 2 , total_length_imm, size=shape[0])
seq_len[np.random.randint(1, batch, size= 1)] = 0

logging.debug(f"seq_len is {seq_len}")
for i in range(batch):
    if seq_len[i] == 0:
        input_ids[i][0: (total_length_imm - 4)] = np.random.randint(1, 2, size= total_length_imm - 4)
    else :
        input_ids[i][0: seq_len[i]] = np.random.randint(1, 30, size=seq_len[i])

temp_arr = batch * [total_length_imm]
zeros_num = np.array(temp_arr) - np.array(seq_len)
cum_offsets_now = zeros_num
cum_offsets_now = np.cumsum(zeros_num)        
token_num = np.sum(seq_len)

class TestUnpad(op_test.OpTest):
    def golden_calc(self, in_tensors):

        batch = in_tensors[0].shape[0]
        total_length_imm = in_tensors[0].shape[1]
        seq_len = in_tensors[3]
        token_num = in_tensors[2]
        cum_offsets_now = in_tensors[1]
        input_ids = in_tensors[0]

        x_remove_padding = input_ids[0][0 : seq_len[0]]
        for i in range(1, batch):
            if seq_len[i] == 0:
                continue
            else:
                x_remove_padding = np.concatenate((x_remove_padding, input_ids[i][0:seq_len[i]]))

        x_remove_padding = np.pad(x_remove_padding, (0, batch * total_length_imm - token_num[0][0]))
        cum_offsets_out = np.zeros(batch)
        for i in range(1, batch):
            cum_offsets_out[i] = cum_offsets_now[i - 1]
        padding_offset =seq_len[0] * [0]
        for i in range(1, batch):
            if seq_len[i] == 0:
                continue
            else:
                temp_pad_out = np.array(seq_len[i] * [cum_offsets_now[i - 1]]).reshape(-1)
                padding_offset = np.concatenate((padding_offset, temp_pad_out))
        padding_offset = np.pad(padding_offset, (0, batch * total_length_imm - token_num[0][0]))
        x_remove_padding = torch.from_numpy(x_remove_padding.reshape(1, batch * total_length_imm)).long()
        cum_offsets_out = torch.from_numpy(cum_offsets_out.reshape(batch, 1)).int()
        padding_offset = torch.from_numpy(padding_offset.reshape(1, batch * total_length_imm).astype(np.int32))
        return [x_remove_padding, cum_offsets_out, padding_offset]

    def golden_compare(self, out_tensors, golden_out_tensors):
        for i in range(token_num):
            if out_tensors[0][0][i] != golden_out_tensors[0][0][i]:
                return False
            if out_tensors[2][0][i] != golden_out_tensors[2][0][i]:
                return False

        sec_res = torch.allclose(out_tensors[1], golden_out_tensors[1], rtol=0.001, atol=0.001)
        return sec_res

    @op_test.skip_310b
    @op_test.skip_910a
    def test_2d_half(self):
        x_remove_padding_length = batch * total_length_imm
        cum_offsets_out_length = batch

        input_ids1 = np.array(input_ids).astype(np.int64)
        cum_offsets_now1 = np.array(cum_offsets_now).reshape(batch, 1).astype(np.int32)
        token_num1 = np.array(token_num).reshape(1, 1).astype(np.int64)
        seq_len1 = np.array(seq_len).reshape(batch, 1).astype(np.int32)

        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.from_numpy(input_ids1), torch.from_numpy(cum_offsets_now1),\
                     torch.from_numpy(token_num1), torch.from_numpy(seq_len1)],\
                     [torch.zeros((1, x_remove_padding_length)).long(), torch.zeros((cum_offsets_out_length, 1)).int(),\
                     torch.zeros((1, x_remove_padding_length)).int()])

if __name__ == '__main__':
    unittest.main()
