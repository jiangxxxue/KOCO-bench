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
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

OP_NAME = "UnpadOperation"
PARAM = {"":1}

class TestUnpadOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        batch = in_tensors[0].shape[0]
        total_length_imm = in_tensors[0].shape[1]
        seq_len = in_tensors[3].cpu()
        token_num = in_tensors[2].cpu()
        cum_offsets_now = in_tensors[1].cpu()
        input_ids = in_tensors[0].cpu()
 
        x_remove_padding = input_ids[0][0 : seq_len[0]].cpu()
        for i in range(1, batch):
            x_remove_padding = np.concatenate((x_remove_padding, input_ids[i][0:seq_len[i]]))
        x_remove_padding = np.pad(x_remove_padding, (0, (batch * total_length_imm - token_num[0][0])))
        cum_offsets_out = np.zeros(batch)
        for i in range(1, batch):
            cum_offsets_out[i] = cum_offsets_now[i - 1]
        padding_offset =seq_len[0] * [0]
        for i in range(1, batch):
            temp_pad_out =seq_len[i] * [cum_offsets_now[i - 1][0]]
            padding_offset = np.concatenate((padding_offset, temp_pad_out))
        zero_offset = np.zeros((1,batch * total_length_imm - token_num[0][0]))
        padding_offset = np.append(padding_offset, zero_offset)
        x_remove_padding = torch.from_numpy(x_remove_padding.reshape(1, batch * total_length_imm)).long()
        cum_offsets_out = torch.from_numpy(cum_offsets_out.reshape(batch, 1)).int()
        padding_offset = torch.from_numpy(padding_offset.reshape(1, batch * total_length_imm).astype(np.int32))
        return [x_remove_padding.npu(), cum_offsets_out.npu(), padding_offset.npu()]
 
    def test_2d_half(self):
        if  operation_test.get_soc_version() == 'Ascend310B':
            print("this testcase don't supports Ascend310B")
            return True
        batch = 32
        total_length_imm = 64
        shape = (batch, total_length_imm)
        input_ids = np.zeros(shape)
        seq_len = np.random.randint(1, 65, size=shape[0])
        for i in range(batch):
            input_ids[i][0: seq_len[i]] = np.random.randint(1, 50, size=seq_len[i])
 
        temp_arr = batch * [total_length_imm]
        zeros_num = np.array(temp_arr) - np.array(seq_len)
        cum_offsets_now = zeros_num
        cum_offsets_now = np.cumsum(zeros_num)        
        token_num = np.sum(seq_len)
 
        x_remove_padding_length = batch * total_length_imm
        cum_offsets_out_length = batch
 
        input_ids = np.array(input_ids).astype(np.int32)
        cum_offsets_now = np.array(cum_offsets_now).reshape(batch, 1).astype(np.int32)
        token_num = np.array(token_num).reshape(1, 1).astype(np.int64)
        seq_len = np.array(seq_len).reshape(batch, 1).astype(np.int32)
 
        self.execute(OP_NAME, PARAM,[torch.from_numpy(input_ids).npu().long(), torch.from_numpy(cum_offsets_now).npu(),\
                     torch.from_numpy(token_num).npu(), torch.from_numpy(seq_len).npu()])
    
        
if __name__ == '__main__':
    unittest.main()