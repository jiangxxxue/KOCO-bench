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

OP_NAME = "PadOperation"
PARAM = {"":1}

class TestPadOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        tmp_out = in_tensors[0]
        padding_offset = in_tensors[1]
        seq_len = in_tensors[2]
        input_ids = in_tensors[3]
        batch = input_ids.shape[0]
        hidden_dim = tmp_out.shape[1]
        total_length_imm = input_ids.shape[1]
 
        out = np.zeros((batch,hidden_dim)).astype(np.float16)
        tempVal = 0
        for i in range(batch):
            tempVal = tempVal + seq_len[i][0]
            out[i] = tmp_out[tempVal - 1].cpu()
        out = torch.from_numpy(out)
        return [out.npu()]
    
    def test_2d_half(self):
        if  operation_test.get_soc_version() == 'Ascend310B':
            print("this testcase don't supports Ascend310B")
            return True
        batch = 32
        total_length_imm = 64
        hidden_dim = 4096
 
        seq_len = np.random.randint(1, 64, size=batch) #几个真实数据
        input_ids = np.zeros((batch,total_length_imm))
        token_num = np.sum(seq_len)
        temp_arr = batch * [total_length_imm]
        zeros_num = np.array(temp_arr) - np.array(seq_len)
        cum_offsets_now = zeros_num
        cum_offsets_now = np.cumsum(zeros_num)
        tmp_out = np.random.uniform(-1,1,(token_num, hidden_dim)).astype(np.float16)
        for i in range(batch):
            input_ids[i][0: seq_len[i]] = np.random.randint(1, 50, size=seq_len[i])
        padding_offset = seq_len[0] * [0]
        for i in range(1, batch):
            temp_pad_out = seq_len[i] * [cum_offsets_now[i - 1]]
            padding_offset = np.concatenate((padding_offset, temp_pad_out))
        padding_offset = np.array(padding_offset).reshape(1, token_num)
        seq_len = seq_len.reshape(batch, 1)
        self.execute(OP_NAME, PARAM, [torch.from_numpy(tmp_out).npu().half(), torch.from_numpy(padding_offset).npu().int(), torch.from_numpy(seq_len).npu().int(), torch.from_numpy(input_ids).npu().long()])
    
        
if __name__ == '__main__':
    unittest.main()
