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


sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402


OP_NAME = "RopeOperation"


def rotate_half(x):
    x0, x1 = x.chunk(2, -1)
    return torch.cat((-x1, x0), dim=x0.ndim - 1)

class TestUnpadRopeOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        ntoken = in_tensors[0].size()[0]
        seqlen = in_tensors[4].tolist()
        batch = in_tensors[4].shape[0]
        hidden_size = in_tensors[0].size()[1]
        head_size = in_tensors[2].size()[1]
        head_num = hidden_size // head_size
        q_list = []
        k_list = []
        offset = 0
        for i, _ in enumerate(range(batch)):
            cur_seqlen = seqlen[i]
            next_offset = offset + cur_seqlen
            qlayer = in_tensors[0][offset:next_offset].view(cur_seqlen, head_num, head_size)
            klayer = in_tensors[1][offset:next_offset].view(cur_seqlen, head_num, head_size)
            cos = in_tensors[2][:cur_seqlen].unsqueeze(1)
            sin = in_tensors[3][:cur_seqlen].unsqueeze(1)
            qlayer = (qlayer * cos) + (rotate_half(qlayer) * sin)
            klayer = (klayer * cos) + (rotate_half(klayer) * sin)
            q = qlayer.view(cur_seqlen, hidden_size)
            q_list.append(q)
            k = klayer.view(cur_seqlen, hidden_size)
            k_list.append(k)
            offset = next_offset
        q_sum = torch.cat(tuple(q_list), dim=0)
        k_sum = torch.cat(tuple(k_list), dim=0)
        return [q_sum, k_sum]

    def test(self):
        min_seqlen = 1
        max_seqlen = 2048
        batch = 4
        seqlen = torch.randint(min_seqlen, max_seqlen, (batch,), dtype=torch.int32)
        ntoken = int(seqlen.sum())
        hidden_size = 1024
        head_size = 128
        intensor0 = torch.rand(ntoken, hidden_size).npu().half()
        intensor1 = torch.rand(ntoken, hidden_size).npu().half()
        intensor2 = torch.rand(max_seqlen, head_size).npu().half()
        intensor3 = torch.rand(max_seqlen, head_size).npu().half()
        intensor4 = seqlen.npu()
        self.execute(OP_NAME, {"rotaryCoeff": 2,"cosFormat":1},
                    [intensor0, intensor1, intensor2, intensor3, intensor4])


if __name__ == '__main__':
    unittest.main()
