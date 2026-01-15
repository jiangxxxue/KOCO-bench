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
import logging


sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402


OP_NAME = "RopeOperation"


def rotate_half(x):
    x0, x1 = x.chunk(2, -1)
    return torch.cat((-x1, x0), dim=x0.ndim - 1)


class TestRopeOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        ntoken = in_tensors[0].size()[0]
        seqlen = int(in_tensors[4][0])
        batch = ntoken // seqlen
        hidden_size = in_tensors[0].size()[1]
        head_size = in_tensors[2].size()[1]
        head_num = hidden_size // head_size
        qlayer = in_tensors[0].view(seqlen, batch, head_num, head_size)
        q0, q1 = qlayer.chunk(2, -1)
        klayer = in_tensors[1].view(seqlen, batch, head_num, head_size)
        k0, k1 = klayer.chunk(2, -1)
        cos0, cos1 = in_tensors[2].unsqueeze(1).unsqueeze(1).chunk(2, -1)
        sin0, sin1 = in_tensors[3].unsqueeze(1).unsqueeze(1).chunk(2, -1)
        q0 = (q0 * cos0) + (rotate_half(q0) * sin0)
        k0 = (k0 * cos0) + (rotate_half(k0) * sin0)
        q1 = (q1 * cos1) + (rotate_half(q1) * sin1)
        k1 = (k1 * cos1) + (rotate_half(k1) * sin1)
        q = torch.concat([q0, q1], dim=(q0.ndim - 1)).view(ntoken, hidden_size)
        k = torch.concat([k0, k1], dim=(k0.ndim - 1)).view(ntoken, hidden_size)
        return [q, k]

    def test(self):
        if operation_test.get_soc_version() == 'Ascend310B':
            logging.info("this testcase don't supports Ascend310B")
            return True
        ntoken = 4
        seqlen = 4
        hidden_size = 4096
        head_size = 128
        intensor0 = torch.rand(ntoken, hidden_size).npu().half()
        intensor1 = torch.rand(ntoken, hidden_size).npu().half()
        intensor2 = torch.rand(ntoken, head_size).npu().half()
        intensor3 = torch.rand(ntoken, head_size).npu().half()
        intensor4 = torch.tensor([seqlen], dtype=torch.int32).npu()
        self.execute(OP_NAME, {"rotaryCoeff": 4},
                     [intensor0, intensor1, intensor2, intensor3, intensor4])


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
            q0, q1 = qlayer.chunk(2, -1)
            klayer = in_tensors[1][offset:next_offset].view(cur_seqlen, head_num, head_size)
            k0, k1 = klayer.chunk(2, -1)
            cos0, cos1 = in_tensors[2][offset:next_offset].unsqueeze(1).chunk(2, -1)
            sin0, sin1 = in_tensors[3][offset:next_offset].unsqueeze(1).chunk(2, -1)
            q0 = (q0 * cos0) + (rotate_half(q0) * sin0)
            k0 = (k0 * cos0) + (rotate_half(k0) * sin0)
            q1 = (q1 * cos1) + (rotate_half(q1) * sin1)
            k1 = (k1 * cos1) + (rotate_half(k1) * sin1)
            q = torch.concat([q0, q1], dim=(q0.ndim - 1)).view(cur_seqlen, hidden_size)
            q_list.append(q)
            k = torch.concat([k0, k1], dim=(k0.ndim - 1)).view(cur_seqlen, hidden_size)
            k_list.append(k)
            offset = next_offset
        q_sum = torch.cat(tuple(q_list), dim=0)
        k_sum = torch.cat(tuple(k_list), dim=0)
        return [q_sum, k_sum]

    def test(self):
        if operation_test.get_soc_version() == 'Ascend310B':
            logging.info("this testcase don't supports Ascend310B")
            return True
        min_seqlen = 1
        max_seqlen = 5
        batch = 3
        seqlen = torch.randint(min_seqlen, max_seqlen, (batch,), dtype=torch.int32)
        ntoken = int(seqlen.sum())
        hidden_size = 4096
        head_size = 128
        intensor0 = torch.rand(ntoken, hidden_size).npu().half()
        intensor1 = torch.rand(ntoken, hidden_size).npu().half()
        intensor2 = torch.rand(ntoken, head_size).npu().half()
        intensor3 = torch.rand(ntoken, head_size).npu().half()
        intensor4 = seqlen.npu()
        self.execute(OP_NAME, {"rotaryCoeff": 4},
                     [intensor0, intensor1, intensor2, intensor3, intensor4])


if __name__ == '__main__':
    unittest.main()
