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



class TestRopeOperation2(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        seqlen = in_tensors[0].size()[1]
        batch = in_tensors[0].size()[0]
        q_head_num = in_tensors[0].size()[2]
        k_head_num = in_tensors[1].size()[2]
        rot_dim = in_tensors[2].size()[1]

        q = in_tensors[0]
        k = in_tensors[1]
        qshaped = q.reshape(batch, -1, q_head_num, rot_dim // 2, 2)
        kshaped = k.reshape(batch, -1, k_head_num, rot_dim // 2, 2)
        cos = in_tensors[2].view(-1, 2)[:, 0].view(batch, -1, 1, qshaped.size(3))
        sin = in_tensors[3].view(-1, 2)[:, 0].view(batch, -1, 1, qshaped.size(3))

        q_out2 = torch.stack(
            [
                qshaped[..., 0] * cos - qshaped[..., 1] * sin,
                qshaped[..., 1] * cos + qshaped[..., 0] * sin,
            ],
            -1,
        )

        q_out2 = q_out2.flatten(3)
        k_out2 = torch.stack(
            [
                kshaped[..., 0] * cos - kshaped[..., 1] * sin,
                kshaped[..., 1] * cos + kshaped[..., 0] * sin,
            ],
            -1,
        )
        k_out2 = k_out2.flatten(3)

        return [q_out2, k_out2]

    def test_2d_half(self):
        if operation_test.get_soc_version() == 'Ascend310B':
            logging.info("this testcase don't supports Ascend310B")
            return True
        ntoken = 512    
        seqlen = 256
        batch = 2
        q_head_num = 32
        k_head_num = 2
        head_size = 128
        # head_dim / 2时，前半部分旋转，后半部分不做
        intensor0 = torch.rand(batch, seqlen, q_head_num, head_size // 2).npu().half()
        intensor1 = torch.rand(batch, seqlen, k_head_num, head_size // 2).npu().half()
        # op需要cos/sin重复一次
        intensor2 = torch.rand(ntoken, head_size // 4, 1).repeat(1, 1, 2).view(ntoken, head_size // 2).npu().half()
        intensor3 = torch.rand(ntoken, head_size // 4, 1).repeat(1, 1, 2).view(ntoken, head_size // 2).npu().half()
        intensor4 = torch.tensor([seqlen, seqlen], dtype=torch.int32).npu()
        self.execute(OP_NAME, {"rotaryCoeff": 64},
                     [intensor0, intensor1, intensor2, intensor3, intensor4])

def rotate_half(x):
    x0, x1 = x.chunk(2, -1)
    return torch.cat((-x1, x0), dim=x0.ndim - 1)

class TestRopeOperation3(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        ntoken = in_tensors[0].size()[0]
        seqlen = int(in_tensors[4][0])
        batch = ntoken // seqlen
        hidden_size = in_tensors[0].size()[1]
        hidden_size1 = in_tensors[1].size()[1]
        head_size = in_tensors[2].size()[1]
        head_num = hidden_size // head_size
        head_num1 = hidden_size1 // head_size
        q = in_tensors[0].view(batch, seqlen, head_num, head_size).float()
        k = in_tensors[1].view(batch, seqlen, head_num1, head_size).float()
        cos = in_tensors[2].view(batch, seqlen, head_size).unsqueeze(2)
        sin = in_tensors[3].view(batch, seqlen, head_size).unsqueeze(2)
        q_embed = ((q * cos) + (rotate_half(q) * sin)).view(ntoken, hidden_size)
        k_embed = ((k * cos) + (rotate_half(k) * sin)).view(ntoken, hidden_size1)
        return [q_embed.half(), k_embed.half()]

    def test(self):
        return
        ntoken = 128
        seqlen = 128
        hidden_sizeq = 1024
        hidden_sizek = 128
        head_size = 128
        intensor0 = torch.rand(ntoken, hidden_sizeq).npu().half()
        intensor1 = torch.rand(ntoken, hidden_sizek).npu().half()
        intensor2 = torch.rand(ntoken, head_size).npu()
        intensor3 = torch.rand(ntoken, head_size).npu()
        intensor4 = torch.tensor([seqlen], dtype=torch.int32).npu()
        # llama highPrecision精度验证
        self.execute(OP_NAME, {"rotaryCoeff": 2, "cosFormat": 0},
                     [intensor0, intensor1, intensor2, intensor3, intensor4])

class TestRopeOperation4(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        ntoken = in_tensors[0].size()[0]
        seqlen = int(in_tensors[4][0])
        batch = ntoken // seqlen
        hidden_size = in_tensors[0].size()[1]
        hidden_size1 = in_tensors[1].size()[1]
        head_size = in_tensors[2].size()[1]
        head_num = hidden_size // head_size
        head_num1 = hidden_size1 // head_size
        q = in_tensors[0].view(batch, seqlen, head_num, head_size)
        k = in_tensors[1].view(batch, seqlen, head_num1, head_size)
        cos = in_tensors[2].view(batch, seqlen, head_size).unsqueeze(2)
        sin = in_tensors[3].view(batch, seqlen, head_size).unsqueeze(2)
        q_embed = ((q * cos) + (rotate_half(q) * sin)).view(ntoken, hidden_size)
        k_embed = ((k * cos) + (rotate_half(k) * sin)).view(ntoken, hidden_size1)
        return [q_embed, k_embed]

    def test(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        ntoken = 128
        seqlen = 128
        hidden_sizeq = 1024
        hidden_sizek = 128
        head_size = 128
        intensor0 = torch.rand(ntoken, hidden_sizeq).npu().bfloat16()
        intensor1 = torch.rand(ntoken, hidden_sizek).npu().bfloat16()
        intensor2 = torch.rand(ntoken, head_size).npu().bfloat16()
        intensor3 = torch.rand(ntoken, head_size).npu().bfloat16()
        intensor4 = torch.tensor([seqlen], dtype=torch.int32).npu()
        self.execute(OP_NAME, {"rotaryCoeff": 2}, [intensor0, intensor1, intensor2, intensor3, intensor4])

if __name__ == '__main__':
    unittest.main()
