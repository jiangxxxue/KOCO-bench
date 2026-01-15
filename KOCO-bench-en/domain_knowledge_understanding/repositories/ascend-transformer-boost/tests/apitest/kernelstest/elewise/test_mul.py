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
import op_test


OP_NAME = "ElewiseOperation"
OP_PARAM_MUL = {"elewiseType": 9}

class TestElewiseMul(op_test.OpTest):
    def golden_calc(self, in_tensors):
        actvation_type = self.op_desc["specificParam"]["elewiseType"]
        if actvation_type == 9:
            x = in_tensors[0].float()
            z = in_tensors[1].float()
            y = torch.mul(x, z)
            return [y]
        else:
            return [torch.zeros_like(x) for x in in_tensors]

    def golden_compare(self, out_tensors, golden_out_tensors):
        if (out_tensors[0].dtype == torch.bfloat16):
            return torch.allclose(out_tensors[0].bfloat16(), golden_out_tensors[0].bfloat16(), rtol=0.001, atol=0.001)
        return torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)
    
    @op_test.only_910b
    def test_mul_bf16(self):
        shape = (1000000)
        input0 = np.random.uniform(low=0, high=100, size=shape).astype(np.float32)
        input1 = np.random.uniform(low=0, high=100, size=shape).astype(np.float32)

        self.set_param(OP_NAME, OP_PARAM_MUL)
        self.execute([torch.from_numpy(input0).bfloat16(),torch.from_numpy(input1).bfloat16()],
                     [torch.zeros(shape).bfloat16()])

    def test_mul_fp16(self):
        shape = (1000000)
        input0 = np.random.uniform(low=0, high=100, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=0, high=100, size=shape).astype(np.float16)

        self.set_param(OP_NAME, OP_PARAM_MUL)
        self.execute([torch.from_numpy(input0).to(torch.float16), torch.from_numpy(input1).to(torch.float16)],
                     [torch.zeros(shape).to(torch.float16)])

if __name__ == '__main__':
    unittest.main()