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
import op_test

epsilon = 1e-6
OP_NAME = "NormOperation"
OP_PARAM = {"normType": 3, "epsilon": epsilon}

class TestRmsNormForward(op_test.OpTest):
    def golden_calc(self, in_tensors):
        x = in_tensors[0]
        gamma = in_tensors[1]
        result = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + epsilon)
        result = result * gamma
        return [result]

    def golden_compare(self, out_tensors, golden_out_tensors):
        return torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)
    
    @op_test.only_910b
    def test_rmsnorm(self):
        shape=[8, 8, 8]
        shape_gamma=[8]
        shape_rstd=[8, 8, 1]
        input0 = np.random.uniform(low=0, high=100, size=shape).astype(np.float32)
        input1 = np.random.uniform(low=0, high=100, size=shape_gamma).astype(np.float32)
        in_tensors = [torch.from_numpy(input0), torch.from_numpy(input1)]
        out_tensors = [torch.zeros(shape).float(), torch.zeros(shape_rstd).float()]

        self.set_param(OP_NAME, OP_PARAM)
        self.execute(in_tensors, out_tensors)

if __name__ == '__main__':
    unittest.main()
