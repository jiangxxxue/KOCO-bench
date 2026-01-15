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
import math
import op_test
import torch.nn as nn
OP_NAME = "ActivationOperation"
OP_PARAM0 = {"activationType": 2, "approx": 0} # apporx: tanh
OP_PARAM1 = {"activationType": 2, "approx": 1} # apporx: none

class TestActivation(op_test.OpTest):
    def gelu_approx(self):
        def forward(in_tensor):
            return 0.5 * in_tensor * (1 + F.tanh(math.sqrt(2 / math.pi) * (in_tensor + 0.044715 * torch.pow(in_tensor, 3))))
        return forward

    def golden_calc(self, in_tensors):
        actvation_type = self.op_desc["specificParam"]["activationType"]
        if actvation_type == 2:
            approx = self.op_desc["specificParam"]["approx"]
            m = nn.GELU() if approx != 0 else self.gelu_approx()
            x = in_tensors[0].float()
            y = m(x).to(in_tensors[0].dtype)
            return [y]
        else:
            return [torch.zeros_like(x) for x in in_tensors]

    def golden_compare(self, out_tensors, golden_out_tensors):
        return torch.allclose(out_tensors[0], golden_out_tensors[0], atol=0.001, rtol=0.001)

    @op_test.only_910b
    def test_gelu_bf16(self):
        shape = (100000)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float32)
        self.set_param(OP_NAME, OP_PARAM1)
        self.execute([torch.from_numpy(input0).bfloat16()],
                     [torch.zeros(shape).bfloat16()])

    @op_test.skip_310b
    def test_gelu_approx_f32(self):
        shape = (100000)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float32)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.from_numpy(input0)],
                     [torch.zeros(shape)])

    @op_test.skip_310b
    def test_gelu_approx_f16(self):
        shape = (100000)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float32)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.from_numpy(input0).half()],
                     [torch.zeros(shape).half()])
    
    @op_test.only_910b
    def test_gelu_approx_bf16(self):
        shape = (100000)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float32)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.from_numpy(input0).bfloat16()],
                     [torch.zeros(shape).bfloat16()])

if __name__ == '__main__':
    unittest.main()
