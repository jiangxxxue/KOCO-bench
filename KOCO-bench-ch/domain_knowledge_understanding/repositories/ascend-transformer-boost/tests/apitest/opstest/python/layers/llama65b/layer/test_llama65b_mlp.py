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
import math
import json
import torch
import torch_npu
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.nn.parameter import Parameter

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../operations"))
import operation_test

LAYER_NAME = "Llama65BMlpLayer"

PARAM = {}

intensor0 = torch.rand(28, 8, 128, dtype=torch.float16).npu().half() # in_hidden_states
intensor1 = torch.rand(4, 128, dtype=torch.float16).npu().half() # in_weight

outtensor0 = torch.rand(28, 8, 2, dtype=torch.float16).npu().half()

class TestLlama65BMlpLayer(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        # linear
        matA = in_tensors[0].reshape(224, 128)
        matB = in_tensors[1].transpose(0, 1)
        linear_out = torch.matmul(matA, matB) # (224, 4)
        # split
        linear_out = linear_out.reshape(1, 224, 4)
        intermidiate_matmul_gate_out, intermidiate_matmul_up_out = torch.chunk(linear_out, chunks=2, dim=2)
        # activation
        intermidiate_swish_out = intermidiate_matmul_gate_out.float() / (1 + torch.exp(-intermidiate_matmul_gate_out.float()*1.0))
        # elewise mul
        outtensor0 = intermidiate_swish_out * intermidiate_matmul_up_out
        outtensor0 = outtensor0.reshape(28, 8, 2)
        return [outtensor0.half()]

    def test(self):
        in_tensors = [intensor0, intensor1]
        self.execute(LAYER_NAME, PARAM, in_tensors)

if __name__ == '__main__':
    unittest.main()