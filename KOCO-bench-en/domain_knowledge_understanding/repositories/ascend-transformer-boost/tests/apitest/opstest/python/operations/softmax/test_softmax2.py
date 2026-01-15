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
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test

OP_NAME = "SoftmaxOperation"
PARAM = {"axes": [-1]}


class TestSoftmaxOperation2(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        softmaxLast = nn.Softmax(dim=-1)
        return [softmaxLast(in_tensors[0])]

    def test_2d_float(self):
        self.execute(OP_NAME, PARAM, [torch.randn([2, 16, 256, 256]).npu().half()])

if __name__ == '__main__':
    unittest.main()