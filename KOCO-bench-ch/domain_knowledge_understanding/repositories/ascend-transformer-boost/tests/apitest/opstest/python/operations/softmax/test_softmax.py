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
PARAM = {"axes": [1, 2]}


class TestSoftmaxOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        in_tensor_dim_num = in_tensors[0].dim()
        PARAM["axes"] = [ i % in_tensor_dim_num for i in PARAM["axes"] ]
        target_shape = in_tensors[0].shape[PARAM["axes"][0]:PARAM["axes"][-1] + 1]
        in_tensor_flatten = in_tensors[0].flatten(start_dim=PARAM["axes"][0], end_dim=PARAM["axes"][-1])
        softmax0 = nn.Softmax(dim=PARAM["axes"][0])
        out_tensor = softmax0(in_tensor_flatten)
        out_tensor = out_tensor.unflatten(PARAM["axes"][0], target_shape)
        return [out_tensor]

    def test_2d_float(self):
        self.execute(OP_NAME, PARAM, [torch.randn(size=[2, 4, 4]).npu().half()])

if __name__ == '__main__':
    unittest.main()
