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
import operation_test
epsilon = 1e-5
OP_NAME = "RmsNormOperation"

class TestRmsNormOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        x = in_tensors[0]
        gamma = in_tensors[1]
        reduceDims=[]
        edim = x.dim()-gamma.dim()
        for i in range(gamma.dim()):
            reduceDims.append(edim + i)
        rstd = torch.rsqrt(x.pow(2).mean(reduceDims, keepdim=True) + epsilon)
        result = x * rstd
        result = result * gamma
        return [result, rstd]

    def test(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        shape=[8, 8, 8]
        shape_gamma=[8]
        shape_rstd=[8, 8, 1]
        input0 = np.random.uniform(low=0, high=100, size=shape).astype(np.float32)
        input1 = np.random.uniform(low=0, high=100, size=shape_gamma).astype(np.float32)
        in_tensors = [torch.from_numpy(input0).npu(), torch.from_numpy(input1).npu()]
        self.execute(OP_NAME, {"layerType": 1, "normParam":{"rstd": True}},
                     in_tensors)

    def test2(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        shape=[16, 16, 16]
        shape_gamma=[16, 16]
        shape_rstd=[16, 1, 1]
        input0 = np.random.uniform(low=0, high=100, size=shape).astype(np.float32)
        input1 = np.random.uniform(low=0, high=100, size=shape_gamma).astype(np.float32)
        in_tensors = [torch.from_numpy(input0).npu(), torch.from_numpy(input1).npu()]
        self.execute(OP_NAME, {"layerType": 1, "normParam":{"rstd": True}},
                     in_tensors)

if __name__ == '__main__':
    unittest.main()
