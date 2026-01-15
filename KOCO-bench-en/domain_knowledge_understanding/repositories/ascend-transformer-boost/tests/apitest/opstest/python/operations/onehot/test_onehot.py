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
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test

OP_NAME = "OnehotOperation"
PARAM = {"axis": -1, "depth": 10}

inputs = []
inputs1 = []

def generate_data():
    shape0 = (16,)
    shape1 = (1,)
    input0 = np.random.randint(PARAM["depth"], size=shape0).astype(np.int64)
    input1 = np.ones(shape=shape1, dtype=np.int64)
    input2 = np.zeros(shape=shape1, dtype=np.int64)

    return [input0, input1, input2]

data = generate_data()
for i in range(3):
    inputs.append(torch.from_numpy(data[i]).long().npu())
    inputs1.append(torch.from_numpy(data[i]).int().npu())

class TestSoftmaxOperationInt64(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        in_tensors[0] = in_tensors[0].cpu()
        data_type = in_tensors[0].dtype
        input0 = in_tensors[0].numpy()
        res = np.eye(PARAM["depth"])[input0]
        res = torch.from_numpy(res).to(data_type).npu()
        return [res]

    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.allclose(out_tensor, golden_out_tensor, rtol=0.001, atol=0.001)

    def test(self):
        if  operation_test.get_soc_version() == 'Ascend310B':
            print("this testcase don't supports Ascend310B")
            return True
        self.execute(OP_NAME, PARAM, inputs)

class TestSoftmaxOperationInt32(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        in_tensors[0] = in_tensors[0].cpu()
        data_type = in_tensors[0].dtype
        input0 = in_tensors[0].numpy()
        res = np.eye(PARAM["depth"])[input0]
        res = torch.from_numpy(res).to(data_type).npu()
        return [res]

    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.allclose(out_tensor, golden_out_tensor, rtol=0.001, atol=0.001)

    def test(self):
        if  operation_test.get_soc_version() == 'Ascend310B':
            print("this testcase don't supports Ascend310B")
            return True
        self.execute(OP_NAME, PARAM, inputs1)

if __name__ == '__main__':
    unittest.main()
