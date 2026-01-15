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
 
 
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402
 
OP_NAME = "ElewiseOperation"
 
class TestElewiseCastOperation1(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        return [in_tensors[0].type(torch.bfloat16)]
 
    def golden_compare(self, out_tensor, golden_out_tensor):
        print(f"out_tensor is:{out_tensor}")
        print(f"golden_out_tensor is:{golden_out_tensor}")
        # return True
        return torch.allclose(out_tensor.cpu(), golden_out_tensor.cpu(), rtol=0.001, atol=0.001)
 
    def test_cast_float_to_bfloat16(self):
        if operation_test.get_soc_version() != 'Ascend910B':
            print("this testcase don't supports Atlas inference products")
            return True
        input0 = torch.randint(1, 100, (10,), dtype=torch.float)
        print(f"input0:{input0}")
        input0 = input0.npu()
        self.execute(OP_NAME, {"elewiseType": 1, "outTensorType": 27}, [input0])

class TestElewiseCastOperation2(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        return [in_tensors[0].type(torch.float)]
 
    def golden_compare(self, out_tensor, golden_out_tensor):
        print(f"out_tensor is:{out_tensor}")
        print(f"golden_out_tensor is:{golden_out_tensor}")
        # return True
        return torch.allclose(out_tensor.cpu(), golden_out_tensor.cpu(), rtol=0.001, atol=0.001)
 
    def test_cast_bfloat16_to_float(self):
        if operation_test.get_soc_version() != 'Ascend910B':
            print("this testcase don't supports Atlas inference products")
            return True
        input0 = torch.randint(1, 100, (10,), dtype=torch.bfloat16)
        print(f"input0:{input0}")
        input0 = input0.npu()
        self.execute(OP_NAME, {"elewiseType": 1, "outTensorType": 0}, [input0])
 
if __name__ == '__main__':
    unittest.main()