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


OP_NAME = "TransposeOperation"


class TestTransposeOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        return [in_tensors[0].transpose(0, 1)]

    def test_2d_float(self):
        self.execute(OP_NAME, {"perm": [1, 0]}, [torch.randn(
            32, 128).npu().half()])
        
    def test_3d_float(self):
        self.execute(OP_NAME, {"perm": [1, 0, 2]}, [torch.randn(
            1, 4, 4096).npu().half()])
        
    def test_2d_int(self):
        self.execute(OP_NAME, {"perm": [1, 0]}, [torch.randint(1, 4,
            (32, 128)).npu()])


if __name__ == '__main__':
    unittest.main()
