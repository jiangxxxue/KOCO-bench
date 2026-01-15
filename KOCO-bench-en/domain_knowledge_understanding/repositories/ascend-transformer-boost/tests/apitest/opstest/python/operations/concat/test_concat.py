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

OP_NAME = "ConcatOperation"


class TestConcatOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        golden_result = torch.cat((in_tensors[0], in_tensors[1]), dim=1)
        return [golden_result.npu()]

    def test_float16(self):
        PARAM = {"concatDim":1}
        self.execute(OP_NAME, PARAM, [torch.randn(
            28, 32, 4096, dtype=torch.float16).npu(), torch.randn(28, 64, 4096, dtype=torch.float16).npu()])

    def test_bf16(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        PARAM = {"concatDim":1}
        self.execute(OP_NAME, PARAM, [torch.randn(
            21, 32, 4096, dtype=torch.bfloat16).npu(), torch.randn(21, 64, 4096, dtype=torch.bfloat16).npu()])

    def test_float16n(self):
        PARAM = {"concatDim":-2}
        self.execute(OP_NAME, PARAM, [torch.randn(
            28, 32, 4096, dtype=torch.float16).npu(), torch.randn(28, 64, 4096, dtype=torch.float16).npu()])
        
if __name__ == '__main__':
    unittest.main()
