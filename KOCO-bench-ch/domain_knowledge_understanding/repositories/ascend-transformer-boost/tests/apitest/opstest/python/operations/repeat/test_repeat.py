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

OP_NAME = "RepeatOperation"
PARAM1 = {"multiples": [1, 4, 4]}
PARAM2 = {"multiples": [1, 1, 1, 2]}
PARAM3 = {"multiples": [1, 1, 1, 16, 1]}


class TestRepeatOperation1(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        outtensor = in_tensors[0].repeat(PARAM1["multiples"])
        return [outtensor]

    def test(self):
        intensor0 = torch.rand(2, 3, 5).npu().half()
        self.execute(OP_NAME, PARAM1, [intensor0])


class TestRepeatOperation2(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        outtensor = in_tensors[0].repeat(PARAM2["multiples"])
        return [outtensor]

    def test(self):
        intensor0 = torch.rand(1, 2, 32, 1).npu().half()
        self.execute(OP_NAME, PARAM2, [intensor0])


class TestRepeatOperation3(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        outtensor = in_tensors[0].repeat(PARAM2["multiples"])
        return [outtensor]

    def test(self):
        intensor0 = torch.rand(256, 2, 32, 1).npu().half()
        self.execute(OP_NAME, PARAM2, [intensor0])


class TestRepeatOperation4(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        outtensor = in_tensors[0].repeat(PARAM3["multiples"])
        return [outtensor]

    def test(self):
        intensor0 = torch.rand(256, 2, 1, 1, 128).npu().half()
        intensor0 = torch_npu.npu_format_cast(intensor0, 2)
        self.execute(OP_NAME, PARAM3, [intensor0])


if __name__ == '__main__':
    unittest.main()
