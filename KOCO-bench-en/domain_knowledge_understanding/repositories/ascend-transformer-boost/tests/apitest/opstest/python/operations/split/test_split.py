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

OP_NAME = "SplitOperation"
SPLIT_NUM = 2
SPLIT_DIM = -1
PARAM = {"splitDim": SPLIT_DIM, "splitNum": SPLIT_NUM}


class TestAddOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        return torch.chunk(in_tensors[0], chunks=SPLIT_NUM, dim=SPLIT_DIM)

    def test_2d_half(self):
        intensor0 = torch.rand(4096, 22016).npu().half()
        x = self.execute(OP_NAME, PARAM, [intensor0])

    def test_2d_bf16(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        intensor0 = torch.rand(4096, 22016).npu().bfloat16()
        x = self.execute(OP_NAME, PARAM, [intensor0])


if __name__ == '__main__':
    unittest.main()
