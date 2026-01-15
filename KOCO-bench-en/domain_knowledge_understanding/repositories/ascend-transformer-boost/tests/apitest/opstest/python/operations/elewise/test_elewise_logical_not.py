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


class TestElewiseLogicalNotOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        gold_bool = torch.logical_not(in_tensors[0])
        return [gold_bool.type(torch.int8)]
        
    def test_2d_bool(self):
        self.execute(OP_NAME, {"elewiseType": 7}, [torch.randint(
            2, size=(2, 2), dtype=torch.int8).npu()])

if __name__ == '__main__':
    unittest.main()
