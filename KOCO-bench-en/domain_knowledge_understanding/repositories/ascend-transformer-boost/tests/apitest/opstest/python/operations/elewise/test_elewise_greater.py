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


class TestElewiseGreaterOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        return [torch.gt(in_tensors[0], in_tensors[1]).type(torch.int8)]

    def test_3d_half(self):
        self.execute(OP_NAME, {"elewiseType": 14}, [torch.randn(
            16, 4, 128).npu().half(), torch.randn(16, 4, 128).npu().half()])
        

if __name__ == '__main__':
    unittest.main()
