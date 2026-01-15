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

OP_NAME = "FillOperation"
PARAM = {"withMask": True, "value": -10000}


class TestFillOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        value = self.op_param["value"]
        outtensor = in_tensors[0].masked_fill(in_tensors[1], value)
        return [outtensor]

    def test_float16(self):
        if  operation_test.get_soc_version() == 'Ascend310B':
            print("this testcase don't supports Ascend310B")
            return True
        intensor0 = torch.rand(5, 5).npu().half()
        intentor1 = (torch.randint(2, (5, 5)) == torch.randint(1, (5, 5))).to(torch.bool).npu()
        PARAM = {"withMask": True, "value": -10000}
        self.execute(OP_NAME, PARAM, [intensor0, intentor1])
        PARAM = {"withMask": True, "value": 10000}
        self.execute_update_param(OP_NAME,PARAM,[intensor0, intentor1])

    def test_bfloat16(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        intensor0 = torch.rand(5, 5).npu().bfloat16()
        intentor1 = (torch.randint(2, (5, 5)) == torch.randint(1, (5, 5))).to(torch.bool).npu()
        PARAM = {"withMask": True, "value": -10000}
        self.execute(OP_NAME, PARAM, [intensor0, intentor1])
        PARAM = {"withMask": True, "value": 10000}
        self.execute_update_param(OP_NAME,PARAM,[intensor0, intentor1])

if __name__ == '__main__':
    unittest.main()
