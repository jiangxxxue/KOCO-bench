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
import json
import unittest
import torch
import torch_npu

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

OP_NAME = "SetValueOperation"


class TestSetValueOperation(operation_test.OperationTest):
    def compare(self, in_tensors, PARAM, outtensor):
        operation = torch.classes.OperationTorch.OperationTorch(
            OP_NAME)
        operation.set_param(PARAM)
        if isinstance(PARAM, dict):
            operation.set_param(json.dumps(PARAM))
        elif isinstance(PARAM, str):
            operation.set_param(PARAM)
        out_tensors = operation.execute(in_tensors)
        result = torch.allclose(outtensor.npu(), in_tensors[0], rtol=0.001, atol=0.001)
        return result

    def test_int64(self):
        if  operation_test.get_soc_version() == 'Ascend310B':
            print("this testcase don't supports Ascend310B")
            return True
        PARAM = '{"starts": [0,0,0], "ends": [28,3,4096], "strides": [1,2,1]}'
        intensor1 = torch.randint(20, (28, 10, 4096), dtype=torch.int64)
        intensor2 = torch.randint(20, (28, 2, 4096), dtype=torch.int64)
        outtensor = intensor1.clone()
        outtensor[0:28, 0:3:2, 0:4096].copy_(intensor2)
        in_tensors = [intensor1.npu(), intensor2.npu()]
        self.assertFalse(self.compare(in_tensors, PARAM, outtensor))

    def test_2d_int64(self):
        if  operation_test.get_soc_version() == 'Ascend310B':
            print("this testcase don't supports Ascend310B")
            return True
        PARAM = '{"starts": [0,0,0], "ends": [28,2,9], "strides": [1,1,2]}'
        intensor1 = torch.randint(20, (28, 10, 4096), dtype=torch.int64)
        intensor2 = torch.randint(20, (28, 2, 5), dtype=torch.int64)
        outtensor = intensor1.clone()
        outtensor[0:28, 0:2, 0:9:2].copy_(intensor2)
        in_tensors = [intensor1.npu(), intensor2.npu()]
        self.assertFalse(self.compare(in_tensors, PARAM, outtensor))

    def test_3d_int64(self):
        if  operation_test.get_soc_version() == 'Ascend310B':
            print("this testcase don't supports Ascend310B")
            return True
        PARAM = '{"starts": [0,0,0], "ends": [13,2,4096], "strides": [1,1,1]}'
        intensor1 = torch.randint(20, (28, 10, 4096), dtype=torch.int64)
        intensor2 = torch.randint(20, (13, 2, 4096), dtype=torch.int64)
        outtensor = intensor1.clone()
        outtensor[0:13, 0:2, 0:4096].copy_(intensor2)
        in_tensors = [intensor1.npu(), intensor2.npu()]
        self.assertTrue(self.compare(in_tensors, PARAM, outtensor))

    def test_4d_int64(self):
        if  operation_test.get_soc_version() == 'Ascend310B':
            print("this testcase don't supports Ascend310B")
            return True
        PARAM = '{"starts": [1,3], "ends": [3,10], "strides": [1,3]}'
        intensor1 = torch.randint(20, (7, 16), dtype=torch.int64)
        intensor2 = torch.randint(20, (2, 3), dtype=torch.int64)
        outtensor = intensor1.clone()
        print("in_tensor0", intensor1)
        print("in_tensor1", intensor2)
        outtensor[1:3, 3:10:3].copy_(intensor2)
        in_tensors = [intensor1.npu(), intensor2.npu()]
        self.assertFalse(self.compare(in_tensors, PARAM, outtensor))

    def test_5d_int64(self):
        if  operation_test.get_soc_version() == 'Ascend310B':
            print("this testcase don't supports Ascend310B")
            return True
        PARAM = '{"starts": [1,3], "ends": [3,6], "strides": [1,1]}'
        intensor1 = torch.randint(20, (7, 16), dtype=torch.int64)
        intensor2 = torch.randint(20, (2, 3), dtype=torch.int64)
        outtensor = intensor1.clone()
        print("in_tensor0", intensor1)
        print("in_tensor1", intensor2)
        outtensor[1:3, 3:6].copy_(intensor2)
        in_tensors = [intensor1.npu(), intensor2.npu()]
        self.assertTrue(self.compare(in_tensors, PARAM, outtensor))

    def test_6d_int64(self):
        if  operation_test.get_soc_version() == 'Ascend310B':
            print("this testcase don't supports Ascend310B")
            return True
        PARAM = '{"starts": [2,0,1], "ends": [23,3,66], "strides": [4,2,1]}'
        intensor1 = torch.randint(20, (40, 11, 4096), dtype=torch.int64)
        intensor2 = torch.randint(20, (6, 2, 65), dtype=torch.int64)
        outtensor = intensor1.clone()
        outtensor[2:23:4, 0:3:2, 1:66].copy_(intensor2)
        in_tensors = [intensor1.npu(), intensor2.npu()]
        self.assertFalse(self.compare(in_tensors, PARAM, outtensor))

    def test_int32(self):
        if  operation_test.get_soc_version() == 'Ascend310B':
            print("this testcase don't supports Ascend310B")
            return True
        PARAM = '{"starts": [0,0,0], "ends": [13,2,4096], "strides": [1,1,1]}'
        intensor1 = torch.randint(20, (28, 10, 4096), dtype=torch.int32)
        intensor2 = torch.randint(20, (13, 2, 4096), dtype=torch.int32)
        outtensor = intensor1.clone()
        outtensor[0:13, 0:2, 0:4096].copy_(intensor2)
        in_tensors = [intensor1.npu(), intensor2.npu()]
        self.assertTrue(self.compare(in_tensors, PARAM, outtensor))

    def test_float16(self):
        if  operation_test.get_soc_version() == 'Ascend310B':
            print("this testcase don't supports Ascend310B")
            return True
        PARAM = '{"starts": [0,0,0], "ends": [13,2,4096], "strides": [1,1,1]}'
        intensor1 = torch.randn(28, 10, 4096, dtype=torch.float16)
        intensor2 = torch.randn(13, 2, 4096, dtype=torch.float16)
        outtensor = intensor1.clone()
        outtensor[0:13, 0:2, 0:4096].copy_(intensor2)
        in_tensors = [intensor1.npu(), intensor2.npu()]
        self.assertTrue(self.compare(in_tensors, PARAM, outtensor))
    
    def test_float32(self):
        if  operation_test.get_soc_version() == 'Ascend310B':
            print("this testcase don't supports Ascend310B")
            return True
        PARAM = '{"starts": [0,0,0], "ends": [13,2,4096], "strides": [1,1,1]}'
        intensor1 = torch.randn(28, 10, 4096, dtype=torch.float32)
        intensor2 = torch.randn(13, 2, 4096, dtype=torch.float32)
        outtensor = intensor1.clone()
        outtensor[0:13, 0:2, 0:4096].copy_(intensor2)
        in_tensors = [intensor1.npu(), intensor2.npu()]
        self.assertTrue(self.compare(in_tensors, PARAM, outtensor))

if __name__ == '__main__':
    unittest.main()
