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
import numpy as np
import torch
import torch_npu
import random


sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

def generateSplitSizes(splitNum, dimNum):
    splitSizes = []
    splitPoint = random.sample(range(1, dimNum), splitNum -1)
    splitPoint.sort()
    lastPoint = 0
    for i in range(len(splitPoint)):
        splitSizes.append(splitPoint[i] - lastPoint)
        lastPoint = splitPoint[i]
    splitSizes.append(dimNum - lastPoint)
    return splitSizes

class TestAddOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        if self.splitNum == 3: 
            x = in_tensors[0].half()             
            y = torch.split(x, self.OP_PARAM["splitSizes"], dim = self.splitDim)
            return [y[0], y[1], y[2]]
        elif self.splitNum == 2:
            x = in_tensors[0].half()             
            y = torch.split(x, self.OP_PARAM["splitSizes"], dim = self.splitDim)
            return [y[0], y[1]]
        else:
            return [torch.zeros_like(x) for x in in_tensors]

    def test_2d_half(self):
        shape = (8192,8192)
        self.OP_NAME = "SplitOperation"
        self.splitNum = 2
        self.splitDim = 0
        self.OP_PARAM = {
            "splitNum": self.splitNum,
            "splitDim": self.splitDim,
            "splitSizes": generateSplitSizes(self.splitNum, 8192) # equal to shape 
        }
        intensor0 = np.random.uniform(low=1, high=10, size=shape).astype(np.float32) 
        x = self.execute(self.OP_NAME, self.OP_PARAM, [torch.from_numpy(intensor0).half().npu()])

    def test_3d_half(self):
        shape = (8192,8192)
        self.OP_NAME = "SplitOperation"
        self.splitNum = 3
        self.splitDim = 0
        self.OP_PARAM = {
            "splitNum": self.splitNum,
            "splitDim": self.splitDim,
            "splitSizes": generateSplitSizes(self.splitNum, 8192) # equal to dimNum of shape 
        }
        intensor0 = np.random.uniform(low=1, high=10, size=shape).astype(np.float32)
        x = self.execute(self.OP_NAME, self.OP_PARAM, [torch.from_numpy(intensor0).half().npu()])

if __name__ == '__main__':
    unittest.main()
