#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import unittest
import numpy as np
import torch
import torch.nn.functional as F
import op_test
 
 
OP_NAME = "CopyOperation"
OP_PARAM_VIEW_COPY = {"dstSize": [2, 2, 2],"dstStride": [6, 3, 1], "dstOffset": [0]}

class TestCopy(op_test.OpTest):
    def golden_calc(self, in_tensors):
        dstSize = self.op_desc["specificParam"]["dstSize"]
        dstStride = self.op_desc["specificParam"]["dstStride"]
        dstOffset = self.op_desc["specificParam"]["dstOffset"]

        x = in_tensors[0]
        y = in_tensors[1]
        z = torch.as_strided(x, size=dstSize, stride=dstStride, storage_offset=dstOffset[0])
        z.copy_(y)
        return [x]
 
    def golden_compare(self, out_tensors, golden_out_tensors):
        return torch.equal(out_tensors[0], golden_out_tensors[0])
    
    @op_test.only_910b
    def test_viewcopy_fp16(self):
        shape0 = (3,2,3)
        shape1 = (2,2,2)
        input0 = np.random.uniform(low=-5, high=10, size=shape0).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=10, size=shape1).astype(np.float16)
        self.set_param(OP_NAME, OP_PARAM_VIEW_COPY)
        self.execute([torch.from_numpy(input0), torch.from_numpy(input1)],
                     [torch.from_numpy(input0)])
        
    @op_test.only_910b
    def test_viewcopy_bf16(self):
        shape0 = (3,2,3)
        shape1 = (2,2,2)
        input0 = np.random.uniform(low=-5, high=10, size=shape0).astype(np.float32)
        input1 = np.random.uniform(low=-5, high=10, size=shape1).astype(np.float32)
        self.set_param(OP_NAME, OP_PARAM_VIEW_COPY)
        self.execute([torch.from_numpy(input0).bfloat16(), torch.from_numpy(input1).bfloat16()],
                     [torch.from_numpy(input0).bfloat16()])
if __name__ == '__main__':
    unittest.main()