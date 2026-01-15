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
 
 
OP_NAME = "ReverseOperation"
OP_PARAM_REVERSE = {'axis': [0]}

class TestCopy(op_test.OpTest):
    def golden_calc(self, in_tensors):
        axis = self.op_desc["specificParam"]["axis"]
        res = torch.flip(in_tensors[0], axis)
        return [res]
 
    def golden_compare(self, out_tensors, golden_out_tensors):
        return torch.equal(out_tensors[0], golden_out_tensors[0])
    
    @op_test.only_910b
    def test_viewcopy_fp16(self):
        shape0 = (3,2,3)
        input0 = np.random.uniform(low=-5, high=10, size=shape0).astype(np.float16)
        self.set_param(OP_NAME, OP_PARAM_REVERSE)
        self.execute([torch.from_numpy(input0)],
                     [torch.zeros(shape0).half()])
        
    @op_test.only_910b
    def test_viewcopy_bf16(self):
        shape0 = (3,2,3)
        input0 = np.random.uniform(low=-5, high=10, size=shape0).astype(np.float32)
        self.set_param(OP_NAME, OP_PARAM_REVERSE)
        self.execute([torch.from_numpy(input0).bfloat16()],
                     [torch.zeros(shape0).bfloat16()])
if __name__ == '__main__':
    unittest.main()