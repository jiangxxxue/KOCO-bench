#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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
 
 
OP_NAME = "SplitOperation"
OP_PARAM_Splite3 = {"splitDim": 0,"splitNum": 3}
OP_PARAM_Splite2 = {"splitDim": 0,"splitNum": 2}

class TestSplit(op_test.OpTest):
    def golden_calc(self, in_tensors):
        splitNum = self.op_desc["specificParam"]["splitNum"]
        if splitNum == 3: 
            x = in_tensors[0]
            y = torch.split(x, x.shape[0]//3, dim=0)
            return [y]
        elif splitNum == 2:
            x = in_tensors[0]
            y = torch.split(x, x.shape[0]//2, dim=0)
            return [y]
        else:
            return [torch.zeros_like(x) for x in in_tensors]
 
    def golden_compare(self, out_tensors, golden_out_tensors):
        splitNum = self.op_desc["specificParam"]["splitNum"]
        for i in range(splitNum):
            out_tensor = out_tensors[i]
            golden_tensor = golden_out_tensors[0][i]
            if not torch.equal(out_tensor, golden_tensor):
                return False
        return True
    
    @op_test.only_910b
    def test_Splite3_bf16(self):
        shape = (4096 * 3)
        input0 = np.random.uniform(low=4, high=7, size=shape).astype(np.float32) 
        self.set_param(OP_NAME, OP_PARAM_Splite3)
        self.execute([torch.from_numpy(input0).bfloat16()],
                     [torch.zeros(shape//3).bfloat16(), torch.zeros(shape//3).bfloat16(), torch.zeros(shape//3).bfloat16()])
    
    @op_test.only_910b
    def test_Splite2_bf16(self):
        shape = (4096 * 2)
        input1 = np.random.uniform(low=4, high=7, size=shape).astype(np.float32) 
        self.set_param(OP_NAME, OP_PARAM_Splite2)
        self.execute([torch.from_numpy(input1).bfloat16()],
                     [torch.zeros(shape//2).bfloat16(), torch.zeros(shape//2).bfloat16()])

    def test_Splite3_fp16(self):
        shape = (4096 * 3)
        input0 = np.random.uniform(low=4, high=7, size=shape).astype(np.float16)
        self.set_param(OP_NAME, OP_PARAM_Splite3)
        self.execute([torch.from_numpy(input0).half()],
                     [torch.zeros(shape//3).half(), torch.zeros(shape//3).half(), torch.zeros(shape//3).half()])

    def test_Splite2_fp16(self):
        shape = (4096 * 2)
        input1 = np.random.uniform(low=4, high=7, size=shape).astype(np.float16)
        self.set_param(OP_NAME, OP_PARAM_Splite2)
        self.execute([torch.from_numpy(input1).half()],
                     [torch.zeros(shape//2).half(), torch.zeros(shape//2).half()])

if __name__ == '__main__':
    unittest.main()