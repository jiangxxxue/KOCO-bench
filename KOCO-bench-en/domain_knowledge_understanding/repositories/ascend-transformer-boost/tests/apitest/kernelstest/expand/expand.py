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

OP_NAME = "ExpandOperation"
OP_PARAM = {"shape": [4, 4]}
class TestExpand(op_test.OpTest):
    def golden_calc(self, in_tensors):
        shape = self.op_desc["specificParam"]["shape"]
        res = in_tensors[0].expand(shape)
        return [res]

    def golden_compare(self, out_tensors, golden_out_tensors):
        result0 = torch.equal(out_tensors[0], golden_out_tensors[0])
        if not result0:
            return False
        return True

    @op_test.only_910b
    def test_expand_bf16(self):
        shape0 = (4, 1)
        shape1 = (4, 4)
        input0 = np.random.uniform(low=4, high=7, size=shape0).astype(np.float32) 
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([torch.from_numpy(input0).bfloat16()],
                     [torch.zeros(shape1).bfloat16()])

    def test_expand_fp16(self):
        shape0 = (4, 1)
        shape1 = (4, 4)
        input0 = np.random.uniform(low=4, high=7, size=shape0).astype(np.float16)
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([torch.from_numpy(input0).half()],
                     [torch.zeros(shape1).half()])
    
if __name__ == "__main__":
    unittest.main()
