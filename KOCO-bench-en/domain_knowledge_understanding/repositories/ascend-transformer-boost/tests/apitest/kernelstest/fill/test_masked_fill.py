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

OP_NAME = "FillOperation"
OP_PARAM = {"withMask": True, "value": [1]}
class TestFill(op_test.OpTest):
    def golden_calc(self, in_tensors):
        value = self.op_desc["specificParam"]["value"]
        res = in_tensors[0].masked_fill(in_tensors[1].bool(), value[0])
        return [res]

    def golden_compare(self, out_tensors, golden_out_tensors):
        result0 = torch.equal(out_tensors[0], golden_out_tensors[0])
        return result0

    def test_masked_fill_fp16(self):
        shape0 = (4, 4)
        input0 = np.random.uniform(low=4, high=7, size=shape0).astype(np.float16)
        input1 = np.random.choice([0, 1], size=shape0).astype(np.int8)
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([torch.from_numpy(input0).half(), torch.from_numpy(input1).to(torch.int8)],
                     [torch.zeros(shape0).half()])

    @op_test.only_910b        
    def test_masked_fill_bf16(self):
        shape0 = (4, 4)
        input0 = np.random.uniform(low=4, high=7, size=shape0).astype(np.float32)
        input1 = np.random.choice([0, 1], size=shape0).astype(np.int8)
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([torch.from_numpy(input0).bfloat16(), torch.from_numpy(input1).to(torch.int8)],
                     [torch.zeros(shape0).bfloat16()])
    
if __name__ == "__main__":
    unittest.main()
