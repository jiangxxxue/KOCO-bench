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


OP_NAME = "ElewiseOperation"
OP_PARAM = {"elewiseType": 17}

class TestElewise(op_test.OpTest):
    def golden_calc(self, in_tensors):
        res = torch.eq(in_tensors[0], in_tensors[1]).int().to(torch.int8)
        return [res]

    def golden_compare(self, out_tensors, golden_out_tensors):
        result0 = torch.equal(out_tensors[0], golden_out_tensors[0])
        if not result0:
            return False
        return True
    
    @op_test.only_910b
    def test_cos_bf16(self):
        shape = (8, 6)
        input0 = np.random.uniform(low=0, high=100, size=shape).astype(np.float32)
        input1 = np.random.uniform(low=0, high=100, size=shape).astype(np.float32)

        self.set_param(OP_NAME, OP_PARAM)
        self.execute([torch.from_numpy(input0).bfloat16(), torch.from_numpy(input1).bfloat16()],
                     [torch.zeros(shape).to(torch.int8)])

    def test_cos_fp16(self):
        shape = (8, 6)
        input0 = np.random.uniform(low=0, high=100, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=0, high=100, size=shape).astype(np.float16)

        self.set_param(OP_NAME, OP_PARAM)
        self.execute([torch.from_numpy(input0).to(torch.float16), torch.from_numpy(input1).to(torch.float16)],
                     [torch.zeros(shape).to(torch.int8)])

    def test_cos_fp32(self):
        shape = (8, 6)
        input0 = np.random.uniform(low=0, high=100, size=shape).astype(np.float32)
        input1 = np.random.uniform(low=0, high=100, size=shape).astype(np.float32)
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([torch.from_numpy(input0).to(torch.float32), torch.from_numpy(input1).to(torch.float32)],
                     [torch.zeros(shape).to(torch.int8)])

if __name__ == '__main__':
    unittest.main()
