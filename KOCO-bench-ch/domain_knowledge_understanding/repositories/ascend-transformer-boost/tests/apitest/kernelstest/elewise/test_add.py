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
OP_PARAM_ADD = {"elewiseType": 8}

class TestElewise(op_test.OpTest):
    def golden_calc(self, in_tensors):
        actvation_type = self.op_desc["specificParam"]["elewiseType"]
        if actvation_type == 8:
            x = in_tensors[0].float()
            z = in_tensors[1].float()
            y = torch.add(x, z)
            return [y]
        else:
            return [torch.zeros_like(x) for x in in_tensors]

    def golden_compare(self, out_tensors, golden_out_tensors):
        dtype = out_tensors[0].dtype
        if (dtype == torch.bfloat16):
            return torch.allclose(out_tensors[0].bfloat16(), golden_out_tensors[0].bfloat16(), rtol=0.001, atol=0.001)
        elif dtype in [torch.int32, torch.int64]:
            return torch.equal(out_tensors[0].to(dtype), golden_out_tensors[0].to(dtype))
        return torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)

    @op_test.only_910b
    def test_add_bf16(self):
        shape = (1000000)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float32)
        input1 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float32)

        self.set_param(OP_NAME, OP_PARAM_ADD)
        self.execute([torch.from_numpy(input0).bfloat16(),torch.from_numpy(input1).bfloat16()],
                     [torch.zeros(shape).bfloat16()])

    def test_add_fp16(self):
        shape = (1000000)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)

        self.set_param(OP_NAME, OP_PARAM_ADD)
        self.execute([torch.from_numpy(input0).to(torch.float16),torch.from_numpy(input1).to(torch.float16)],
                     [torch.zeros(shape).to(torch.float16)])

    def test_add_fp32(self):
        shape = (1000000)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float32)
        input1 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float32)

        self.set_param(OP_NAME, OP_PARAM_ADD)
        self.execute([torch.from_numpy(input0).to(torch.float32),torch.from_numpy(input1).to(torch.float32)],
                     [torch.zeros(shape).to(torch.float32)])
    
    def test_add_i32(self):
        shape = (1000000)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.int32)
        input1 = np.random.uniform(low=-5, high=5, size=shape).astype(np.int32)

        self.set_param(OP_NAME, OP_PARAM_ADD)
        self.execute([torch.from_numpy(input0).to(torch.int32),torch.from_numpy(input1).to(torch.int32)],
                     [torch.zeros(shape).to(torch.int32)])
        
    def test_add_i64(self):
        shape = (1000000)
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.int64)
        input1 = np.random.uniform(low=-5, high=5, size=shape).astype(np.int64)

        self.set_param(OP_NAME, OP_PARAM_ADD)
        self.execute([torch.from_numpy(input0).to(torch.int64),torch.from_numpy(input1).to(torch.int64)],
                     [torch.zeros(shape).to(torch.int64)])

if __name__ == '__main__':
    unittest.main()
