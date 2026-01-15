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
import op_test


OP_NAME = "GatherOperation"


class TestGather(op_test.OpTest):
    def golden_calc(self, in_tensors):
        dim = self.op_desc["specificParam"]["axis"][0]
        dtype = in_tensors[0].dtype
        return [torch.gather(in_tensors[0], dim, in_tensors[1].long()).to(dtype)]

    def golden_compare(self, out_tensors, golden_out_tensors):
        return torch.allclose(out_tensors[0], golden_out_tensors[0], rtol=0.001, atol=0.001)
    
    def test_gather_f16_i64_8_32000(self):
        op_param = {"batchDims": 1, "axis": [1]}
        shape = (8, 32000)
        input0 = np.random.uniform(low=-10000, high=10000, size=shape).astype(np.float16)
        indices = np.random.uniform(low=0, high=shape[1], size=shape).astype(np.int64)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(indices)],
                     [torch.zeros(shape).half()])

    def test_gather_f16_i32_8_32000(self):
        op_param = {"batchDims": 1, "axis": [1]}
        shape = (8, 32000)
        input0 = np.random.uniform(low=-10000, high=10000, size=shape).astype(np.float16)
        indices = np.random.uniform(low=0, high=shape[1], size=shape).astype(np.int32)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(indices)],
                     [torch.zeros(shape).half()])
        
    def test_gather_int32_int64_8_24000(self):
        op_param = {"batchDims": 1, "axis": [1]}
        shape = (8, 24000)
        input0 = np.random.uniform(low=-10000, high=10000, size=shape).astype(np.int32)
        indices = np.random.uniform(low=0, high=shape[1], size=shape).astype(np.int64)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(indices)],
                     [torch.zeros(shape).int()])
        
    def test_gather_int32_int32_8_24000(self):
        op_param = {"batchDims": 1, "axis": [1]}
        shape = (8, 24000)
        input0 = np.random.uniform(low=-10000, high=10000, size=shape).astype(np.int32)
        indices = np.random.uniform(low=0, high=shape[1], size=shape).astype(np.int32)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(indices)],
                     [torch.zeros(shape).int()])
    
    @op_test.only_910b
    def test_gather_bf16_int32_12_24000(self):
        op_param = {"batchDims": 1, "axis": [1]}
        shape = (12, 24000)
        input0 = torch.rand(shape) * 10000 + 20000
        input0 = input0.bfloat16()
        indices = np.random.uniform(low=0, high=shape[1], size=shape).astype(np.int32)
        self.set_param(OP_NAME, op_param)
        self.execute([input0, torch.from_numpy(indices)],
                     [torch.zeros(shape).bfloat16()])
        
    def test_gather_int64_int64_8_24000(self):
        op_param = {"batchDims": 1, "axis": [1]}
        shape = (8, 24000)
        input0 = np.random.uniform(low=-10000, high=10000, size=shape).astype(np.int64)
        indices = np.random.uniform(low=0, high=shape[1], size=shape).astype(np.int64)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(indices)],
                     [torch.zeros(shape).to(torch.int64)])
        
    def test_gather_int64_int32_8_24000(self):
        op_param = {"batchDims": 1, "axis": [1]}
        shape = (8, 24000)
        input0 = np.random.uniform(low=-10000, high=10000, size=shape).astype(np.int64)
        indices = np.random.uniform(low=0, high=shape[1], size=shape).astype(np.int32)
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0), torch.from_numpy(indices)],
                     [torch.zeros(shape).to(torch.int64)])


if __name__ == '__main__':
    unittest.main()
