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

OP_NAME = "TransposeOperation"

class TestTranspose(op_test.OpTest):
    def golden_calc(self, in_tensors):
        perm = self.op_desc["specificParam"]["perm"]
        return [in_tensors[0].permute(perm)]

    def golden_compare(self, out_tensors, golden_out_tensors):
        return torch.equal(out_tensors[0], golden_out_tensors[0])

    @op_test.only_910b
    def test_transpose_bf16(self):
        shape = (8192, 2752)
        transpose_shape = (2752, 8192)
        param = {"perm": [1, 0]}
        input0 = np.random.uniform(low=-100, high=100, size=shape).astype(np.float32)
        self.set_param(OP_NAME, param)

        self.execute([torch.from_numpy(input0).bfloat16()],
                     [torch.zeros(transpose_shape).bfloat16()])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_transpose_bf32(self):
        shape = (8192, 2752)
        transpose_shape = (2752, 8192)
        param = {"perm": [1, 0]}
        input0 = np.random.uniform(low=-100, high=100, size=shape).astype(np.float32)
        self.set_param(OP_NAME, param)

        self.execute([torch.from_numpy(input0).float()],
                     [torch.zeros(transpose_shape).float()])
        
    def test_transpose_fp16(self):
        shape = (8192, 2752)
        transpose_shape = (2752, 8192)
        param = {"perm": [1, 0]}
        input0 = np.random.uniform(low=-100, high=100, size=shape).astype(np.float16)
        self.set_param(OP_NAME, param)

        self.execute([torch.from_numpy(input0).half()],
                     [torch.zeros(transpose_shape).half()])
        
    @op_test.skip_310b
    @op_test.skip_910a
    def test_transpose_i32(self):
        shape = (8192, 2752)
        transpose_shape = (2752, 8192)
        param = {"perm": [1, 0]}
        input0 = np.random.uniform(low=-100, high=100, size=shape).astype(np.int32)
        self.set_param(OP_NAME, param)

        self.execute([torch.from_numpy(input0).to(torch.int32)],
                     [torch.zeros(transpose_shape).to(torch.int32)])
    
    def test_transpose_i64(self):
        shape = (8192, 2752)
        transpose_shape = (2752, 8192)
        param = {"perm": [1, 0]}
        input0 = np.random.uniform(low=-100, high=100, size=shape).astype(np.int64)
        self.set_param(OP_NAME, param)

        self.execute([torch.from_numpy(input0).to(torch.int64)],
                     [torch.zeros(transpose_shape).to(torch.int64)])
    
    @op_test.skip_310b
    @op_test.skip_910a
    def test_transpose_int8(self):
        shape = (8192, 2752)
        transpose_shape = (2752, 8192)
        param = {"perm": [1, 0]}
        input0 = np.random.uniform(low=-100, high=100, size=shape).astype(np.int8)
        self.set_param(OP_NAME, param)

        self.execute([torch.from_numpy(input0)],
                     [torch.zeros(transpose_shape,dtype=torch.int8)])
        
if __name__ == '__main__':
    unittest.main()