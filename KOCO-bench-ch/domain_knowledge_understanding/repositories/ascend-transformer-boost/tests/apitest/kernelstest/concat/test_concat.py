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


OP_NAME = "ConcatOperation"
OP_PARAM = {"concatDim": 0}


class TestConcat(op_test.OpTest):
    def golden_calc(self, in_tensors):
        num = self.op_desc["specificParam"]["concatDim"]
        x = in_tensors[0].float()
        z = in_tensors[1].float()
        y = torch.cat((x, z), num)
        return [y]

    def golden_compare(self, out_tensors, golden_out_tensors):
        if out_tensors[0].dtype == torch.bfloat16:
            return torch.allclose(out_tensors[0].bfloat16(), golden_out_tensors[0].bfloat16(), rtol=0.001, atol=0.001)
        elif out_tensors[0].dtype == torch.float16:
            return torch.allclose(out_tensors[0].half(), golden_out_tensors[0].half(), rtol=0.001, atol=0.001)
        elif out_tensors[0].dtype == torch.float32:
            return torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)
        else:
            return False
    
    @op_test.only_910b
    def test_concat_bf16(self):
        shape = (2, 4)
        shape0 = (4, 4)
        input0 = np.random.uniform(low=-1000, high=1000, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-1000, high=1000, size=shape).astype(np.float16)
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([torch.from_numpy(input0).bfloat16(),torch.from_numpy(input1).bfloat16()],
                     [torch.zeros(shape0).bfloat16()])

    def test_concat_fp16(self):
        shape = (2, 4)
        shape0 = (4, 4)
        input0 = np.random.uniform(low=-1000, high=1000, size=shape).astype(np.float16)
        input1 = np.random.uniform(low=-1000, high=1000, size=shape).astype(np.float16)
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([torch.from_numpy(input0).half(),torch.from_numpy(input1).half()],
                     [torch.zeros(shape0).half()])

    @op_test.only_910b
    def test_concat_fp32(self):
        shape = (2, 4)
        shape0 = (4, 4)
        input0 = np.random.uniform(low=-1000, high=1000, size=shape).astype(np.float32)
        input1 = np.random.uniform(low=-1000, high=1000, size=shape).astype(np.float32)
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([torch.from_numpy(input0).float(),torch.from_numpy(input1).float()],
                     [torch.zeros(shape0).float()])

if __name__ == '__main__':
    unittest.main()

