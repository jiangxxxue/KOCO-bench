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
import torch_npu
import torch.nn.functional as F
import op_test


OP_NAME = "CumsumOperation"


class TestCumsum(op_test.OpTest):
    def golden_calc(self, in_tensors):
        axis = self.op_desc["specificParam"]["axis"][0]
        if in_tensors[0].dtype == torch.bfloat16:
            x = in_tensors[0]
            x = x.to(torch.float32)
            x = x.numpy()
        else:
            x = in_tensors[0].numpy()
        return [torch.from_numpy(np.cumsum(x, axis=axis))]

    def golden_compare(self, out_tensors, golden_out_tensors):
        if out_tensors[0].dtype == torch.bfloat16:
            return torch.allclose(out_tensors[0].bfloat16(), golden_out_tensors[0].bfloat16(), rtol= 2 ** -7, atol= 2 ** -7)
        else:
            return torch.allclose(out_tensors[0], golden_out_tensors[0].half(), rtol= 2 ** -8, atol= 2 ** -8)

    @op_test.skip_310b
    @op_test.skip_910a
    def test_cumsum_dtm(self):
        shape = (128, 1111)
        input0 = np.random.uniform(low=-10, high=10, size=shape).astype(np.float16)

        op_param = {"axis": [1], "deterministic": True}
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0)], [torch.zeros(shape).half()])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_cumsum(self):
        shape = (128, 2222)
        input0 = np.random.uniform(low=-10, high=10, size=shape).astype(np.float16)

        op_param = {"axis": [1]}
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0)], [torch.zeros(shape).half()])

    @op_test.only_910b
    def test_cumsum_bf16_dtm(self):
        shape = (128, 1111)
        input0 = np.random.uniform(low=-10, high=10, size=shape).astype(np.float32)

        op_param = {"axis": [1], "deterministic": True}
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0).bfloat16()], [torch.zeros(shape).bfloat16()])

    @op_test.only_910b
    def test_cumsum_bf16(self):
        shape = (128, 2222)
        input0 = np.random.uniform(low=-10, high=10, size=shape).astype(np.float32)

        op_param = {"axis": [1]}
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0).bfloat16()], [torch.zeros(shape).bfloat16()])

if __name__ == '__main__':
    unittest.main()
