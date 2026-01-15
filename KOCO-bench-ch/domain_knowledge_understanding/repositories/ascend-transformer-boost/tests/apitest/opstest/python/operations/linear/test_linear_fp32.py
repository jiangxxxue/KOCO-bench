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
import sys, os
import random
import math

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test


OP_NAME = "LinearOperation"
OP_PARAM = {"transposeA": False, "transposeB": True, "hasBias": False}

def generate_data(m, n, k):
    matrix_a = torch.randn(size=(m, k), dtype=torch.float32)
    matrix_b = torch.randn(size=(n, k), dtype=torch.float32)

    golden = torch.matmul(matrix_a.to(torch.float32), matrix_b.t().to(torch.float32)).to(torch.float32)

    return matrix_a, matrix_b, golden

class TestLinearFP32(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        return [self.golden.npu()]

    def golden_compare(self, out_tensors, golden_out_tensors):
        output = out_tensors[0].to(torch.float32)
        golden = golden_out_tensors[0]

        absolute_error = torch.abs(output - golden)
        error_bound = self.rtol * torch.max(torch.tensor(1).npu(), torch.abs(golden))
        error_coords = (absolute_error > error_bound).nonzero()

        for error_index in range(error_coords.shape[0]):
            error_coord = tuple(error_coords[error_index].tolist())
            actual = output[error_coord]
            expect = golden[error_coord]
            print(
                f"Element index: {error_coord}, expected: {expect:-.4f}, actual: {actual:-.4f}, "
                f"rdiff: {abs(actual - expect) / expect:-.4f}"
            )
            if error_index >= 10:
                break
        return len(error_coords) == 0

    def testcase(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return

        n = 256
        k = 7168
        self.rtol = 2 ** -10

        shapes = []
        for m in range(1, 257):
            shapes.append([m, n, k])
        for shape in shapes:
            matrix_a, matrix_b, golden = generate_data(shape[0], shape[1], shape[2])
            self.golden = golden

            in_tensors = [matrix_a.npu(), matrix_b.npu()]
            self.execute(OP_NAME, OP_PARAM, in_tensors)

if __name__ == '__main__':
    unittest.main()
