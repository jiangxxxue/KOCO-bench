#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import sys
import os
import unittest
import torch
import torch_npu
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

OP_NAME = "ElewiseOperation"


class TestElewiseDequantOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        input_y = in_tensors[0].cpu().numpy()
        input_scale = in_tensors[1].cpu().numpy()
        input_offset = in_tensors[2].cpu().numpy()
        if len(input_offset) == 0:
            out = np.clip(input_y.astype(np.float16) * input_scale, -65504, 65504)
        else:
            out = np.clip((input_y.astype(np.float16) - input_offset.astype(np.float16)) * input_scale, -65504, 65504)
        return [torch.from_numpy(out).to(torch.float16)]

    def test_2d_half_10_20_32_32(self):
        y = torch.randint(-128, 127, size=(10, 20, 32, 32), dtype=torch.int8).npu()

        scale = torch.rand(32, 32, dtype=torch.float16).npu()
        scale = torch.clamp(scale, 0.1, 1)

        offset = torch.randint(-128, 127, size=(32, 32), dtype=torch.int8).npu()
        if operation_test.get_soc_version() == 'Ascend910B':
            self.execute(OP_NAME, {"elewiseType": 18}, [y, scale, offset])

    def test_2d_half_10_8192(self):
        y = torch.randint(-128, 127, size=(10, 8192), dtype=torch.int8).npu()

        scale = torch.rand(8192, dtype=torch.float16).npu()
        scale = torch.clamp(scale, 0.1, 1)

        offset = torch.randint(-128, 127, size=(8192,), dtype=torch.int8).npu()
        if operation_test.get_soc_version() == 'Ascend910B':
            self.execute(OP_NAME, {"elewiseType": 18}, [y, scale, offset])

    def test_2d_half_10_8192_offset_not_exist(self):
        y = torch.randint(-128, 127, size=(10, 8192), dtype=torch.int8).npu()

        scale = torch.rand(8192, dtype=torch.float16).npu()
        scale = torch.clamp(scale, 0.1, 1)

        offset = torch.Tensor([]).npu()
        offset = offset.to(torch.int8)
        if operation_test.get_soc_version() == 'Ascend910B':
            self.execute(OP_NAME, {"elewiseType": 18}, [y, scale, offset])

    def test_2d_half_10_10_1024_1024(self):
        y = torch.randint(-128, 127, size=(10, 10, 1024, 1024), dtype=torch.int8).npu()

        scale = torch.rand(1024, 1024, dtype=torch.float16).npu()
        scale = torch.clamp(scale, 0.1, 1)

        offset = torch.randint(-128, 127, size=(1024, 1024), dtype=torch.int8).npu()
        if operation_test.get_soc_version() == 'Ascend910B':
            self.execute(OP_NAME, {"elewiseType": 18}, [y, scale, offset])
    
    def test_2d_half_10_10_1024_1024_offset_not_exist(self):
        y = torch.randint(-128, 127, size=(10, 10, 1024, 1024), dtype=torch.int8).npu()

        scale = torch.rand(1024, 1024, dtype=torch.float16).npu()
        scale = torch.clamp(scale, 0.1, 1)

        offset = torch.Tensor([]).npu()
        offset = offset.to(torch.int8)
        if operation_test.get_soc_version() == 'Ascend910B':
            self.execute(OP_NAME, {"elewiseType": 18}, [y, scale, offset])

    def test_2d_half_1_1024_1024(self):
        y = torch.randint(-128, 127, size=(1, 1024, 1024), dtype=torch.int8).npu()

        scale = torch.rand(1024, 1024, dtype=torch.float16).npu()
        scale = torch.clamp(scale, 0.1, 1)

        offset = torch.randint(-128, 127, size=(1024, 1024), dtype=torch.int8).npu()
        if operation_test.get_soc_version() == 'Ascend910B':
            self.execute(OP_NAME, {"elewiseType": 18}, [y, scale, offset])

    def test_2d_half_1_1024_1024_offset_not_exist(self):
        y = torch.randint(-128, 127, size=(1, 1024, 1024), dtype=torch.int8).npu()

        scale = torch.rand(1024, 1024, dtype=torch.float16).npu()
        scale = torch.clamp(scale, 0.1, 1)

        offset = torch.Tensor([]).npu()
        offset = offset.to(torch.int8)
        if operation_test.get_soc_version() == 'Ascend910B':
            self.execute(OP_NAME, {"elewiseType": 18}, [y, scale, offset])

    def test_2d_half_1024_1024_scalar(self):
        y = torch.randint(-128, 127, size=(1024, 1024), dtype=torch.int8).npu()

        scale = torch.rand(1, dtype=torch.float16).npu()
        scale = torch.clamp(scale, 0.1, 1)

        offset = torch.randint(-128, 127, size=(1,), dtype=torch.int8).npu()
        if operation_test.get_soc_version() == 'Ascend910B':
            self.execute(OP_NAME, {"elewiseType": 18}, [y, scale, offset])

    def test_2d_half_1024_1024_scalar_offset_not_exist(self):
        y = torch.randint(-128, 127, size=(1024, 1024), dtype=torch.int8).npu()

        scale = torch.rand(1, dtype=torch.float16).npu()
        scale = torch.clamp(scale, 0.1, 1)

        offset = torch.Tensor([]).npu()
        offset = offset.to(torch.int8)
        if operation_test.get_soc_version() == 'Ascend910B':
            self.execute(OP_NAME, {"elewiseType": 18}, [y, scale, offset])

    def golden_compare(self, out_tensor, golden_out_tensor):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        result = torch.allclose(out_tensor.cpu(), golden_out_tensor.cpu(), atol=0.001, rtol=0.001)
        if not result:
            print("out_tensor.shape", out_tensor.shape,
                  "\ngolden_out_tensor.shape:", golden_out_tensor.shape)
            print("out_tensor:", out_tensor,
                  ", \ngolden_oute_tensor:", golden_out_tensor)
        return result

if __name__ == '__main__':
    unittest.main()
