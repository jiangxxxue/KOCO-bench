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


class TestElewiseQuantOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        input_x = in_tensors[0].type(torch.float16).cpu().numpy()
        input_scale = in_tensors[1].type(torch.float16).cpu().numpy()
        input_offset = in_tensors[2].cpu().numpy()
        if len(input_offset) == 0:
            out = np.clip((np.round((input_x / input_scale))), -128, 127)
        else:
            out = np.clip((np.round((input_x / input_scale)) + input_offset), -128, 127)
        return [torch.from_numpy(out).to(torch.int8)]

    def test_2d_half_10_20_32_32(self):
        x = torch.rand(10, 20, 32, 32, dtype=torch.float16).npu()
        x = torch.clamp(x, -1000, 1000)

        scale = torch.rand(32, 32, dtype=torch.float16).npu()
        scale = torch.clamp(scale, 0.1, 1)

        offset = torch.randint(-10, 10, size=(32, 32), dtype=torch.int8).npu()
        self.execute(OP_NAME, {"elewiseType": 17}, [x, scale, offset])

    def test_2d_half_10_8192(self):
        x = torch.rand(10, 8192, dtype=torch.float16).npu()
        x = torch.clamp(x, -1000, 1000)

        scale = torch.rand(8192, dtype=torch.float16).npu()
        scale = torch.clamp(scale, 0.1, 1)

        offset = torch.randint(-10, 10, size=(8192,), dtype=torch.int8).npu()
        self.execute(OP_NAME, {"elewiseType": 17}, [x, scale, offset])

    def test_2d_half_10_8192_offset_not_exist(self):
        x = torch.rand(10, 8192, dtype=torch.float16).npu()
        x = torch.clamp(x, -1000, 1000)

        scale = torch.rand(8192, dtype=torch.float16).npu()
        scale = torch.clamp(scale, 0.1, 1)

        offset = torch.Tensor([]).npu()
        offset = offset.to(torch.int8)
        self.execute(OP_NAME, {"elewiseType": 17}, [x, scale, offset])

    def test_2d_half_10_10_1024_1024(self):
        x = torch.rand(10, 10, 1024, 1024, dtype=torch.float16).npu()
        x = torch.clamp(x, -1000, 1000)

        scale = torch.rand(1024, 1024, dtype=torch.float16).npu()
        scale = torch.clamp(scale, 0.1, 1)

        offset = torch.randint(-10, 10, size=(1024, 1024), dtype=torch.int8).npu()
        self.execute(OP_NAME, {"elewiseType": 17}, [x, scale, offset])

    def test_2d_half_10_10_1024_1024_offset_not_exist(self):
        x = torch.rand(10, 10, 1024, 1024, dtype=torch.float16).npu()
        x = torch.clamp(x, -1000, 1000)

        scale = torch.rand(1024, 1024, dtype=torch.float16).npu()
        scale = torch.clamp(scale, 0.1, 1)

        offset = torch.Tensor([]).npu()
        offset = offset.to(torch.int8)
        self.execute(OP_NAME, {"elewiseType": 17}, [x, scale, offset])

    def test_2d_half_1_1024_1024(self):
        x = torch.rand(1, 1024, 1024, dtype=torch.float16).npu()
        x = torch.clamp(x, -1000, 1000)

        scale = torch.rand(1024, 1024, dtype=torch.float16).npu()
        scale = torch.clamp(scale, 0.1, 1)

        offset = torch.randint(-10, 10, size=(1024, 1024), dtype=torch.int8).npu()
        self.execute(OP_NAME, {"elewiseType": 17}, [x, scale, offset])

    def test_2d_half_1_1024_1024_offset_not_exist(self):
        x = torch.rand(1, 1024, 1024, dtype=torch.float16).npu()
        x = torch.clamp(x, -1000, 1000)

        scale = torch.rand(1024, 1024, dtype=torch.float16).npu()
        scale = torch.clamp(scale, 0.1, 1)

        offset = torch.Tensor([]).npu()
        offset = offset.to(torch.int8)
        self.execute(OP_NAME, {"elewiseType": 17}, [x, scale, offset])

    def test_2d_half_1024_1024_scalar(self):
        x = torch.rand(1024, 1024, dtype=torch.float16).npu()
        x = torch.clamp(x, -1000, 1000)

        scale = torch.rand(1, dtype=torch.float16).npu()
        scale = torch.clamp(scale, 0.1, 1)

        offset = torch.randint(-10, 10, size=(1,), dtype=torch.int8).npu()
        self.execute(OP_NAME, {"elewiseType": 17}, [x, scale, offset])

    def test_2d_half_1024_1024_scalar_offset_not_exist(self):
        x = torch.rand(1024, 1024, dtype=torch.float16).npu()
        x = torch.clamp(x, -1000, 1000)

        scale = torch.rand(1, dtype=torch.float16).npu()
        scale = torch.clamp(scale, 0.1, 1)

        offset = torch.Tensor([]).npu()
        offset = offset.to(torch.int8)
        self.execute(OP_NAME, {"elewiseType": 17}, [x, scale, offset])

    def test_2d_half_1024_1024_scalar_bf16(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        x = torch.rand(1024, 1024, dtype=torch.bfloat16).npu()
        x = torch.clamp(x, -1000, 1000)

        scale = torch.rand(1, dtype=torch.bfloat16).npu()
        scale = torch.clamp(scale, 0.1, 1)

        offset = torch.randint(-10, 10, size=(1,), dtype=torch.int8).npu()
        self.execute(OP_NAME, {"elewiseType": 17}, [x, scale, offset])

    def test_2d_half_10_8192_bf16(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        x = torch.rand(10, 8192, dtype=torch.bfloat16).npu()
        x = torch.clamp(x, -1000, 1000)

        scale = torch.rand(8192, dtype=torch.bfloat16).npu()
        scale = torch.clamp(scale, 0.1, 1)

        offset = torch.randint(-10, 10, size=(8192,), dtype=torch.int8).npu()
        self.execute(OP_NAME, {"elewiseType": 17}, [x, scale, offset])
    
    def golden_compare(self, out_tensor, golden_out_tensor):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        result = torch.allclose(out_tensor.cpu(), golden_out_tensor.cpu(), atol=1)
        if not result:
            print("out_tensor.shape", out_tensor.shape,
                  "\ngolden_out_tensor.shape:", golden_out_tensor.shape)
            print("out_tensor:", out_tensor,
                  ", \ngolden_out_tensor:", golden_out_tensor)
        return result

if __name__ == '__main__':
    unittest.main()
