#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import os
import sys
import unittest

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

OP_NAME = "ScatterElementsV2Operation"
PARAM = {"axis": -1, "reduction": 1}


class TestScatterElementsV2AddOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        input_tensor = in_tensors[0]
        indice_tensor = in_tensors[1]
        update_tensor = in_tensors[2]

        axis = PARAM["axis"]
        reduction = "add" if "reduction" in PARAM and PARAM[ "reduction"] == 1 else None

        if reduction:
            output = input_tensor.scatter_(axis, indice_tensor.long(), update_tensor, reduce=reduction)
        else:
            output = input_tensor.scatter_(axis, indice_tensor.long(), update_tensor)

        return [output]

    def golden_compare(self, out_tensor, golden_out_tensor, rtol=0.02, atol=0.02):
        result = torch.allclose(out_tensor.cpu(), golden_out_tensor.cpu(), rtol=rtol, atol=atol)
        return result


    def test_basic_scatter_v2_uint8_int64_add(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        input_tensor = torch.ones(10, 10, dtype=torch.uint8)
        indice_tensor = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        update_tensor = torch.tensor([[1, 2], [3, 4]], dtype=input_tensor.dtype)

         # 原地写不会创建新的tensor
        self.execute_inplace_v2(OP_NAME, PARAM, [input_tensor.npu(), indice_tensor.npu(), update_tensor.npu()], [0])

    def test_basic_scatter_v2_int32_int32_add(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        input_tensor = torch.ones(10, 10, dtype=torch.int32)
        indice_tensor = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        update_tensor = torch.tensor([[1, 2], [3, 4]], dtype=input_tensor.dtype)
         # 原地写不会创建新的tensor # 原地写不会创建新的tensor
        self.execute_inplace_v2(OP_NAME, PARAM, [input_tensor.npu(), indice_tensor.npu(), update_tensor.npu()], [0])


    def test_basic_scatter_v2_int32_int64_add(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        input_tensor = torch.ones(10, 10, dtype=torch.int32)
        indice_tensor = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        update_tensor = torch.tensor([[1, 2], [3, 4]], dtype=input_tensor.dtype)
         # 原地写不会创建新的tensor
        self.execute_inplace_v2(OP_NAME, PARAM, [input_tensor.npu(), indice_tensor.npu(), update_tensor.npu()], [0])

    def test_basic_scatter_v2_uint8_int32_add(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        input_tensor = torch.ones(10, 10, dtype=torch.uint8)
        indice_tensor = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        update_tensor = torch.tensor([[1, 2], [3, 4]], dtype=input_tensor.dtype)
         # 原地写不会创建新的tensor
        self.execute_inplace_v2(OP_NAME, PARAM, [input_tensor.npu(), indice_tensor.npu(), update_tensor.npu()], [0])

    def test_basic_scatter_v2_uint8_int64_add(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        input_tensor = torch.ones(10, 10, dtype=torch.uint8)
        indice_tensor = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        update_tensor = torch.tensor([[1, 2], [3, 4]], dtype=input_tensor.dtype)

         # 原地写不会创建新的tensor
        self.execute_inplace_v2(OP_NAME, PARAM, [input_tensor.npu(), indice_tensor.npu(), update_tensor.npu()], [0])


    def test_basic_scatter_v2_int8_int32_add(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        input_tensor = torch.ones(10, 10, dtype=torch.int8)
        indice_tensor = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        update_tensor = torch.tensor([[1, 2], [3, 4]], dtype=input_tensor.dtype)
         # 原地写不会创建新的tensor
        self.execute_inplace_v2(OP_NAME, PARAM, [input_tensor.npu(), indice_tensor.npu(), update_tensor.npu()], [0])


    def test_basic_scatter_v2_int8_int64_add(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        input_tensor = torch.ones(10, 10, dtype=torch.int8)
        indice_tensor = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        update_tensor = torch.tensor([[1, 2], [3, 4]], dtype=input_tensor.dtype)
         # 原地写不会创建新的tensor
        self.execute_inplace_v2(OP_NAME, PARAM, [input_tensor.npu(), indice_tensor.npu(), update_tensor.npu()], [0])

    def test_basic_scatter_v2_float16_int32_add(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        input_tensor = torch.ones(10, 10, dtype=torch.float16)
        indice_tensor = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        update_tensor = torch.tensor([[1, 2], [3, 4]], dtype=input_tensor.dtype)
         # 原地写不会创建新的tensor
        self.execute_inplace_v2(OP_NAME, PARAM, [input_tensor.npu(), indice_tensor.npu(), update_tensor.npu()], [0])

    def test_basic_scatter_v2_float16_int64_add(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        input_tensor = torch.ones(10, 10, dtype=torch.float16)
        indice_tensor = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        update_tensor = torch.tensor([[1, 2], [3, 4]], dtype=input_tensor.dtype)
         # 原地写不会创建新的tensor
        self.execute_inplace_v2(OP_NAME, PARAM, [input_tensor.npu(), indice_tensor.npu(), update_tensor.npu()], [0])

    def test_basic_scatter_v2_float32_int32_add(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        input_tensor = torch.ones(10, 10, dtype=torch.float32)
        indice_tensor = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        update_tensor = torch.tensor([[1, 2], [3, 4]], dtype=input_tensor.dtype)
         # 原地写不会创建新的tensor
        self.execute_inplace_v2(OP_NAME, PARAM, [input_tensor.npu(), indice_tensor.npu(), update_tensor.npu()], [0])


    def test_basic_scatter_v2_float32_int64_add(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        input_tensor = torch.ones(10, 10, dtype=torch.float32)
        indice_tensor = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        update_tensor = torch.tensor([[1, 2], [3, 4]], dtype=input_tensor.dtype)
         # 原地写不会创建新的tensor
        self.execute_inplace_v2(OP_NAME, PARAM, [input_tensor.npu(), indice_tensor.npu(), update_tensor.npu()], [0])

    def test_basic_scatter_v2_bfloat16_int32_add(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        input_tensor = torch.ones(10, 10, dtype=torch.bfloat16)
        indice_tensor = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        update_tensor = torch.tensor([[1, 2], [3, 4]], dtype=input_tensor.dtype)
         # 原地写不会创建新的tensor
        self.execute_inplace_v2(OP_NAME, PARAM, [input_tensor.npu(), indice_tensor.npu(), update_tensor.npu()], [0])

    def test_basic_scatter_v2_bfloat16_int64_add(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        input_tensor = torch.ones(10, 10, dtype=torch.bfloat16)
        indice_tensor = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        update_tensor = torch.tensor([[1, 2], [3, 4]], dtype=input_tensor.dtype)
         # 原地写不会创建新的tensor
        self.execute_inplace_v2(OP_NAME, PARAM, [input_tensor.npu(), indice_tensor.npu(), update_tensor.npu()], [0])



if __name__ == '__main__':
    unittest.main()
