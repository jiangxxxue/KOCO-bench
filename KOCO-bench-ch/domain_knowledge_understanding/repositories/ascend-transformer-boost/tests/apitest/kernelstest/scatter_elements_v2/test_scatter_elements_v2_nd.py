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

import op_test
import torch

OP_NAME = "ScatterElementsV2Operation"


class TestScatterElements(op_test.OpTest):
    def golden_calc(self, in_tensors):
        specificParam = self.op_desc["specificParam"]
        axis = specificParam["axis"]
        reduction = "add" if "reduction" in specificParam and specificParam[ "reduction"] == 1 else None

        input_tensor = in_tensors[0]
        indices = in_tensors[1]
        updates = in_tensors[2]
        # 使用 PyTorch 的 scatter 函数计算预期结果
        if reduction:
            output = input_tensor.scatter_(axis, indices.long(), updates, reduce=reduction)
        else:
            output = input_tensor.scatter_(axis, indices.long(), updates)

        self.assertTrue(id(input_tensor) == id(output))
        return [output]

    def golden_compare(self, out_tensors, golden_out_tensors):
        return torch.equal(out_tensors[0], golden_out_tensors[0])

    @op_test.only_910b
    def test_basic_scatter_v2_int32_int32_add_3d(self):
        input_tensor = torch.ones(4, 4, 4, dtype=torch.int32)
        indice_tensor = torch.ones(2, 4, 2, dtype=torch.int64)
        update_tensor = torch.ones(2, 4, 2, dtype=torch.int32)

        op_para = {"axis": -1, "reduction": 1}
        self.set_param(OP_NAME, op_para)
        output_tensor = [0]  # 原地写不会创建新的tensor # 原地写不会创建新的tensor
        self.execute([input_tensor, indice_tensor, update_tensor], output_tensor)

    @op_test.only_910b
    def test_basic_scatter_v2_int32_int32_add_4d(self):
        input_tensor = torch.ones(4, 4, 4, 4, dtype=torch.int32)
        indice_tensor = torch.ones(2, 4, 4, 2, dtype=torch.int64)
        update_tensor = torch.ones(2, 4, 4, 2, dtype=torch.int32)

        op_para = {"axis": -1, "reduction": 1}
        self.set_param(OP_NAME, op_para)
        output_tensor = [0]  # 原地写不会创建新的tensor # 原地写不会创建新的tensor
        self.execute([input_tensor, indice_tensor, update_tensor], output_tensor)

    @op_test.only_910b
    def test_basic_scatter_v2_int32_int32_add_5d(self):
        input_tensor = torch.ones(4, 4, 4, 4, 4, dtype=torch.int32)
        indice_tensor = torch.ones(2, 4, 4, 4, 2, dtype=torch.int64)
        update_tensor = torch.ones(2, 4, 4, 4, 2, dtype=torch.int32)

        op_para = {"axis": -1, "reduction": 1}
        self.set_param(OP_NAME, op_para)
        output_tensor = [0]  # 原地写不会创建新的tensor # 原地写不会创建新的tensor
        self.execute([input_tensor, indice_tensor, update_tensor], output_tensor)

    @op_test.only_910b
    def test_basic_scatter_v2_int32_int32_add_6d(self):
        input_tensor = torch.ones(4, 4, 4, 4, 4, 4, dtype=torch.int32)
        indice_tensor = torch.ones(2, 4, 4, 4, 4, 2, dtype=torch.int64)
        update_tensor = torch.ones(2, 4, 4, 4, 4, 2, dtype=torch.int32)

        op_para = {"axis": -1, "reduction": 1}
        self.set_param(OP_NAME, op_para)
        output_tensor = [0]  # 原地写不会创建新的tensor # 原地写不会创建新的tensor
        self.execute([input_tensor, indice_tensor, update_tensor], output_tensor)

    @op_test.only_910b
    def test_basic_scatter_v2_int32_int32_add_7d(self):
        input_tensor = torch.ones(4, 4, 4, 4, 4, 4, 4, dtype=torch.int32)
        indice_tensor = torch.ones(2, 4, 4, 4, 4, 4, 2, dtype=torch.int64)
        update_tensor = torch.ones(2, 4, 4, 4, 4, 4, 2, dtype=torch.int32)

        op_para = {"axis": -1, "reduction": 1}
        self.set_param(OP_NAME, op_para)
        output_tensor = [0]  # 原地写不会创建新的tensor # 原地写不会创建新的tensor
        self.execute([input_tensor, indice_tensor, update_tensor], output_tensor)

    @op_test.only_910b
    def test_basic_scatter_v2_int32_int32_add_8d(self):
        input_tensor = torch.ones(4, 4, 4, 4, 4, 4, 4, 4, dtype=torch.int32)
        indice_tensor = torch.ones(2, 4, 4, 4, 4, 4, 4, 2, dtype=torch.int64)
        update_tensor = torch.ones(2, 4, 4, 4, 4, 4, 4, 2, dtype=torch.int32)

        op_para = {"axis": -1, "reduction": 1}
        self.set_param(OP_NAME, op_para)
        output_tensor = [0]  # 原地写不会创建新的tensor # 原地写不会创建新的tensor
        self.execute([input_tensor, indice_tensor, update_tensor], output_tensor)


if __name__ == '__main__':
    unittest.main()
