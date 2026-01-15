#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import logging
import unittest
import op_test
import torch

OP_NAME = "ReduceOperation"

class ReduceSumBF16Test(op_test.OpTest):
    def golden_calc(self, in_tensors):
        axis = self.op_desc["specificParam"]["axis"]
        return [torch.sum(in_tensors[0], axis)]

    def golden_compare(self, out_tensors, golden_out_tensors):
        err = 2 ** (-7)
        out_tensors[0] = out_tensors[0].to(torch.float32)
        golden_out_tensors[0] = golden_out_tensors[0].to(torch.float32)
        diff = torch.abs(torch.subtract(out_tensors[0], golden_out_tensors[0]))
        tensor_max = torch.maximum(torch.ones(golden_out_tensors[0].shape, dtype=golden_out_tensors[0].dtype),
                                   torch.abs(golden_out_tensors[0]))
        if torch.any(torch.greater(diff, err * tensor_max)):
            return False
        return True

    @op_test.only_910b
    def test_reduce_sum_bf16_case0(self):
        self.set_param(OP_NAME, {"axis": [1], "reduceType": 3})
        self.execute([torch.rand((2, 3)).bfloat16()],
                     [torch.zeros((2)).bfloat16()])

if __name__ == '__main__':
    unittest.main()
