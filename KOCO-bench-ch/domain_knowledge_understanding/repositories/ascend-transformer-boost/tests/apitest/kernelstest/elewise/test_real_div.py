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
from elewise_type import ElewiseType

OP_NAME = "ElewiseOperation"
OP_PARAM_DIV = {"elewiseType": ElewiseType.ELEWISE_REALDIV.value}


def get_eb(out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor) -> float:
    diff = torch.sub(out_tensor, golden_out_tensor)
    golden_nmax = torch.clamp(torch.abs(golden_out_tensor), min=1)
    return torch.mean(torch.div(diff, golden_nmax)).item()


class TestElewiseOperationRealDiv(op_test.OpTest):
    def calculate_times(self, in_tensors: list[torch.Tensor]) -> int:
        return in_tensors[0].numel()

    def golden_calc(self, in_tensors: list[torch.Tensor]):
        x = in_tensors[0].float()
        y = in_tensors[1].float()
        z = torch.div(x, y)
        return [z]

    def golden_compare(self, out_tensors, golden_out_tensors):
        # compute high precision single golden
        dtype = out_tensors[0].dtype
        ERR_EB = {
            torch.float16: (-11, -10),
            torch.bfloat16: (-8, -7),
            torch.float32: (-14, -14)
        }
        calculate_times = self.calculate_times(out_tensors)
        err_base, eb_base = ERR_EB[dtype]
        if calculate_times >= 2048:
            err_base += 1
        if calculate_times >= 16384 and dtype == torch.float32:
            err_base += 1
        err = 2**err_base
        eb = 2**eb_base
        out_tensors[0] = out_tensors[0].to(torch.float32)
        golden_out_tensors[0] = golden_out_tensors[0].to(torch.float32)
        diff = torch.abs(torch.subtract(out_tensors[0], golden_out_tensors[0]))
        tensor_max = torch.maximum(torch.ones(golden_out_tensors[0].shape, dtype=golden_out_tensors[0].dtype),
                                   torch.abs(golden_out_tensors[0]))
        if torch.any(torch.greater(diff, err * tensor_max)):
            return False
        if get_eb(out_tensors[0], golden_out_tensors[0]) > eb:
            return False
        return True

    @op_test.only_910b
    def test_div_bf16(self):
        shape = (1000000)
        input0 = torch.from_numpy(np.random.uniform(
            low=0, high=100, size=shape).astype(np.float32))
        input1 = torch.from_numpy(np.random.uniform(
            low=0, high=100, size=shape).astype(np.float32))
        input1 = torch.where(input1 == 0, torch.tensor(2**(-4)), input1)

        self.set_param(OP_NAME, OP_PARAM_DIV)
        self.execute([input0.bfloat16(), input1.bfloat16()],
                     [torch.zeros(shape).bfloat16()])


if __name__ == '__main__':
    unittest.main()
