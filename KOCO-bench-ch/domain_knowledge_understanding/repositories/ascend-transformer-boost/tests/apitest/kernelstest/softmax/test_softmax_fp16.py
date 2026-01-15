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
import sys
import unittest

sys.path.append("../")
import op_test
import torch
import torch.nn.functional as F

OP_NAME = "SoftmaxOperation"


class SoftmaxTest(op_test.OpTest):
    def golden_calc(self, in_tensors):
        axes = self.op_desc["specificParam"]["axes"]
        for i in range(len(axes)):
            if axes[i] < 0:
                axes[i] = axes[i] % (len(axes))+1
        input = in_tensors[0].to(torch.float32)
        ori_shape = input.shape[axes[0]:axes[-1]+1]
        input_flatten = input.flatten(start_dim=axes[0], end_dim=axes[-1])
        input_tr = input_flatten.transpose(0, axes[0])
        res = F.softmax(input_tr, 0)
        res_tr = res.transpose(0, axes[0])
        output_unflatten = res_tr.unflatten(axes[0], ori_shape)
        return [output_unflatten]

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

    def test_softmax_fp16_case0(self):
        # dim > 0
        self.set_param(OP_NAME, {"axes": [1, 2]})
        self.execute([torch.rand((3, 3, 4)).to(torch.float16)], [torch.zeros((3, 3, 4)).to(torch.float16)])

    def test_softmax_fp16_case1(self):
        # dim < 0
        self.set_param(OP_NAME, {"axes": [-2, -1]})
        self.execute([torch.rand((3, 3, 4)).half()], [torch.zeros((3, 3, 4)).half()])

if __name__ == "__main__":
    unittest.main()