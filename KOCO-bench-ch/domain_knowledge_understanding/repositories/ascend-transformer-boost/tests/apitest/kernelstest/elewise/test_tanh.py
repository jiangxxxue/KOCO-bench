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
import torch.nn.functional as F
import sys
import op_test
import logging


OP_NAME = "ElewiseOperation"
OP_PARAM_TANH = {"elewiseType": 16}

class TestElewise(op_test.OpTest):
    def golden_calc(self, in_tensors):
        actvation_type = self.op_desc["specificParam"]["elewiseType"]
        if actvation_type == 16:
            return [torch.tanh(in_tensors[0].float())]
        else:
            return [torch.zeros_like(x) for x in in_tensors]

    def golden_compare(self, out_tensors, golden_out_tensors):
        err = 2**(-8)
        if out_tensors[0].dtype == torch.bfloat16:
            err = 2**(-7)
        elif out_tensors[0].dtype == torch.float32:
            err = 2**(-11)
        out_tensors[0] = out_tensors[0].to(torch.float32)
        golden_out_tensors[0] = golden_out_tensors[0].to(torch.float32)
        diff = torch.abs(torch.subtract(out_tensors[0], golden_out_tensors[0]))
        tensor_max = torch.maximum(torch.ones(golden_out_tensors[0].shape, dtype=golden_out_tensors[0].dtype), torch.abs(golden_out_tensors[0]))
        if torch.any(torch.greater(diff, err * tensor_max)):
            logging.info("new golden result: failed")
            return False
        logging.info("new golden result: true")
        return True

    def test_tanh_bf16(self):
        self.set_support_910b_only()
        shape = (1000000)
        input0 = np.random.uniform(low=-100, high=100, size=shape).astype(np.float32)

        self.set_param(OP_NAME, OP_PARAM_TANH)
        self.execute([torch.from_numpy(input0).bfloat16()],
                     [torch.zeros(shape).bfloat16()])

if __name__ == '__main__':
    unittest.main()
