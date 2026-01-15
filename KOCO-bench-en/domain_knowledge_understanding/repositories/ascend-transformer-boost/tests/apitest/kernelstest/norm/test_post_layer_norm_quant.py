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
import torch.nn as nn
import torch.nn.functional as F
import op_test
from tensor_file import read_tensor

LAYER_NORM = 1
OP_NAME = "NormOperation"
OP_PARAM0 = {"normType": LAYER_NORM,
             "epsilon": 1e-5,
             "inGamma": True,
             "inBeta": True,
             "inRes": True,
             "outResQuant": True}

HALF_FLOAT_MIN = -5.0
HALF_FLOAT_MAX = 5.0
QUANTMIN = -128
QUANTMAX = 127


class TestPostLayerNormQuant(op_test.OpTest):
    def golden_calc(self, in_tensors):
        eps = self.op_desc["specificParam"]["epsilon"]
        inputX = in_tensors[0].float()
        inputGamma = in_tensors[1].float()
        inputBeta = in_tensors[2].float()
        inputResIn = in_tensors[3].float()
        inputScale = in_tensors[4].item()
        inputOffset = in_tensors[5].item()
        sumX = inputX + inputResIn
        output = F.layer_norm(sumX,
                              inputGamma.shape,
                              weight=inputGamma,
                              bias=inputBeta,
                              eps=eps)
        outputQ = output / inputScale + inputOffset
        return [outputQ.clamp(QUANTMIN, QUANTMAX).round().to(torch.int8),
                output.half()]

    def golden_compare(self, out_tensors, golden_out_tensors):
        diff = torch.abs(torch.subtract(out_tensors[0], golden_out_tensors[0]))
        if torch.any(torch.greater(diff, 1)):
            print("[new standards] output0 accuracy failed")
            return False

        diff = torch.abs(torch.subtract(out_tensors[1], golden_out_tensors[1]))
        tensor_max = torch.maximum(torch.ones(golden_out_tensors[1].shape, dtype=golden_out_tensors[1].dtype),
                                   torch.abs(golden_out_tensors[1]))

        if torch.any(torch.greater(diff, 2**(-8) * tensor_max)):
            print("[new standards] output1 accuracy failed")
            return False

        if torch.any(torch.greater(torch.abs(torch.mean(torch.div(diff, tensor_max))), 2**(-10))):
            print("[new standards] output1 eb failed")
            return False
        return True

    @op_test.skip_310b
    @op_test.skip_910a
    def test_post_layer_norm_quant_f16_case0(self):
        '''
            基础场景
            shape: (4096, 1, 10336) / (1, 10336)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        a, b = 4096, 10336
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputBeta = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputResIn = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputScale = torch.empty((1,), dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputOffset = torch.randint(-128, 128, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma, inputBeta, inputResIn, inputScale, inputOffset],
                     [torch.zeros(shape0, dtype=torch.int8), torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_post_layer_norm_quant_f16_case1(self):
        '''
            基础场景
            shape: (1, 1, 128) / (1, 128)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        a, b = 1, 128
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputGamma = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputBeta = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputResIn = torch.empty(shape0).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputScale = torch.empty((1,)).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputOffset = torch.randint(-128, 128, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma, inputBeta, inputResIn, inputScale, inputOffset],
                     [torch.zeros(shape0, dtype=torch.int8), torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_post_layer_norm_quant_f16_case3(self):
        '''
            基础场景
            shape: (2048, 1, 20672) / (1, 20672)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        a, b = 2048, 20672
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputBeta = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputResIn = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputScale = torch.empty((1,), dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputOffset = torch.randint(-128, 128, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma, inputBeta, inputResIn, inputScale, inputOffset],
                     [torch.zeros(shape0, dtype=torch.int8), torch.zeros(shape0, dtype=torch.float16)])

if __name__ == '__main__':
    unittest.main()
