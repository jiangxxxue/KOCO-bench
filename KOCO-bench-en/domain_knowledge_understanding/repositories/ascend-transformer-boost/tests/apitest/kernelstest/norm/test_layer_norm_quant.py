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


OP_NAME = "NormOperation"
OP_PARAM0 = {"normType": 1, "epsilon": 1e-6,
             "inGamma": True, "inBeta": True, "inRes": False, "outMean": False, "outVarience": False, "outResQuant": True}

HALF_FLOAT_MIN = -2.0
HALF_FLOAT_MAX = 2.0
QUANTMIN = -128
QUANTMAX = 127


class TestLayerNormQuant(op_test.OpTest):
    def golden_calc(self, in_tensors):
        eps = self.op_desc["specificParam"]["epsilon"]
        scale = in_tensors[3].item()
        offset = in_tensors[4].item()
        output = F.layer_norm(in_tensors[0].float(),
                              in_tensors[1].shape,
                              weight=in_tensors[1].float(),
                              bias=in_tensors[2].float(),
                              eps=eps)
        output = output / scale + offset
        return [output.clamp(QUANTMIN, QUANTMAX).round().to(torch.int8)]

    def golden_compare(self, out_tensors, golden_out_tensors):
        expect0 = golden_out_tensors[0]
        actual0 = out_tensors[0]
        diff_count = torch.sum(expect0 != actual0).item()
        
        abs_diff = torch.abs(expect0-actual0)

        return not (abs_diff > 1).nonzero().size(0) > 0

    @op_test.skip_310b
    @op_test.skip_910a
    def test_layer_norm_quant_f16_case0(self):
        '''
            基础场景
            shape: (4096, 1, 25760) / (1, 25760)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        a, b = 4096, 25760
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputBeta = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputScale = torch.empty((1,), dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputOffset = torch.randint(-128, 128, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma, inputBeta, inputScale, inputOffset],
                     [torch.zeros(shape0, dtype=torch.int8)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_layer_norm_quant_f16_case1(self):
        '''
            基础场景
            shape: (1, 1, 10240) / (1, 10240)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        a, b = 1, 10240
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputGamma = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputBeta = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputScale = torch.empty((1,)).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputOffset = torch.randint(-128, 128, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma, inputBeta, inputScale, inputOffset],
                     [torch.zeros(shape0, dtype=torch.int8)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_layer_norm_quant_f16_case2(self):
        '''
            基础场景
            shape: (4096, 1, 128) / (1, 128)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        a, b = 4096, 128
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputGamma = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputBeta = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputScale = torch.empty((1,)).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputOffset = torch.randint(-128, 128, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma, inputBeta, inputScale, inputOffset],
                     [torch.zeros(shape0, dtype=torch.int8)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_layer_norm_quant_f16_case3(self):
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
        inputScale = torch.empty((1,)).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputOffset = torch.randint(-128, 128, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma, inputBeta, inputScale, inputOffset],
                     [torch.zeros(shape0, dtype=torch.int8)])

    @op_test.only_910b
    def test_layer_norm_quant_bf16_case0(self):
        '''
            基础场景
            shape: (4096, 1, 25760) / (1, 25760)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        a, b = 4096, 25760
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputBeta = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputScale = torch.empty((1,), dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputOffset = torch.randint(-128, 128, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma, inputBeta, inputScale, inputOffset],
                     [torch.zeros(shape0, dtype=torch.int8)])
    
    @op_test.only_910b
    def test_layer_norm_quant_bf16_case1(self):
        '''
            基础场景
            shape: (1, 1, 10240) / (1, 10240)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        a, b = 1, 10240
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputBeta = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputScale = torch.empty((1,), dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputOffset = torch.randint(-128, 128, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma, inputBeta, inputScale, inputOffset],
                     [torch.zeros(shape0, dtype=torch.int8)])
    
    @op_test.only_910b
    def test_layer_norm_quant_bf16_case2(self):
        '''
            基础场景
            shape: (4096, 1, 128) / (4096, 128)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        a, b = 4096, 128
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputBeta = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputScale = torch.empty((1,), dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputOffset = torch.randint(-128, 128, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma, inputBeta, inputScale, inputOffset],
                     [torch.zeros(shape0, dtype=torch.int8)])

    @op_test.only_910b
    def test_layer_norm_quant_bf16_case3(self):
        '''
            基础场景
            shape: (1, 1, 128) / (1, 128)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        a, b = 1, 128
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.bfloat16)
        inputGamma = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.bfloat16)
        inputBeta = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.bfloat16)
        inputScale = torch.empty((1,)).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.bfloat16)
        inputOffset = torch.randint(-128, 128, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma, inputBeta, inputScale, inputOffset],
                     [torch.zeros(shape0, dtype=torch.int8)])

if __name__ == '__main__':
    unittest.main()
