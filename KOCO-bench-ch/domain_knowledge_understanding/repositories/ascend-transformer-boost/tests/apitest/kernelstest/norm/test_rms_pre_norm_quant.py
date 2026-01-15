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
import torch
import op_test

RMS_NORM = 2
OP_NAME = "NormOperation"
OP_PARAM0 = {"normType": RMS_NORM,
             "epsilon": 1e-5,
             "inGamma": True,
             "inNormBias": True,
             "inRes": True,
             "outRes": True}

HALF_FLOAT_MIN = -5.0
HALF_FLOAT_MAX = 5.0
QUANTMIN = -128
QUANTMAX = 127

class TestRmsPreNormQuant(op_test.OpTest):
    def rms_norm(self, x, weight, bias, eps=1e-5):
        square_sum = torch.sum(torch.square(x), axis=-1, keepdims=True)
        factor = 1.0 / torch.sqrt(square_sum / x.shape[-1] + eps)
        return x * factor * weight + bias

    def golden_calc(self, in_tensors):
        eps = self.op_desc["specificParam"]["epsilon"]
        inputX = in_tensors[0].float()
        inputResIn = in_tensors[1].float()
        inputGamma = in_tensors[2].float()
        inputBeta = in_tensors[3].float()
        inputScale = in_tensors[4].item()
        inputOffset = in_tensors[5].item()
        sumX = inputX + inputResIn
        output = self.rms_norm(sumX,
                               weight=inputGamma,
                               bias=inputBeta,
                               eps=eps)
        outputQ = output / inputScale + inputOffset
        return [outputQ.clamp(QUANTMIN, QUANTMAX).round().to(torch.int8),
                sumX.half()]

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
    def test_rms_pre_norm_quant_f16_case0(self):
        '''
            基础场景
            shape: (20, 1, 5120) / (1, 5120)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        a, b = 20, 5120
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputResIn = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputBeta = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputScale = torch.empty((1,), dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputOffset = torch.randint(-5, 5, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputResIn, inputGamma, inputBeta, inputScale, inputOffset],
                     [torch.zeros(shape0, dtype=torch.int8), torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    def test_rms_pre_norm_quant_f16_case1(self):
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
        inputResIn = torch.empty(shape0).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputGamma = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputBeta = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputScale = torch.empty((1,)).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputOffset = torch.randint(-5, 5, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputResIn, inputGamma, inputBeta, inputScale, inputOffset],
                     [torch.zeros(shape0, dtype=torch.int8), torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    def test_rms_pre_norm_quant_f16_case8192(self):
        '''
            基础场景
            shape: (20, 1, 8192) / (1, 8192)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        a, b = 20, 8192
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputResIn = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputBeta = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputScale = torch.empty((1,), dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputOffset = torch.randint(-5, 5, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputResIn, inputGamma, inputBeta, inputScale, inputOffset],
                     [torch.zeros(shape0, dtype=torch.int8), torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    def test_rms_pre_norm_quant_f16_case_last_dim_12288(self):
        '''
            基础场景
            shape: (128, 1, 12288) / (1, 12288)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        a, b = 128, 12288
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputResIn = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputBeta = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputScale = torch.empty((1,), dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputOffset = torch.randint(-5, 5, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputResIn, inputGamma, inputBeta, inputScale, inputOffset],
                     [torch.zeros(shape0, dtype=torch.int8), torch.zeros(shape0, dtype=torch.float16)])

if __name__ == '__main__':
    unittest.main()
