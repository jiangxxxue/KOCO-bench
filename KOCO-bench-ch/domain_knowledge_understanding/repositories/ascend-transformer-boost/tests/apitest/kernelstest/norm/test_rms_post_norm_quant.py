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
             "inRes": True,
             "outRes": True}
OP_PARAM1 = {"normType": RMS_NORM,
             "epsilon": 1e-8,
             "inGamma": True,
             "inRes": True,
             "outRes": True}
 
HALF_FLOAT_MIN = -5.0
HALF_FLOAT_MAX = 5.0
QUANTMIN = -128
QUANTMAX = 127
 
class TestRmsPostNormQuant(op_test.OpTest):
    def rms_norm(self, x, weight, eps=1e-5):
        square_sum = torch.sum(torch.square(x), axis=-1, keepdims=True)
        factor = 1.0 / torch.sqrt(square_sum / x.shape[-1] + eps)
        return x * factor * weight
 
    def golden_calc(self, in_tensors):
        eps = self.op_desc["specificParam"]["epsilon"]
        inputX = in_tensors[0].float()
        orig_dtype = in_tensors[0].dtype
        inputResIn = in_tensors[1].float()
        inputGamma = in_tensors[2].float()
        scale = 1.0 / in_tensors[3].item()
        offset = in_tensors[4].item()
        inputScale = torch.tensor(scale, dtype=torch.float)
        inputOffset = torch.tensor(offset, dtype=torch.float)
        sumX = inputX + inputResIn
        output = self.rms_norm(sumX, weight=inputGamma, eps=eps)
        outputQ = output * inputScale + inputOffset
        outputQ = torch.round(outputQ).to(torch.half)
        outputQ = torch.max(outputQ, torch.tensor(QUANTMIN, dtype=torch.half))
        outputQ = torch.min(outputQ, torch.tensor(QUANTMAX, dtype=torch.half))
        return [outputQ.to(torch.int8), output.to(orig_dtype)]
 
    def golden_compare(self, out_tensors, golden_out_tensors):
        diff = torch.abs(torch.subtract(out_tensors[0], golden_out_tensors[0]))
        if torch.any(torch.greater(diff, 1)):
            print("[new standards] output0 accuracy failed")
            return False
 
        diff = torch.abs(torch.subtract(out_tensors[1].float(), golden_out_tensors[1].float()))
        tensor_max = torch.maximum(torch.ones(golden_out_tensors[1].shape, dtype=golden_out_tensors[1].dtype),
                                   torch.abs(golden_out_tensors[1])).float()
 
        factor1, factor2 = 2**(-8), 2**(-10)
        if golden_out_tensors[1].dtype == torch.bfloat16: 
            factor1, factor2 = 2**(-7), 2**(-7)
 
        if torch.any(torch.greater(diff, factor1 * tensor_max)):
            print("[new standards] output1 accuracy failed")
            return False
 
        if torch.any(torch.greater(torch.abs(torch.mean(torch.div(diff, tensor_max))), factor2)):
            print("[new standards] output1 eb failed")
            return False
 
        return True

    @op_test.skip_310b
    @op_test.skip_910a
    def test_rms_post_norm_quant_f16_small_shape(self):
        '''
            基础场景
            shape: (1, 1, 128) / (1, 128)
            取值范围
            fp16: [-5, 5]; int8: [-128, 128)
        '''
        a, b = 1, 128
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputResIn = torch.empty(shape0).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputGamma = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputScale = torch.empty((1,)).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        while inputScale == 0:
            inputScale = torch.empty((1,), dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputOffset = torch.randint(-5, 5, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputResIn, inputGamma, inputScale, inputOffset],
                    [torch.zeros(shape0, dtype=torch.int8), torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_rms_post_norm_quant_f16_model_shape(self):
        '''
            模型增量场景
            shape: (2500, 8192) / (1, 8192)
            取值范围
            fp16: [-5, 5]; int8: [-128, 128)
        '''
        a, b = 2500, 8192
        shape0 = (a, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputResIn = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputScale = torch.empty((1,), dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        while inputScale == 0:
            inputScale = torch.empty((1,), dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputOffset = torch.randint(-5, 5, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputResIn, inputGamma, inputScale, inputOffset],
                    [torch.zeros(shape0, dtype=torch.int8), torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_rms_post_norm_quant_f16_last_dim_24576(self):
        '''
            分片场景
            shape: (128, 1, 24576) / (1, 24576)
            取值范围
            fp16: [-5, 5]; int8: [-128, 128)
        '''
        a, b = 128, 24576
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputResIn = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputScale = torch.empty((1,), dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        while inputScale == 0:
            inputScale = torch.empty((1,), dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputOffset = torch.randint(-5, 5, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputResIn, inputGamma, inputScale, inputOffset],
                    [torch.zeros(shape0, dtype=torch.int8), torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_rms_post_norm_quant_f16_small_epsilon(self):
        '''
            small shape, epsilon=1e-8
            shape: (1, 1, 128)
            取值范围
            bf16: [-5, 5]; int8: [-128, 128)
        '''
        a, b = 1, 128
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputResIn = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputScale = torch.empty((1,), dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        while inputScale == 0:
            inputScale = torch.empty((1,), dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputOffset = torch.randint(-5, 5, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM1)
        self.execute([inputX, inputResIn, inputGamma, inputScale, inputOffset],
                    [torch.zeros(shape0, dtype=torch.int8), torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_rms_post_norm_quant_f16_big_inputScale(self):
        '''
            small shape, inputScale=1000
            shape: (1, 1, 128)
            取值范围
            bf16: [-5, 5]; int8: [-128, 128)
        '''
        a, b = 1, 128
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputResIn = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputScale = torch.full((1,), 1000, dtype=torch.float16)
        inputOffset = torch.randint(-5, 5, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputResIn, inputGamma, inputScale, inputOffset],
                    [torch.zeros(shape0, dtype=torch.int8), torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_rms_post_norm_quant_f16_small_gamma(self):
        '''
            small shape, small gamma
            shape: (1, 1, 128)
            取值范围
            bf16: [-5, 5]; int8: [-128, 128)
        '''
        a, b = 1, 128
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputResIn = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(-0.001, 0.001)
        inputScale = torch.empty((1,), dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        while inputScale == 0:
            inputScale = torch.empty((1,), dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputOffset = torch.randint(-5, 5, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputResIn, inputGamma, inputScale, inputOffset],
                    [torch.zeros(shape0, dtype=torch.int8), torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_rms_post_norm_quant_f16_small_input(self):
        '''
            small shape, small x and res_in
            shape: (1, 1, 128)
            取值范围
            bf16: [-5, 5]; int8: [-128, 128)
        '''
        a, b = 1, 128
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(-0.001, 0.001)
        inputResIn = torch.empty(shape0, dtype=torch.float16).uniform_(-0.001, 0.001)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputScale = torch.empty((1,), dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        while inputScale == 0:
            inputScale = torch.empty((1,), dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputOffset = torch.randint(-5, 5, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputResIn, inputGamma, inputScale, inputOffset],
                    [torch.zeros(shape0, dtype=torch.int8), torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_rms_post_norm_quant_f16_all_small(self):
        '''
            small shape, all small
            shape: (1, 1, 128)
            取值范围
            bf16: [-5, 5]; int8: [-128, 128)
        '''
        a, b = 1, 128
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(-0.001, 0.001)
        inputResIn = torch.empty(shape0, dtype=torch.float16).uniform_(-0.001, 0.001)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(-0.001, 0.001)
        inputScale = torch.empty((1,), dtype=torch.float16).uniform_(-0.001, 0.001)
        while inputScale == 0:
            inputScale = torch.empty((1,), dtype=torch.float16).uniform_(-0.001, 0.001)
        inputOffset = torch.randint(-5, 5, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputResIn, inputGamma, inputScale, inputOffset],
                    [torch.zeros(shape0, dtype=torch.int8), torch.zeros(shape0, dtype=torch.float16)])
 
    @op_test.only_910b
    def test_rms_post_norm_quant_bf16_model_shape(self):
        '''
            模型增量场景
            shape: (2500, 8192)
            取值范围
            bf16: [-5, 5]; int8: [-128, 128)
        '''
        a, b = 2500, 8192
        shape0 = (a, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputResIn = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputScale = torch.empty((1,), dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        while inputScale == 0:
            inputScale = torch.empty((1,), dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputOffset = torch.randint(-5, 5, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputResIn, inputGamma, inputScale, inputOffset],
                    [torch.zeros(shape0, dtype=torch.int8), torch.zeros(shape0, dtype=torch.bfloat16)])

if __name__ == '__main__':
    unittest.main()
