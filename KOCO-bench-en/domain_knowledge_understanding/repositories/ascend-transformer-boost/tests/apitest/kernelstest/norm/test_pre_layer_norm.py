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
import torch
import torch.nn.functional as F
import op_test

OP_NAME = "NormOperation"
HALF_FLOAT_MIN = -5
HALF_FLOAT_MAX = 5
QUANTMIN = -128
QUANTMAX = 127

class TestPreLayerNorm(op_test.OpTest):
    def golden_calc(self, in_tensors):
        eps = self.op_desc["specificParam"]["epsilon"]
        inputX = in_tensors[0].float()
        inputResIn = in_tensors[1].float()
        inputGamma = in_tensors[2].float()
        inputBeta = in_tensors[3].float()
        if "zoomScaleValue" in self.op_desc["specificParam"]:
            zoomScale = self.op_desc["specificParam"]["zoomScaleValue"]
            sumX = inputX + inputResIn * zoomScale
        else:
            sumX = inputX + inputResIn
        output = F.layer_norm(
            sumX, inputGamma.shape, weight=inputGamma, bias=inputBeta, eps=eps
        )

        return [output.half(), sumX.half()]

    def golden_compare(self, out_tensors, golden_out_tensors):
        diff = torch.abs(torch.subtract(out_tensors[0], golden_out_tensors[0]))
        tensor_max = torch.maximum(torch.ones(golden_out_tensors[0].shape, dtype=golden_out_tensors[0].dtype),
                                   torch.abs(golden_out_tensors[0]))
        
        factor1, factor2 = 2**(-8), 2**(-10)
        if out_tensors[0].dtype == torch.bfloat16: 
            factor1, factor2 = 2**(-8), 2**(-8)

        if torch.any(torch.greater(diff, factor1 * tensor_max)):
            logging.debug("[new standards] output accuracy failed")
            return False

        if torch.any(torch.greater(torch.abs(torch.mean(torch.div(diff, tensor_max))), factor2)):
            logging.debug("[new standards] output eb failed")
            return False
        
        diff = torch.abs(torch.subtract(out_tensors[1], golden_out_tensors[1]))
        tensor_max = torch.maximum(torch.ones(golden_out_tensors[1].shape, dtype=golden_out_tensors[1].dtype),
                                   torch.abs(golden_out_tensors[1]))

        if torch.any(torch.greater(diff, factor1 * tensor_max)):
            logging.debug("[new standards] output accuracy failed")
            return False

        if torch.any(torch.greater(torch.abs(torch.mean(torch.div(diff, tensor_max))), factor2)):
            logging.debug("[new standards] output eb failed")
            return False
        return True

    @op_test.only_910b
    def test_pre_layer_norm_bf16_case0(self):
        '''
            基础场景
            shape: (4096, 1, 10880) / (1, 10880)
            取值范围
            bfp16: [-2, 2]
        '''
        a, b = 4096, 10880
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputBeta = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputResIn = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        opParam = {"normType": 1, "epsilon": 1e-5, "inGamma": True, "inBeta": True, "inRes": True, "outRes":True,
                   "outResQuant": False, "opsMode": 0}
        self.set_param(OP_NAME, opParam)
        self.execute([inputX, inputResIn, inputGamma, inputBeta],
                     [torch.zeros(shape0, dtype=torch.bfloat16), torch.zeros(shape0, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_pre_layer_norm_bf16_case1(self):
        '''
            基础场景
            shape: (1, 1, 128) / (1, 128)
            取值范围
            bfp16: [-2, 2]
        '''
        a, b = 1, 128
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputBeta = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputResIn = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        opParam = {"normType": 1, "epsilon": 1e-5, "inGamma": True, "inBeta": True, "inRes": True, "outRes":True,
                   "outResQuant": False, "opsMode": 0}
        self.set_param(OP_NAME, opParam)
        self.execute([inputX, inputResIn, inputGamma, inputBeta],
                     [torch.zeros(shape0, dtype=torch.bfloat16), torch.zeros(shape0, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_pre_layer_norm_bf16_case3(self):
        '''
            基础场景
            shape: (128, 1, 32) / (1, 32)
            取值范围
            bfp16: [-2, 2]
        '''
        a, b = 128, 32
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.bfloat16).uniform_(1, 1)
        inputBeta = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputResIn = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        opParam = {"normType": 1, "epsilon": 1e-5, "inGamma": True, "inBeta": True, "inRes": True, "outRes":True,
                   "outResQuant": False, "opsMode": 0, "zoomScaleValue": 0.5}
        self.set_param(OP_NAME, opParam)
        self.execute([inputX, inputResIn, inputGamma, inputBeta],
                     [torch.zeros(shape0, dtype=torch.bfloat16), torch.zeros(shape0, dtype=torch.bfloat16)])
    
    @op_test.skip_310b
    @op_test.skip_910a
    def test_pre_layer_norm_f16_case0(self):
        '''
            基础场景
            shape: (4096, 1, 10880) / (1, 10880)
            取值范围
            fp16: [-2, 2]
        '''
        a, b = 4096, 10880
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputBeta = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputResIn = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        opParam = {"normType": 1, "epsilon": 1e-5, "inGamma": True, "inBeta": True, "inRes": True, "outRes":True,
                   "outResQuant": False, "opsMode": 0}
        self.set_param(OP_NAME, opParam)
        self.execute([inputX, inputResIn, inputGamma, inputBeta],
                     [torch.zeros(shape0, dtype=torch.float16), torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_pre_layer_norm_f16_case1(self):
        '''
            基础场景
            shape: (1, 1, 128) / (1, 128)
            取值范围
            fp16: [-2, 2]
        '''
        a, b = 1, 128
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputBeta = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputResIn = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        opParam = {"normType": 1, "epsilon": 1e-5, "inGamma": True, "inBeta": True, "inRes": True, "outRes":True,
                   "outResQuant": False, "opsMode": 0}
        self.set_param(OP_NAME, opParam)
        self.execute([inputX, inputResIn, inputGamma, inputBeta],
                     [torch.zeros(shape0, dtype=torch.float16), torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_pre_layer_norm_f16_case3(self):
        '''
            基础场景
            shape: (128, 1, 32) / (1, 32)
            取值范围
            fp16: [-2, 2]
        '''
        a, b = 128, 32
        shape0 = (a, 1, b)
        shape1 = (1, b)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(1, 1)
        inputBeta = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputResIn = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        opParam = {"normType": 1, "epsilon": 1e-5, "inGamma": True, "inBeta": True, "inRes": True, "outRes":True,
                   "outResQuant": False, "opsMode": 0, "zoomScaleValue": 0.5}
        self.set_param(OP_NAME, opParam)
        self.execute([inputX, inputResIn, inputGamma, inputBeta],
                     [torch.zeros(shape0, dtype=torch.float16), torch.zeros(shape0, dtype=torch.float16)])

if __name__ == '__main__':
    unittest.main()