#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
             "inGamma": True, "inBeta": True, "inRes": False, "outMean": False, "outVarience": False, "outResQuant": False}

HALF_FLOAT_MIN = -2.0
HALF_FLOAT_MAX = 2.0


class TestAdaLayerNorm(op_test.OpTest):
    def modulate(self, input, gamma, beta):
        if gamma.dim() == 2:
            assert gamma.dim() == 2
            input_bs, input_seq, input_dim = input.shape
            g_bs, g_dim = gamma.shape
            assert input_bs % g_bs == 0 and input_dim == g_dim
            input = input.view(g_bs, input_bs//g_bs, input_seq, input_dim)
            beta = beta.view(g_bs, 1, 1, g_dim)
            gamma = gamma.view(g_bs, 1, 1, g_dim)
            input = input * (1 + gamma) + beta
            input = input.view(input_bs, input_seq, input_dim)
        else:
            assert input.dim() == gamma.dim() == beta.dim() == 3
            assert gamma.shape == beta.shape
            input_bs, input_seq, input_dim = input.shape
            g_bs, g_seq, g_dim = gamma.shape
            assert input_bs % g_bs == 0 and input_seq % g_seq == 0 and input_dim == g_dim
            input = input.view(g_bs, input_bs//g_bs, g_seq, input_seq//g_seq, input_dim)
            beta = beta.view(g_bs, 1, g_seq, 1, g_dim)
            gamma = gamma.view(g_bs, 1, g_seq, 1, g_dim)
            input = input * (1 + gamma) + beta
            input = input.view(input_bs, input_seq, input_dim)
        return input

    def golden_calc(self, in_tensors):
        eps = self.op_desc["specificParam"]["epsilon"]
        output = F.layer_norm(in_tensors[0].float(),
                              [in_tensors[0].shape[-1]],
                              eps=eps)
        output = self.modulate(output, in_tensors[1], in_tensors[2])
        return [output]

    def get_eb(self, golden, actual):
        golden = golden[0].to(torch.float32)
        golden_nmax = torch.clamp(torch.abs(golden), min=1)
        actual_error = actual[0].to(torch.float32) - golden
        EB = torch.mean(actual_error / golden_nmax)
        result = EB <= 2 ** (-7)
        return result

    def ref_compare(self, golden, actual, thresh):
        golden = golden[0].to(torch.float32)
        golden_nmax = torch.clamp(torch.abs(golden), min=1)
        abs_error = torch.abs(actual[0].to(torch.float32) - golden)
        result = (abs_error <= thresh * golden_nmax).all()
        return result

    def golden_compare(self, out_tensors, golden_out_tensors):
        eb = self.get_eb(golden_out_tensors, out_tensors)
        cmp = self.ref_compare(golden_out_tensors, out_tensors, 2**-7 if out_tensors[0].numel() < 2048 else 2**-6)
        return eb and cmp

    @op_test.only_910b
    def test_adalayer_norm_bf16_case0(self):
        '''
            基础场景
            shape: (2, 32400, 1152) / (1, 1152)
            取值范围
            bf16: [-2, 2]
        '''
        a, b, c = 2, 32400, 1152
        shape0 = (a, b, c)
        shape1 = (1, c)
        inputX = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputBeta = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma, inputBeta],
                     [torch.zeros(shape0, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_adalayer_norm_bf16_case1(self):
        '''
            基础场景
            shape: (96, 1, 32) / (96, 32)
            取值范围
            bf16: [-2, 2]
        '''
        a, b = 96, 32
        shape0 = (a, 1, b)
        shape1 = (a, b)
        inputX = torch.empty(shape0).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.bfloat16)
        inputGamma = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.bfloat16)
        inputBeta = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.bfloat16)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma, inputBeta],
                     [torch.zeros(shape0, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_adalayer_norm_bf16_case2(self):
        '''
            基础场景
            shape: (4096, 20, 128) / (256, 2, 128)
            取值范围
            bf16: [-2, 2]
        '''
        a, b, c, d, e = 4096, 20, 128, 256, 2
        shape0 = (a, b, c)
        shape1 = (d, e, c)
        inputX = torch.empty(shape0).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.bfloat16)
        inputGamma = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.bfloat16)
        inputBeta = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.bfloat16)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma, inputBeta],
                     [torch.zeros(shape0, dtype=torch.bfloat16)])
    
    @op_test.only_910b
    def test_adalayer_norm_bf16_case3(self):
        '''
            基础场景
            shape: (100, 30, 8192) / (10, 3, 8192)
            取值范围
            bf16: [-2, 2]
        '''
        a, b, c, d, e = 100, 30, 8192, 10, 3
        shape0 = (a, b, c)
        shape1 = (d, e, c)
        inputX = torch.empty(shape0).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.bfloat16)
        inputGamma = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.bfloat16)
        inputBeta = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.bfloat16)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma, inputBeta],
                     [torch.zeros(shape0, dtype=torch.bfloat16)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_adalayer_norm_fp16_case0(self):
        '''
            基础场景
            shape: (2, 1024, 8192) / (1, 256, 8192)
            取值范围
            fp16: [-2, 2]
        '''
        a, b, c, d, e = 2, 1024, 8192, 1, 256
        shape0 = (a, b, c)
        shape1 = (d, e, c)
        inputX = torch.empty(shape0).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputGamma = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputBeta = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma, inputBeta],
                     [torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_adalayer_norm_fp16_case1(self):
        '''
            基础场景
            shape: (2, 32400, 1152) / (1, 1152)
            取值范围
            fp16: [-2, 2]
        '''
        a, b, c = 2, 32400, 1152
        shape0 = (a, b, c)
        shape1 = (1, c)
        inputX = torch.empty(shape0).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputGamma = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        inputBeta = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma, inputBeta],
                     [torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_adalayer_norm_fp16_case2(self):
        '''
            基础场景
            shape: (20, 1, 1152) / (2, 1152)
            取值范围
            fp16: [-2, 2]
        '''
        a, b = 20, 1152
        shape0 = (a, 1, b)
        shape1 = (2, b)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputBeta = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma, inputBeta],
                     [torch.zeros(shape0, dtype=torch.float16)])

if __name__ == '__main__':
    unittest.main()
