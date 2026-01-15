import os
import unittest
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append('../')
import op_test
from tensor_file import read_tensor

OP_NAME = "NormOperation"
OP_PARAM0 = {"normType": 1, "epsilon": 1e-5,
             "inGamma": True, "inBeta": False, "inRes": False, "outMean": False, "outVarience": False, "outResQuant": False}

HALF_FLOAT_MIN = -2.0
HALF_FLOAT_MAX = 2.0

class CohereLayerNorm(nn.Module):
    def __init__(self, hidden_size=None, eps=1e-5, bias=False):
        """The hidden size can be a tuple or an int. The tuple is used for QKNorm to normalize across head_dim"""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight.to(torch.float32) * hidden_states
        return hidden_states

class TestCohereLayerNorm(op_test.OpTest):
    def golden_calc(self, in_tensors):
        eps = self.op_desc["specificParam"]["epsilon"]
        co_layernorm = CohereLayerNorm
        co_layernorm_dim = co_layernorm(in_tensors[0].shape, eps=eps).cpu()
        a_dim = in_tensors[0]
        w_dim = in_tensors[1]
        co_layernorm_dim.weight.data = w_dim

        with torch.no_grad():
            y_co_dim = co_layernorm_dim(a_dim)
        return y_co_dim

    def golden_compare(self, out_tensors, golden_out_tensors):
        expect = golden_out_tensors
        tensor_dtype = expect.dtype
        actual = out_tensors[0].to(tensor_dtype)
        diff = torch.abs(torch.subtract(actual, expect))
        tensor_max = torch.maximum(torch.ones(expect.shape, dtype=expect.dtype), torch.abs(expect))

        if torch.any(torch.greater(diff, 2**(-8) * tensor_max)):
            print("[new standards] output accuracy failed")
            return False
        if torch.any(torch.greater(torch.abs(torch.mean(torch.div(diff, tensor_max))), 2**(-7))):
            print("[new standards] output eb failed")
            return False
        return True

    @op_test.skip_310b
    @op_test.skip_910a
    def test_layer_norm_cohere_f16_case0(self):
        '''
            基础场景
            shape: (8, 8, 64, 25760) / (64, 25760)
            取值范围
            fp16: [-2.0, 2.0]
        '''
        batch, seqLen, numHead, headSize = 8, 8, 64, 25760
        shape0 = (batch, seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_layer_norm_cohere_f16_case1(self):
        '''
            基础场景
            shape: (1, 1, 1, 10240) / (1, 10240)
            取值范围
            fp16: [-2.0, 2.0]
        '''
        batch, seqLen, numHead, headSize = 1, 1, 1, 10240
        shape0 = (batch, seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_layer_norm_cohere_f16_case2(self):
        '''
            基础场景
            shape: (8, 8, 64, 128) / (8, 128)
            取值范围
            fp16: [-2.0, 2.0]
        '''
        batch, seqLen, numHead, headSize = 8, 8, 64, 128
        shape0 = (batch, seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_layer_norm_cohere_f16_case3(self):
        '''
            基础场景
            shape: (2, 1, 2, 32) / (2, 32)
            取值范围
            fp16: [-2.0, 2.0]
            '''
        batch, seqLen, numHead, headSize = 2, 1, 2, 32
        shape0 = (batch, seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_layer_norm_cohere_f16_case4(self):
        '''
            基础场景
            shape: (1, 1, 131073, 256) / (131073, 256)
            取值范围
            fp16: [-2.0, 2.0]
            '''
        batch, seqLen, numHead, headSize = 1, 1, 131073, 256
        shape0 = (batch, seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_layer_norm_cohere_f16_case5(self):
        '''
            基础场景
            shape: (1, 1, 1111, 288) / (1111, 288)
            取值范围
            fp16: [-2.0, 2.0]
            '''
        batch, seqLen, numHead, headSize = 1, 1, 1111, 288
        shape0 = (batch, seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_layer_norm_cohere_f16_case6(self):
        '''
            基础场景
            shape: (1, 1, 2120, 64) / (2120, 64)
            取值范围
            fp16: [-2.0, 2.0]
            '''
        batch, seqLen, numHead, headSize = 1, 1, 2120, 64
        shape0 = (batch, seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_layer_norm_cohere_f16_case7(self):
        '''
            基础场景
            shape: (1, 1, 20, 15584) / (20, 15584)
            取值范围
            fp16: [-2.0, 2.0]
            '''
        batch, seqLen, numHead, headSize = 1, 1, 20, 15584
        shape0 = (batch, seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_layer_norm_cohere_f16_case8(self):
        '''
            基础场景
            shape: (1, 1, 63, 19552) / (63, 19552)
            取值范围
            fp16: [-2.0, 2.0]
            '''
        batch, seqLen, numHead, headSize = 1, 1, 63, 19552
        shape0 = (batch, seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_layer_norm_cohere_f16_case9(self):
        '''
            基础场景
            shape: (1, 63, 19552) / (63, 19552)
            取值范围
            fp16: [-2.0, 2.0]
            '''
        seqLen, numHead, headSize = 1, 63, 19552
        shape0 = (seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.float16)])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_layer_norm_cohere_f16_case10(self):
        '''
            基础场景
            shape: (1, 1, 63, 64) / (63, 64)
            取值范围
            fp16: [-2.0, 2.0]
            '''
        seqLen, numHead, headSize = 1, 63, 64
        shape0 = (seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.float16)])

    @op_test.only_910b
    def test_layer_norm_cohere_bf16_case0(self):
        '''
            基础场景
            shape: (8, 8, 64, 25760) / (64, 25760)
            取值范围
            fp16: [-2.0, 2.0]
        '''
        batch, seqLen, numHead, headSize = 8, 8, 64, 25760
        shape0 = (batch, seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_layer_norm_cohere_bf16_case1(self):
        '''
            基础场景
            shape: (1, 1, 1, 10240) / (1, 10240)
            取值范围
            fp16: [-2.0, 2.0]
        '''
        batch, seqLen, numHead, headSize = 1, 1, 1, 10240
        shape0 = (batch, seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.bfloat16)])
            

    @op_test.only_910b
    def test_layer_norm_cohere_bf16_case2(self):
        '''
            基础场景
            shape: (8, 8, 64, 128) / (64, 128)
            取值范围
            fp16: [-2.0, 2.0]
        '''
        batch, seqLen, numHead, headSize = 8, 8, 64, 128
        shape0 = (batch, seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.bfloat16)])
            

    @op_test.only_910b
    def test_layer_norm_cohere_bf16_case3(self):
        '''
            基础场景
            shape: (2, 1, 2, 32) / (2, 32)
            取值范围
            fp16: [-2.0, 2.0]
        '''
        batch, seqLen, numHead, headSize = 2, 1, 2, 32
        shape0 = (batch, seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.bfloat16)])
        

    @op_test.only_910b
    def test_layer_norm_cohere_bf16_case4(self):
        '''
            基础场景
            shape: (1, 1, 131073, 256) / (131073, 256)
            取值范围
            fp16: [-2.0, 2.0]
        '''
        batch, seqLen, numHead, headSize = 1, 1, 131073, 256
        shape0 = (batch, seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                        [torch.zeros(shape0, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_layer_norm_cohere_bf16_case5(self):
        '''
            基础场景
            shape: (1, 1, 1111, 288) / (1111, 288)
            取值范围
            fp16: [-2.0, 2.0]
            '''
        batch, seqLen, numHead, headSize = 1, 1, 1111, 288
        shape0 = (batch, seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_layer_norm_cohere_bf16_case6(self):
        '''
            基础场景
            shape: (1, 1, 2120, 64) / (2120, 64)
            取值范围
            fp16: [-2.0, 2.0]
            '''
        batch, seqLen, numHead, headSize = 1, 1, 2120, 64
        shape0 = (batch, seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_layer_norm_cohere_bf16_case7(self):
        '''
            基础场景
            shape: (1, 1, 20, 15584) / (20, 15584)
            取值范围
            fp16: [-2.0, 2.0]
            '''
        batch, seqLen, numHead, headSize = 1, 1, 20, 15584
        shape0 = (batch, seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_layer_norm_cohere_bf16_case8(self):
        '''
            基础场景
            shape: (1, 1, 63, 19552) / (63, 19552)
            取值范围
            fp16: [-2.0, 2.0]
            '''
        batch, seqLen, numHead, headSize = 1, 1, 63, 19552
        shape0 = (batch, seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_layer_norm_cohere_bf16_case9(self):
        '''
            基础场景
            shape: (1, 63, 19552) / (63, 19552)
            取值范围
            fp16: [-2.0, 2.0]
            '''
        seqLen, numHead, headSize = 1, 63, 19552
        shape0 = (seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.bfloat16)])

    @op_test.only_910b
    def test_layer_norm_cohere_bf16_case10(self):
        '''
            基础场景
            shape: (1, 63, 64) / (63, 64)
            取值范围
            fp16: [-2.0, 2.0]
            '''
        seqLen, numHead, headSize = 1, 63, 64
        shape0 = (seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.bfloat16)])

    @op_test.skip_310b
    @op_test.skip_910a
    @unittest.expectedFailure
    def test_layer_norm_cohere_failed_case0(self):
        '''
            基础场景
            shape: (1, 1, 1, 32) / (1, 32)
            取值范围
            fp16: [-2.0, 2.0]
            '''
        batch, seqLen, numHead, headSize = 1, 1, 1, 32
        shape0 = (batch, seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.float16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.bfloat16)])

    @op_test.skip_310b
    @op_test.skip_910a
    @unittest.expectedFailure
    def test_layer_norm_cohere_failed_case1(self):
        '''
            基础场景
            shape: (1, 1, 1, 32) / (1, 32)
            取值范围
            fp16: [-2.0, 2.0]
            '''
        batch, seqLen, numHead, headSize = 1, 1, 1, 32
        shape0 = (batch, seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.float32).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.float32).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.float32)])

    @op_test.skip_310b
    @op_test.skip_910a
    @unittest.expectedFailure
    def test_layer_norm_cohere_failed_case2(self):
        '''
            基础场景
            shape: (1, 1, 1, 32) / (32, 1)
            取值范围
            fp16: [-2.0, 2.0]
            '''
        batch, seqLen, numHead, headSize = 1, 1, 1, 32
        shape0 = (batch, seqLen, numHead, headSize)
        shape1 = (headSize, numHead)
        inputX = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.bfloat16)])

    @op_test.skip_310b
    @op_test.skip_910a
    @unittest.expectedFailure
    def test_layer_norm_cohere_failed_case3(self):
        '''
            基础场景
            shape: (1, 32) / (1, 32)
            取值范围
            fp16: [-2.0, 2.0]
            '''
        numHead, headSize = 1, 32
        shape0 = (numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.bfloat16)])

    @op_test.skip_310b
    @op_test.skip_910a
    @unittest.expectedFailure
    def test_layer_norm_cohere_failed_case4(self):
        '''
            基础场景
            shape: (1, 1, 1, 1, 32) / (1, 32)
            取值范围
            fp16: [-2.0, 2.0]
            '''
        dummy_batch, batch, seqLen, numHead, headSize = 1, 1, 1, 1, 32
        shape0 = (dummy_batch, batch, seqLen, numHead, headSize)
        shape1 = (numHead, headSize)
        inputX = torch.empty(shape0, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        inputGamma = torch.empty(shape1, dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([inputX, inputGamma],
                    [torch.zeros(shape0, dtype=torch.bfloat16)])

if __name__ == '__main__':
    unittest.main()
