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
import sys, os
import op_test
sys.path.append('../')
from precision_calcu import *
from enum import Enum

class ActivationType(Enum):
    GELU = 0
    FASTGELU = 1
    FASTGELUV2 = 2

class ImplMode(Enum):
    HIGH_PRECISION = 0
    HIGH_PERFORMANCE = 1

DRANGE = (-2, 2)
SCALE_DRANGE = (-0.01, 0.01)
BIAS_DRANGE = (-10, 10)
SCALE_DQUANT_DRANGE = (-2, 2)
BIAS_DQUANT_DRANGE = (-10, 10)

OP_NAME = "FFNOperation"
OP_PARAM = {"activationType": 0, "transX": False, "transW1": True, "transW2": True, "implMode": 0}

def process_deq_scale(deq_scale) -> np.ndarray:
    new_deq_scale = np.frombuffer(deq_scale.tobytes(), dtype=np.uint32)
    return new_deq_scale.astype(np.int64)

class TestFFN(op_test.OpTest):
    def golden_calc(self, in_tensors):
        x = in_tensors[0].npu()
        w1 = in_tensors[1].npu()
        w2 = in_tensors[2].npu()
        scale = in_tensors[3].npu()
        deqScale1 = torch.from_numpy(self.deqScale1Tmp).npu()
        deqScale2 = torch.from_numpy(self.deqScale2Tmp).npu()
        bias = in_tensors[6].npu()
        deqBias1 = in_tensors[7].npu()
        deqBias2 = in_tensors[8].npu()
        activationType = ActivationType(self.op_desc["specificParam"]["activationType"])

        mm1 = torch.matmul(x.float(), torch.transpose(w1, 0, 1).float())
        mm1 = ((mm1.int() + deqBias1).float() * deqScale1).half()
        active = self.activation_func(mm1, activationType)
        quant = active * scale + bias
        activeInt8 =  torch.round(quant.float()).char()
        mm2 = torch.matmul(activeInt8.float(), torch.transpose(w2, 0, 1).float())
        golden = ((mm2.int() + deqBias2).float() * deqScale2).half()

        # 计算真值
        mm1 = torch.matmul(x.float(), torch.transpose(w1, 0, 1).float())
        mm1 = ((mm1.int() + deqBias1).float() * deqScale1)
        active = self.activation_func(mm1, activationType)
        quant = active * scale.float() + bias.float()
        activeInt8 =  torch.round(quant.float()).char()
        mm2 = torch.matmul(activeInt8.float(), torch.transpose(w2, 0, 1).float())
        self.true_out = ((mm2.int() + deqBias2).float() * deqScale2).cpu()

        return [golden.cpu()]

    def golden_compare(self, out_tensors, golden_out_tensors):
        result_double = compare_cv(self.true_out, golden_out_tensors[0], out_tensors[0])
        return result_double
    
    def gelu(self, x):
        y = -1.5957691 * (x + 0.044715 * torch.pow(x, 3))
        res = x / (1 + torch.exp(y))
        return res

    def fast_gelu(self, x):
        return 1 / (1 + torch.exp(-1.702 * x)) * x

    def fast_gelu_v2(self, x):
        offset = 1
        if x.dtype == torch.float16:
            offset = 0.0000001
        elif  x.dtype == torch.float32:
            offset = 0.000000000001
        else :
            print("dtype not support")
            return x
        print("offset:" + str(offset))
        sgn = (x + offset) / torch.abs(x + offset)
        tmp = -0.1444 * torch.pow(torch.clip(torch.abs(0.7071*x), None, 1.769) - 1.769, 2) + 0.5
        res = x * (sgn * tmp + 0.5)
        return res
    
    def activation_func(self, in_tensor, activation_type):
        out_tensor = None
        if activation_type == ActivationType.GELU:
            print("GELU\n")
            out_tensor = self.gelu(in_tensor)
        elif activation_type == ActivationType.FASTGELU:
            print("FASTGELU\n")
            out_tensor = self.fast_gelu(in_tensor)
        elif activation_type == ActivationType.FASTGELUV2:
            print("FASTGELUV2\n")
            out_tensor = self.fast_gelu_v2(in_tensor)
        else:
            print("Not supported activation func\n")
            return in_tensor
        return out_tensor
    
    def gen_data_int8(self, bs, n, h):
        self.bs = bs
        self.n = n
        self.h = h
        self.x = torch.from_numpy(np.random.uniform(*DRANGE, size=(bs, h)).astype(np.int8))
        self.weight1 = torch.from_numpy(np.random.uniform(*DRANGE, size=(n, h)).astype(np.int8))
        self.weight2 = torch.from_numpy(np.random.uniform(*DRANGE, size=(h, n)).astype(np.int8))
        self.scale = torch.from_numpy(np.random.uniform(*SCALE_DRANGE, size=(n)).astype(np.float16))
        self.deqScale1Tmp = np.random.randint(*SCALE_DQUANT_DRANGE, size=(n)).astype(np.float32)
        self.deqScale2Tmp = np.random.randint(*SCALE_DQUANT_DRANGE, size=(h)).astype(np.float32)
        self.deqScale1 = torch.from_numpy(process_deq_scale(self.deqScale1Tmp))
        self.deqScale2 = torch.from_numpy(process_deq_scale(self.deqScale2Tmp))
        self.bias = torch.from_numpy(np.random.uniform(*BIAS_DRANGE, size=(n)).astype(np.float16))
        self.deqBias1 = torch.from_numpy(np.random.randint(*BIAS_DQUANT_DRANGE, size=(n)).astype(np.int32))
        self.deqBias2 = torch.from_numpy(np.random.randint(*BIAS_DQUANT_DRANGE, size=(h)).astype(np.int32))
        self.y = torch.zeros((bs, h)).half()
    
    def test_ffn1(self):
        self.set_support_310p_only()
        bs, n, h = (1024, 384, 96)
        OP_PARAM['activationType'] = 0
        OP_PARAM['transX'] = False
        OP_PARAM['transW1'] = True
        OP_PARAM['transW2'] = True
        OP_PARAM['implMode'] = 0
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([
            self.format_nd, self.format_nd, self.format_nd,
            self.format_nd, self.format_nd, self.format_nd,
            self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.gen_data_int8(bs, n, h)
        self.execute(
            [ 
                self.x, self.weight1, self.weight2,
                self.scale, self.deqScale1, self.deqScale2,
                self.bias, self.deqBias1, self.deqBias2
            ],
            [
                self.y
            ])
    
    def test_ffn2(self):
        self.set_support_310p_only()
        bs, n, h = (75264, 384, 96)
        OP_PARAM['activationType'] = 1
        OP_PARAM['transX'] = False
        OP_PARAM['transW1'] = True
        OP_PARAM['transW2'] = True
        OP_PARAM['implMode'] = 0
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([
            self.format_nd, self.format_nd, self.format_nd,
            self.format_nd, self.format_nd, self.format_nd,
            self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.gen_data_int8(bs, n, h)
        self.execute(
            [ 
                self.x, self.weight1, self.weight2,
                self.scale, self.deqScale1, self.deqScale2,
                self.bias, self.deqBias1, self.deqBias2
            ],
            [
                self.y
            ])
    
    def test_ffn3(self):
        self.set_support_310p_only()
        bs, n, h = (49,3072,768)
        OP_PARAM['activationType'] = 2
        OP_PARAM['transX'] = False
        OP_PARAM['transW1'] = True
        OP_PARAM['transW2'] = True
        OP_PARAM['implMode'] = 0
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([
            self.format_nd, self.format_nd, self.format_nd,
            self.format_nd, self.format_nd, self.format_nd,
            self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.gen_data_int8(bs, n, h)
        self.execute(
            [ 
                self.x, self.weight1, self.weight2,
                self.scale, self.deqScale1, self.deqScale2,
                self.bias, self.deqBias1, self.deqBias2
            ],
            [
                self.y
            ])
    
    def test_ffn4(self):
        self.set_support_310p_only()
        bs, n, h = (1027, 753, 384)
        OP_PARAM['activationType'] = 0
        OP_PARAM['transX'] = False
        OP_PARAM['transW1'] = True
        OP_PARAM['transW2'] = True
        OP_PARAM['implMode'] = 1
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([
            self.format_nd, self.format_nd, self.format_nd,
            self.format_nd, self.format_nd, self.format_nd,
            self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.gen_data_int8(bs, n, h)
        self.execute(
            [ 
                self.x, self.weight1, self.weight2,
                self.scale, self.deqScale1, self.deqScale2,
                self.bias, self.deqBias1, self.deqBias2
            ],
            [
                self.y
            ])
    
    def test_ffn5(self):
        self.set_support_310p_only()
        bs, n, h = (3139, 3027, 768)
        OP_PARAM['activationType'] = 1
        OP_PARAM['transX'] = False
        OP_PARAM['transW1'] = True
        OP_PARAM['transW2'] = True
        OP_PARAM['implMode'] = 1
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([
            self.format_nd, self.format_nd, self.format_nd,
            self.format_nd, self.format_nd, self.format_nd,
            self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.gen_data_int8(bs, n, h)
        self.execute(
            [ 
                self.x, self.weight1, self.weight2,
                self.scale, self.deqScale1, self.deqScale2,
                self.bias, self.deqBias1, self.deqBias2
            ],
            [
                self.y
            ])
    
    def test_ffn6(self):
        self.set_support_310p_only()
        bs, n, h = (4704, 768, 192)
        OP_PARAM['activationType'] = 2
        OP_PARAM['transX'] = False
        OP_PARAM['transW1'] = True
        OP_PARAM['transW2'] = True
        OP_PARAM['implMode'] = 1
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([
            self.format_nd, self.format_nd, self.format_nd,
            self.format_nd, self.format_nd, self.format_nd,
            self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.gen_data_int8(bs, n, h)
        self.execute(
            [ 
                self.x, self.weight1, self.weight2,
                self.scale, self.deqScale1, self.deqScale2,
                self.bias, self.deqBias1, self.deqBias2
            ],
            [
                self.y
            ])

if __name__ == '__main__':
    unittest.main()
