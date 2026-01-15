# 
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# 
import os
import unittest
import random
import logging
import numpy as np
import torch
import sys
import op_test
import subprocess
import pdb

OP_NAME = "SwiGluQuantOperation"

class TestSwiGluQuant(op_test.OpTest):
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def do_swiglu(self, a, b):
        sigmoid_mul_a = self.sigmoid(a) * a
        swiglu_y = sigmoid_mul_a * b
        return swiglu_y
    
    def do_quant(self, swiglu_y):
        y_tmp = swiglu_y
        y_tmp = np.array(y_tmp)
        y_max = np.amax(np.abs(y_tmp), axis=1)  # 动态量化依赖于每一行的最大值来调整缩放因子
        dynamic_scale_tmp = 127 / y_max
        dynamic_scale = dynamic_scale_tmp.reshape(-1, 1)  # 将一维行向量转换为二维，如(256)转为(256,1)
        y_tmp = y_tmp * dynamic_scale
        quant_y_tmp = np.round(y_tmp)  # 使用 numpy 进行乘法和四舍五入
        quant_y = quant_y_tmp.astype(np.int8)  # 转换为 int8 类型
        dynamic_scale_output = np.array([])
        dynamic_scale = 1 / dynamic_scale
        dynamic_scale_output = np.append(dynamic_scale_output, dynamic_scale)
        return quant_y,dynamic_scale_output
    
    def SwiGluQuantGolden(self, x_golden):
        x_golden = np.array(x_golden.cpu().float()).astype(np.float32)
        a, b = np.split(x_golden, 2, axis=1)
        wiglu_y = self.do_swiglu(a, b)
        quant_y,dynamic_scale = self.do_quant(wiglu_y)
        return quant_y,dynamic_scale
    
    def golden_calc(self, in_tensors):
        golden_res = self.SwiGluQuantGolden(in_tensors[0])
        return golden_res
    
    def golden_compare(self, out_tensors, golden_out_tensors):
        actual_output_y = out_tensors[0]
        actual_output_scale = out_tensors[1]
        golden_output_y = golden_out_tensors[0]
        golden_output_scale = golden_out_tensors[1]
        diff_y = np.abs(actual_output_y - golden_output_y)
        return not (diff_y > 1).nonzero().size(0) > 0 and np.allclose(actual_output_scale, golden_output_scale, rtol=0.0001, atol=0.0001)
    
    @op_test.only_910b
    def test_swi_glu_quant_case0(self):
        input_token_num = 128
        input_hidden_size = 4096
        output_hidden_size = 2048
        shape0 = (input_token_num, input_hidden_size)
        shape1 = (input_token_num, output_hidden_size)
        self.shape2 = (input_token_num,)
        x = torch.empty(shape0).uniform_(0, 1).to(torch.float16)
        OP_NAME = "SwiGluQuantOperation"
        OP_PARAM = {}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd])
        self.execute([x],[torch.zeros(shape1, dtype=torch.int8),torch.zeros(self.shape2, dtype=torch.float)])


    @op_test.only_910b
    def test_swi_glu_quant_case_large(self):
        input_token_num = 2**31-1
        input_hidden_size = 4
        output_hidden_size = 2
        shape0 = (input_token_num, input_hidden_size)
        shape1 = (input_token_num, output_hidden_size)
        self.shape2 = (input_token_num,)
        x = torch.empty(shape0).uniform_(0, 1).to(torch.float16)
        OP_NAME = "SwiGluQuantOperation"
        OP_PARAM = {}
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nd])
        self.execute([x],[torch.zeros(shape1, dtype=torch.int8),torch.zeros(self.shape2, dtype=torch.float)])   

if __name__ == '__main__':
    unittest.main()