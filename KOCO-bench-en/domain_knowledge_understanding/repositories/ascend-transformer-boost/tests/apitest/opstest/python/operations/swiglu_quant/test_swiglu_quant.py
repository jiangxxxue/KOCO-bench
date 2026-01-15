#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import sys
import os
import unittest
import torch
import torch_npu
import numpy as np
 
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402
 
OP_NAME = "SwigluQuantOperation"
PARAM = {"quantType":0}
 
class TestSwigluQuantOperation(operation_test.OperationTest):
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
        a, b = np.split(x_golden, 2, axis=1)
        wiglu_y = self.do_swiglu(a, b)
        quant_y, dynamic_scale = self.do_quant(wiglu_y)
        return torch.from_numpy(quant_y).npu(), torch.from_numpy(dynamic_scale).npu()
    
    def golden_calc(self, in_tensors):
        golden_res = self.SwiGluQuantGolden(in_tensors[0].cpu())
        return [golden_res[0], golden_res[1]]
    
    def golden_compare(self, out_tensor, golden_out_tensor):
        actual_output = out_tensor.cpu()
        golden_output = golden_out_tensor.cpu()
        res = False
        if actual_output.dtype == torch.int8:
            diff = np.abs(actual_output - golden_output)
            res = not (diff > 1).nonzero().size(0) > 0
        elif actual_output.dtype == torch.float32:
            res = np.allclose(actual_output, golden_output, rtol=0.0001, atol=0.0001)
        return res
    
    def test_swi_glu_quant_case0(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        input_token_num = 128
        input_hidden_size = 4096
        
        shape = (input_token_num, input_hidden_size)
        
        x = torch.empty(shape).uniform_(0, 1).to(torch.float16)
        self.execute(OP_NAME, PARAM, [x.npu().half()])
    
        
if __name__ == '__main__':
    unittest.main()