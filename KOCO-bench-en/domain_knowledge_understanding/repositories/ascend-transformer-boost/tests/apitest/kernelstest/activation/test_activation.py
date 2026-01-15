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
import op_test
import logging


OP_NAME = "ActivationOperation"
OP_PARAM = {"activationType": 1}
OP_PARAM_GELU = {"activationType": 2}
OP_PARAM_SWISH = {"activationType": 4, "scale": 2.5}
OP_PARAM_SWIGLU_FORWARD = {"activationType": 6, "dim": -1}
OP_PARAM_SWIGLU_BACKWARD = {"activationType": 7, "dim": -1}

class TestActivation(op_test.OpTest):
    def swish(self, x):
        return x * torch.sigmoid(x)

    def swish_grad(self, x):
        return torch.sigmoid(x) + x * (1 - torch.sigmoid(x)) * torch.sigmoid(x)
    
    def golden_calc(self, in_tensors):
        actvation_type = self.op_desc["specificParam"]["activationType"]
        if actvation_type == 1:
            return [torch.nn.functional.relu(in_tensors[0])]
        elif actvation_type == 2:
            x = in_tensors[0].float()
            y = 0.5*x*(1+F.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))
            return [y]
        elif actvation_type == 4:
            x = in_tensors[0].float()
            y = x / (1 + np.exp(-x * OP_PARAM_SWISH['scale']))
            return [y]
        elif actvation_type == 6:
            x = in_tensors[0]
            dtype = x.dtype
            x = x.npu()
            a, b = x.chunk(2, dim=-1)
            a = a.to(torch.float32)
            b = b.to(torch.float32)
            y = F.silu(a) * b
            y = y.to(dtype)
            return [y.cpu()]
        elif actvation_type == 7:
            dtype = in_tensors[1].dtype
            in_tensors[0] = in_tensors[0].npu()
            in_tensors[1] = in_tensors[1].npu()
            tensor_y_grad = in_tensors[0].float()
            x = in_tensors[1].float()
            a, b = x.chunk(2, dim=-1)
            a = a.to(torch.float32)
            b = b.to(torch.float32)
            y1 = b * tensor_y_grad * self.swish_grad(a)
            y2 = tensor_y_grad * self.swish(a)
            y = torch.cat((y1, y2), dim=-1)
            y = y.to(dtype)
            return [y.cpu()]
        elif actvation_type == 8:
            return [torch.sigmoid(in_tensors[0].float())]
        else:
            return [torch.zeros_like(x) for x in in_tensors]

    def golden_compare(self, out_tensors, golden_out_tensors):
        err = 2**(-8)
        if out_tensors[0].dtype == torch.bfloat16:
            err = 2**(-7)
        elif out_tensors[0].dtype == torch.float32:
            err = 2**(-14)
        out_tensors[0] = out_tensors[0].to(torch.float32)
        golden_out_tensors[0] = golden_out_tensors[0].to(torch.float32)
        diff = torch.abs(torch.subtract(out_tensors[0], golden_out_tensors[0]))
        tensor_max = torch.maximum(torch.ones(golden_out_tensors[0].shape, dtype=golden_out_tensors[0].dtype), torch.abs(golden_out_tensors[0]))
        if torch.any(torch.greater(diff, err * tensor_max)):
            logging.debug("new golden result: failed")
            return False
        logging.debug("new golden result: true")
        return True

    def test_3d_float(self):
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([torch.randn(2, 10, 10).float()],
                     [torch.randn(2, 10, 10).float()])

    @op_test.skip_310b
    def test_gelu_float(self):
        shape = (1000000)
        input0 = np.random.uniform(low=-100, high=100, size=shape).astype(np.float32)
        self.set_param(OP_NAME, OP_PARAM_GELU)
        self.execute([torch.from_numpy(input0)],
                     [torch.zeros(shape).float()])

    @op_test.skip_310b
    def test_gelu_half(self):
        shape = (1000000)
        input0 = np.random.uniform(low=-100, high=100, size=shape).astype(np.float16)
        self.set_param(OP_NAME, OP_PARAM_GELU)
        self.execute([torch.from_numpy(input0)],
                     [torch.zeros(shape).half()])

    @op_test.only_910b
    def test_swish_bf16(self):
        shape = (1000000)
        input0 = np.random.uniform(low=0, high=100, size=shape).astype(np.float32)
        self.set_param(OP_NAME, OP_PARAM_SWISH)
        self.execute([torch.from_numpy(input0).bfloat16()],
                     [torch.zeros(shape).bfloat16()])

    @op_test.skip_310b
    def test_swiglu_forward1(self): # coreMultiUbMulti n32B对齐 case1
        input0 = np.random.uniform(-2, 2, (4096,4,8192))
        self.set_param(OP_NAME, OP_PARAM_SWIGLU_FORWARD)
        self.execute([torch.from_numpy(input0).half()],
                     [torch.zeros((4096,4,4096)).half()])

    @op_test.skip_310b
    def test_swiglu_forward2(self): # coreMultiUbMulti n32B对齐 有核间尾行 有核内尾行 case3
        input0 = np.random.uniform(-2, 2, (8192,1,7808))
        self.set_param(OP_NAME, OP_PARAM_SWIGLU_FORWARD)
        self.execute([torch.from_numpy(input0).half()],
                     [torch.zeros((8192,1,3904)).half()])

    @op_test.only_910b
    def test_swiglu_forward3(self): # coreMultiUbMulti n非32B对齐 有核间尾行 无核内尾行 case8
        input0 = np.random.uniform(-2, 2, (8192,1,3900))
        self.set_param(OP_NAME, OP_PARAM_SWIGLU_FORWARD)
        self.execute([torch.from_numpy(input0).half()],
                     [torch.zeros((8192,1,1950)).half()])
    @op_test.only_910b
    def test_swiglu_forward4(self): # coreMultiUbMulti n非32B对齐 有核间尾行 有核内尾行 case9
        input0 = np.random.uniform(-2, 2, (4855,1,5890))
        self.set_param(OP_NAME, OP_PARAM_SWIGLU_FORWARD)
        self.execute([torch.from_numpy(input0).float()],
                     [torch.zeros((4855,1,2945)).float()])
    @op_test.only_910b
    def test_swiglu_forward5(self): # coreMultiUbSingle n非32B对齐 有核间尾行 case2
        input0 = np.random.uniform(-2, 2, (8000,2,57890))
        self.set_param(OP_NAME, OP_PARAM_SWIGLU_FORWARD)
        self.execute([torch.from_numpy(input0).float()],
                     [torch.zeros((8000,2,28945)).float()])

    @op_test.skip_310b
    def test_swiglu_forward6(self): # m小于核数 n32B对齐 case1
        input0 = np.random.uniform(-2, 2, (19,1,8192))
        self.set_param(OP_NAME, OP_PARAM_SWIGLU_FORWARD)
        self.execute([torch.from_numpy(input0).float()],
                     [torch.zeros((19,1,4096)).float()])
    @op_test.only_910b
    def test_swiglu_forward7(self): # n不足32B case3
        input0 = np.random.uniform(-2, 2, (1000,4,18))
        self.set_param(OP_NAME, OP_PARAM_SWIGLU_FORWARD)
        self.execute([torch.from_numpy(input0).bfloat16()],
                     [torch.zeros((1000,4,9)).bfloat16()])

    @op_test.only_910b
    def test_swiglu_backward_is32Align(self):
        input0 = np.random.uniform(-2, 2, (8192,1,1952)).astype("float32")
        input1 = np.random.uniform(-2, 2, (8192,1,3904)).astype("float32")
        self.set_param(OP_NAME, OP_PARAM_SWIGLU_BACKWARD)
        self.execute([torch.from_numpy(input0).bfloat16(), torch.from_numpy(input1).bfloat16()],
                     [torch.zeros((8192,1, 3904)).bfloat16()])

    @op_test.only_910b
    def test_swiglu_backward_not32Align(self):
        input0 = np.random.uniform(-2, 2, (8192,1,700)).astype("float32")
        input1 = np.random.uniform(-2, 2, (8192,1,1400)).astype("float32")
        self.set_param(OP_NAME, OP_PARAM_SWIGLU_BACKWARD)
        self.execute([torch.from_numpy(input0).bfloat16(), torch.from_numpy(input1).bfloat16()],
                     [torch.zeros((8192,1, 1400)).bfloat16()])

    def test_sigmoid_fp16_uniform(self):
        shape = (np.random.randint(1, 10001), np.random.randint(1, 10001))
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float16)
        op_param = {"activationType": 8}
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0)],
                     [torch.zeros(shape).to(torch.float16)])

    def test_sigmoid_fp16_normal(self):
        shape = (np.random.randint(1, 10001), np.random.randint(1, 10001))
        input0 = np.random.normal(loc=(0.5 - np.random.rand()) * 200, scale=24 * np.random.rand() + 1, size=shape).astype(np.float16)
        op_param = {"activationType": 8}
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0)],
                     [torch.zeros(shape).to(torch.float16)])
    
    @op_test.only_910b
    def test_sigmoid_bf16_uniform(self):
        shape = (np.random.randint(1, 10001), np.random.randint(1, 10001))
        input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float32)
        op_param = {"activationType": 8}
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0).bfloat16()],
                     [torch.zeros(shape).bfloat16()])

    @op_test.only_910b
    def test_sigmoid_bf16_normal(self):
        shape = (np.random.randint(1, 10001), np.random.randint(1, 10001))
        input0 = np.random.normal(loc=(0.5 - np.random.rand()) * 200, scale=24 * np.random.rand() + 1, size=shape).astype(np.float32)
        op_param = {"activationType": 8}
        self.set_param(OP_NAME, op_param)
        self.execute([torch.from_numpy(input0).bfloat16()],
                     [torch.zeros(shape).bfloat16()])

if __name__ == '__main__':
    unittest.main()
