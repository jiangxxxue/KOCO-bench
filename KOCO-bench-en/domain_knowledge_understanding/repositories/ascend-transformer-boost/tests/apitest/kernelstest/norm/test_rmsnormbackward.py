#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import sys
import unittest
import numpy as np
import torch
import torch.nn.functional as F
import op_test
epsilon = 1e-6

RMS_NORM_GRAD = 4
OP_NAME = "NormOperation"
OP_PARAM = {"normType": RMS_NORM_GRAD}
@torch.jit.script
def rms_foward(x):
    l2_norm_inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    y = x * l2_norm_inv
    return y, l2_norm_inv


@torch.jit.script
def rms_backward(grad_y, y, l2_norm_inv):
    g = grad_y  # grad * w0
    mean_gy = (g * y).mean(dim=-1, keepdim=True)
    gx = (g - y * mean_gy) * l2_norm_inv
    return gx


class RMSNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        y, l2_norm_inv = rms_foward(x)
        ctx.save_for_backward(y, l2_norm_inv)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        y, l2_norm_inv = ctx.saved_variables
        gx = rms_backward(grad_y, y, l2_norm_inv)
        return gx 


def test_backward(tensor_x, tensor_w, grad_y):
    x0 = tensor_x.detach().clone()
    w0 = tensor_w.detach().clone()
    x0.requires_grad = True
    w0.requires_grad = True
    setattr(w0, 'sequence_parallel', False)

    out = RMSNormFunction.apply(x0.float()) * w0

    # backward
    # 调用的是 2.ii 注册的函数
    # 适配层顺序：dx, dgamma = torch_npu.npu_rms_norm_grad(grad, x, gamma, rstd)

    out.backward(grad_y)

    dx = x0.grad
    dw = w0.grad

    out = out.float()
    dx = dx.float()
    dw = dw.float()
    return dx, dw

class TestRmsNorm(op_test.OpTest):
    def golden_calc(self, in_tensors):
        dy = in_tensors[0]
        x = in_tensors[1]
        rstd = in_tensors[2]
        gamma = in_tensors[3]
        result = test_backward(x, gamma, dy)
        return [result[0]]

    def golden_compare(self, out_tensors, golden_out_tensors):
        return torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)
    
    @op_test.only_910b
    def test_Rms_Norm_Backward(self):
        shape=[4, 1, 8]
        shape_rstd=[4]
        shape_gamma=[8]

        input0 = np.random.uniform(low=0, high=100, size=shape).astype(np.float32)
        input1 = np.random.uniform(low=0, high=100, size=shape).astype(np.float32)
        input2 = torch.rsqrt(torch.from_numpy(input1).pow(2).mean(-1, keepdim=True) + epsilon).numpy()
        input3 = np.random.uniform(low=0, high=100, size=shape_gamma).astype(np.float32)

        # dy x rstd gamma
        in_tensors = [torch.from_numpy(input0).float(), torch.from_numpy(input1).float(),
                      torch.from_numpy(input2).float() , torch.from_numpy(input3).float()]
        out_tensors = [torch.zeros(shape).float(), torch.zeros(shape_gamma).float()]

        self.set_param(OP_NAME, OP_PARAM)
        self.execute(in_tensors, out_tensors)


if __name__ == '__main__':
    unittest.main()
