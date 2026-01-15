import logging
import unittest
import numpy as np
import torch
import op_test
from tensor_file import read_tensor


OP_NAME = "NormOperation"
OP_PARAM0 = {"normType": 2, "inGamma": True, "epsilon": 1e-8, "precisionMode":0,"gemmaMode":0}
OP_PARAM1 = {"normType": 2, "inGamma": True, "epsilon": 1e-8, "precisionMode":1,"gemmaMode":0}
OP_PARAM2 = {"normType": 2, "inGamma": True, "epsilon": 1e-8, "precisionMode":0,"gemmaMode":1}

class TestRmsNorm(op_test.OpTest):
    def golden_calc(self, intensors):
        data_type = intensors[0].dtype
        epsilon = self.op_desc["specificParam"]["epsilon"]
        input0 = torch.tensor(intensors[0]).float()
        input1 = torch.tensor(intensors[1]).float()
        square_sum = torch.sum(torch.square(input0), axis=-1, keepdims=True)
        factor = 1.0 / torch.sqrt(square_sum / input0.shape[-1] + epsilon)
        if self.op_desc["specificParam"]["precisionMode"]:
            output = (input0 * factor).half() * intensors[1]
        else:
            if self.op_desc["specificParam"]["gemmaMode"] == 1:
                output = input0 * factor * (input1 + 1.0)
            else:
                output = input0 * factor * input1
        return [torch.tensor(output).to(data_type)]

    def golden_compare(self, out_tensors, golden_out_tensors):
        absolute_error = torch.abs(golden_out_tensors[0].float() - out_tensors[0].float())
        relative_error = absolute_error / golden_out_tensors[0].float()
        cnt = torch.sum(torch.gt(relative_error, 0.001)).item()
        return ((cnt / torch.prod(torch.tensor(golden_out_tensors[0].size()))).item() < 0.001)

    @op_test.only_910b
    def test_rmsnorm_bf16_case0(self):
        shape0 = (28, 5, 8192)
        shape1 = (1, 8192)
        input0 = torch.empty(shape0, dtype=torch.bfloat16).uniform_(0, 16)
        input1 = torch.empty(shape1, dtype=torch.bfloat16).uniform_(0, 16)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).bfloat16(), torch.tensor(input1).bfloat16()],
                    [torch.zeros(shape0).bfloat16()])
    
    @op_test.only_910b
    def test_rmsnorm_bf16_case1(self):
        shape0 = (20, 5120)
        shape1 = (1, 5120)
        input0 = torch.empty(shape0, dtype=torch.bfloat16).uniform_(0, 16)
        input1 = torch.empty(shape1, dtype=torch.bfloat16).uniform_(0, 16)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).bfloat16(), torch.tensor(input1).bfloat16()],
                    [torch.zeros(shape0).bfloat16()])
    
    @op_test.only_910b
    def test_rmsnorm_bf16_case2(self):
        shape0 = (32, 16384)
        shape1 = (1, 16384)
        input0 = torch.empty(shape0, dtype=torch.bfloat16).uniform_(0, 16)
        input1 = torch.empty(shape1, dtype=torch.bfloat16).uniform_(0, 16)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).bfloat16(), torch.tensor(input1).bfloat16()],
                    [torch.zeros(shape0).bfloat16()])

    def test_rmsnorm_half_case0(self):
        shape0 = (28, 5, 4096)
        shape1 = (1, 4096)
        input0 = torch.empty(shape0, dtype=torch.half).uniform_(0, 16)
        input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).half(), torch.tensor(input1).half()],
                    [torch.zeros(shape0).half()])

    def test_rmsnorm_half_case1(self):
        shape0 = (20, 5120)
        shape1 = (1, 5120)
        input0 = torch.empty(shape0, dtype=torch.half).uniform_(0, 16)
        input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).half(), torch.tensor(input1).half()],
                    [torch.zeros(shape0).half()])

    def test_rmsnorm_half_case2(self):
        shape0 = (32, 16384)
        shape1 = (1, 16384)
        input0 = torch.empty(shape0, dtype=torch.half).uniform_(0, 16)
        input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).half(), torch.tensor(input1).half()],
                    [torch.zeros(shape0).half()])

    def test_rmsnorm_half_case3(self):
        shape0 = (28, 5, 8192)
        shape1 = (1, 8192)
        input0 = torch.empty(shape0, dtype=torch.half).uniform_(0, 16)
        input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
        self.set_param(OP_NAME, OP_PARAM1)
        self.execute([torch.tensor(input0).half(), torch.tensor(input1).half()],
                    [torch.zeros(shape0).half()])

    def test_rmsnorm_half_case4(self):
        shape0 = (20, 5120)
        shape1 = (1, 5120)
        input0 = torch.empty(shape0, dtype=torch.half).uniform_(0, 16)
        input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
        self.set_param(OP_NAME, OP_PARAM1)
        self.execute([torch.tensor(input0).half(), torch.tensor(input1).half()],
                    [torch.zeros(shape0).half()])

    def test_rmsnorm_half_case5(self):
        shape0 = (32, 16384)
        shape1 = (1, 16384)
        input0 = torch.empty(shape0, dtype=torch.half).uniform_(0, 16)
        input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
        self.set_param(OP_NAME, OP_PARAM1)
        self.execute([torch.tensor(input0).half(), torch.tensor(input1).half()],
                    [torch.zeros(shape0).half()])
    
    @op_test.only_910b
    def test_rmsnorm_gema_case1(self):
        shape0 = (20, 5120)
        shape1 = (1, 5120)
        input0 = torch.empty(shape0, dtype=torch.bfloat16).uniform_(0, 16)
        input1 = torch.empty(shape1, dtype=torch.bfloat16).uniform_(0, 16)
        self.set_param(OP_NAME, OP_PARAM2)
        self.execute([torch.tensor(input0).bfloat16(), torch.tensor(input1).bfloat16()],
                    [torch.zeros(shape0).bfloat16()])
    
    @op_test.only_910b
    def test_rmsnorm_gema_case2(self):
        shape0 = (32, 16384)
        shape1 = (1, 16384)
        input0 = torch.empty(shape0, dtype=torch.bfloat16).uniform_(0, 16)
        input1 = torch.empty(shape1, dtype=torch.bfloat16).uniform_(0, 16)
        self.set_param(OP_NAME, OP_PARAM2)
        self.execute([torch.tensor(input0).bfloat16(), torch.tensor(input1).bfloat16()],
                    [torch.zeros(shape0).bfloat16()])

    def test_rmsnorm_gema_case3(self):
        shape0 = (20, 5120)
        shape1 = (1, 5120)
        input0 = torch.empty(shape0, dtype=torch.half).uniform_(0, 16)
        input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
        self.set_param(OP_NAME, OP_PARAM2)
        self.execute([torch.tensor(input0).half(), torch.tensor(input1).half()],
                    [torch.zeros(shape0).half()])

    def test_rmsnorm_gema_case4(self):
        shape0 = (32, 16384)
        shape1 = (1, 16384)
        input0 = torch.empty(shape0, dtype=torch.half).uniform_(0, 16)
        input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
        self.set_param(OP_NAME, OP_PARAM2)
        self.execute([torch.tensor(input0).half(), torch.tensor(input1).half()],
                    [torch.zeros(shape0).half()])

if __name__ == '__main__':
    unittest.main()