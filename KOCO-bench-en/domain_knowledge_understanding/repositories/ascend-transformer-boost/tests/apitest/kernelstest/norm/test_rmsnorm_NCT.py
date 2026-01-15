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
    def test_rmsnorm_bf16_NCT_case0(self):
        oldShape = (1, 1034)
        newShape = (2, 4, 64)
        shape1 = (1, 64)
        oldInput0 = torch.empty(oldShape, dtype=torch.bfloat16).uniform_(0, 16)
        strides = (512, 128, 1)
        offset = 10
        input0 = oldInput0.as_strided(newShape, strides, offset)
        input1 = torch.empty(shape1, dtype=torch.bfloat16).uniform_(0, 16)
        oldOutput = torch.zeros(oldShape).bfloat16()
        output = oldOutput.as_strided(newShape, strides, offset)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).bfloat16(), torch.tensor(input1).bfloat16()],
                     [torch.tensor(output).bfloat16()])
    

    @op_test.only_910b
    def test_rmsnorm_bf16_NCT_case1(self):
        oldShape = (2048*10)
        newShape = (20, 8, 80)
        shape1 = (1, 80)
        oldInput0 = torch.empty(1, 2048*10, dtype=torch.bfloat16).uniform_(0, 16)
        strides = (1024, 128, 1)
        offset = 0
        input0 = oldInput0.as_strided(newShape, strides, offset)
        input1 = torch.empty(shape1, dtype=torch.bfloat16).uniform_(0, 16)
        oldOutput = torch.zeros(oldShape).bfloat16()
        output = oldOutput.as_strided(newShape, strides, offset)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).bfloat16(), torch.tensor(input1).bfloat16()],
                     [torch.tensor(output).bfloat16()])

    @op_test.only_910b
    def test_rmsnorm_bf16_NCT_case2(self):
        oldShape = (4, 4, 128)
        newShape = (4, 4, 80)
        shape1 = (1, 80)
        oldInput0 = torch.empty(oldShape, dtype=torch.bfloat16).uniform_(0, 16)
        strides = (512, 128, 1)
        offset = 0
        input0 = oldInput0.as_strided(newShape, strides, offset)
        input1 = torch.empty(shape1, dtype=torch.bfloat16).uniform_(0, 16)
        oldOutput = torch.zeros(oldShape).bfloat16()
        output = oldOutput.as_strided(newShape, strides, offset)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).bfloat16(), torch.tensor(input1).bfloat16()],
                     [torch.tensor(output).bfloat16()])


    @op_test.only_910b
    def test_rmsnorm_gema_bf16_NCT_case3(self):
        oldShape0 = (32, 5136)
        shape0 = (30, 5120)
        shape1 = (1, 5120)
        oldInput0 = torch.empty(oldShape0, dtype=torch.bfloat16).uniform_(0, 16)
        strides = (5136, 1)
        offset = 2 * 5136
        input0 = oldInput0.as_strided(shape0, strides, offset)
        input1 = torch.empty(shape1, dtype=torch.bfloat16).uniform_(0, 16)
        oldOutput = torch.zeros(oldShape0).bfloat16()
        output = oldOutput.as_strided(shape0, strides, offset)
        self.set_param(OP_NAME, OP_PARAM2)
        self.execute([torch.tensor(input0).bfloat16(), torch.tensor(input1).bfloat16()],
                     [torch.tensor(output).bfloat16()])


    @op_test.only_910b
    def test_rmsnorm_bf16_NCT_case4(self):
        oldShape0 = (32, 16400)
        shape0 = (32, 16384)
        shape1 = (1, 16384)
        oldInput0 = torch.empty(oldShape0, dtype=torch.bfloat16).uniform_(0, 16)
        strides = (16400, 1)
        offset = 0
        input0 = oldInput0.as_strided(shape0, strides, offset)
        input1 = torch.empty(shape1, dtype=torch.bfloat16).uniform_(0, 16)
        oldOutput = torch.zeros(oldShape0).bfloat16()
        output = oldOutput.as_strided(shape0, strides, offset)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).bfloat16(), torch.tensor(input1).bfloat16()],
                     [torch.tensor(output).bfloat16()])
    
    @op_test.only_910b
    def test_rmsnorm_fp16_NCT_case5(self):
        oldShape = (2, 4, 65536)
        newShape = (2, 4, 65504)
        shape1 = (1, 65504)
        oldInput0 = torch.empty(oldShape0, dtype=torch.bfloat16).uniform_(0, 16)
        strides = (262144, 65536, 1)
        offset = 0
        input0 = oldInput0.as_strided(shape0, strides, offset)
        input1 = torch.empty(shape1, dtype=torch.bfloat16).uniform_(0, 16)
        oldOutput = torch.zeros(oldShape0).bfloat16()
        output = oldOutput.as_strided(shape0, strides, offset)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).bfloat16(), torch.tensor(input1).bfloat16()],
                     [torch.tensor(output).bfloat16()])

    @op_test.only_910b
    def test_rmsnorm_fp16_NCT_case0(self):
        oldShape = (1, 1034)
        newShape = (2, 4, 64)
        shape1 = (1, 64)
        oldInput0 = torch.empty(oldShape, dtype=torch.half).uniform_(0, 16)
        strides = (512, 128, 1)
        offset = 10
        input0 = oldInput0.as_strided(newShape, strides, offset)
        input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
        oldOutput = torch.zeros(oldShape).half()
        output = oldOutput.as_strided(newShape, strides, offset)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).half(), torch.tensor(input1).half()],
                     [torch.tensor(output).half()])
    

    @op_test.only_910b
    def test_rmsnorm_fp16_NCT_case1(self):
        oldShape = (2048*10)
        newShape = (20, 8, 80)
        shape1 = (1, 80)
        oldInput0 = torch.empty(1, 2048*10, dtype=torch.half).uniform_(0, 16)
        strides = (1024, 128, 1)
        offset = 0
        input0 = oldInput0.as_strided(newShape, strides, offset)
        input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
        oldOutput = torch.zeros(oldShape).half()
        output = oldOutput.as_strided(newShape, strides, offset)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).half(), torch.tensor(input1).half()],
                     [torch.tensor(output).half()])

    @op_test.only_910b
    def test_rmsnorm_fp16_NCT_case2(self):
        oldShape = (4, 4, 128)
        newShape = (4, 4, 80)
        shape1 = (1, 80)
        oldInput0 = torch.empty(oldShape, dtype=torch.half).uniform_(0, 16)
        strides = (512, 128, 1)
        offset = 0
        input0 = oldInput0.as_strided(newShape, strides, offset)
        input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
        oldOutput = torch.zeros(oldShape).half()
        output = oldOutput.as_strided(newShape, strides, offset)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).half(), torch.tensor(input1).half()],
                     [torch.tensor(output).half()])


    @op_test.only_910b
    def test_rmsnorm_gema_fp16_NCT_case3(self):
        oldShape0 = (32, 5136)
        shape0 = (30, 5120)
        shape1 = (1, 5120)
        oldInput0 = torch.empty(oldShape0, dtype=torch.half).uniform_(0, 16)
        strides = (5136, 1)
        offset = 2 * 5136
        input0 = oldInput0.as_strided(shape0, strides, offset)
        input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
        oldOutput = torch.zeros(oldShape0).half()
        output = oldOutput.as_strided(shape0, strides, offset)
        self.set_param(OP_NAME, OP_PARAM2)
        self.execute([torch.tensor(input0).half(), torch.tensor(input1).half()],
                     [torch.tensor(output).half()])


    @op_test.only_910b
    def test_rmsnorm_fp16_NCT_case4(self):
        oldShape0 = (32, 16400)
        shape0 = (32, 16384)
        shape1 = (1, 16384)
        oldInput0 = torch.empty(oldShape0, dtype=torch.half).uniform_(0, 16)
        strides = (16400, 1)
        offset = 0
        input0 = oldInput0.as_strided(shape0, strides, offset)
        input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
        oldOutput = torch.zeros(oldShape0).half()
        output = oldOutput.as_strided(shape0, strides, offset)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0).half(), torch.tensor(input1).half()],
                     [torch.tensor(output).half()])

    @op_test.only_910b
    def test_rmsnorm_fp16_NCT_case5(self):
        oldShape = (2, 4, 65536)
        newShape = (2, 4, 65504)
        shape1 = (1, 65504)
        oldInput0 = torch.empty(oldShape, dtype=torch.half).uniform_(0, 16)
        strides = (262144, 65536, 1)
        offset = 0
        input0 = oldInput0.as_strided(newShape, strides, offset)
        input1 = torch.empty(shape1, dtype=torch.half).uniform_(0, 16)
        oldOutput = torch.zeros(oldShape).half()
        output = oldOutput.as_strided(newShape, strides, offset)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([input0.half(), input1.half()],
                     [output.half()])

if __name__ == '__main__':
    unittest.main()