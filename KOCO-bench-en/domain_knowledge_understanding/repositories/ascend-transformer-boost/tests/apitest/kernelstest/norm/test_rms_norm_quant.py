import op_test
import logging
import unittest
import numpy as np
import torch


OP_NAME = "NormOperation"
OP_PARAM0 = {"normType": 2, "inGamma": True, "inBeta": True, "epsilon": 1e-8}

HALF_FLOAT_MIN = -2.0
HALF_FLOAT_MAX = 2.0
QUANTMAX = 127
QUANTMIN = -128
epsi = 1e-4


class TestRmsNormQuant(op_test.OpTest):
    def golden_calc(self, intensors):
        out_shape = intensors[0].shape
        epsilon = self.op_desc["specificParam"]["epsilon"]
        scale = 1.0 / intensors[3].item()
        offset = intensors[4].item()
        inputScale = torch.tensor(scale, dtype=torch.float)
        inputOffset = torch.tensor(offset, dtype=torch.float)
        input0 = torch.tensor(intensors[0]).float()
        input1 = torch.tensor(intensors[1]).float()
        input2 = torch.tensor(intensors[2]).float()
        square_sum = torch.sum(torch.square(input0), axis=-1, keepdims=True)
        factor = 1.0 / torch.sqrt(square_sum / out_shape[-1] + epsilon)
        output = input0 * factor * input1
        if torch.numel(input2) != 0:
            output = (output + input2) * inputScale + inputOffset
        else:
            output = output * inputScale + inputOffset
        output = torch.round(output)
        output = torch.tensor(output).to(torch.float16)
        output = torch.min(output, torch.tensor(QUANTMAX, dtype=torch.half))
        output = torch.max(output, torch.tensor(QUANTMIN, dtype=torch.half))
        return [torch.tensor(output).to(torch.int8)]

    def golden_compare(self, out_tensors, golden_out_tensors):
        expect = golden_out_tensors[0]
        actual = out_tensors[0]
        abs_diff = torch.abs(expect-actual)

        return not (abs_diff > 1).nonzero().size(0) > 0

    def test_rmsnormquant_f16_case0(self):
        '''
            基础场景
            尾块开放非32位对齐
            shape: (151, 31073) / (31073,)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        shape0 = (2, 128, 8704)
        shape1 = (1, 8704)
        input0 = torch.empty(shape0).normal_(0, HALF_FLOAT_MAX).clamp(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        input1 = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        input2 = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        input3 = torch.empty((1,)).uniform_(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        if input3[0] == 0:
            input3 = torch.scalar_tensor(epsi).to(torch.float16)
        input4 = torch.randint(-128, 128, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([input0, input1, input2, input3, input4],
                     [torch.zeros(shape0, dtype=torch.int8)])

    def test_rmsnormquant_f16_case1(self):
        '''
            基础场景
            开放尾块32位对齐场景
            shape: (50, 12288) / (1, 12288)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        shape0 = (50, 12288)
        shape1 = (1, 12288)
        input0 = torch.empty(shape0).normal_(0, HALF_FLOAT_MAX).clamp(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        input1 = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        input2 = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        input3 = torch.empty((1,)).uniform_(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        if input3[0] == 0:
            input3 = torch.scalar_tensor(epsi).to(torch.float16)
        input4 = torch.randint(-128, 128, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([input0, input1, input2, input3, input4],
                     [torch.zeros(shape0, dtype=torch.int8)])

    def test_rmsnormquant_f16_case2(self):
        '''
            基础场景
            开放尾块非32位对齐场景
            shape: (50, 19201) / (1, 19201)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        shape0 = (50, 19201)
        shape1 = (1, 19201)
        input0 = torch.empty(shape0).normal_(0, HALF_FLOAT_MAX).clamp(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        input1 = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        input2 = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        input3 = torch.empty((1,)).uniform_(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        if input3[0] == 0:
            input3 = torch.scalar_tensor(epsi).to(torch.float16)
        input4 = torch.randint(-128, 128, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([input0, input1, input2, input3, input4],
                     [torch.zeros(shape0, dtype=torch.int8)])

    def test_rmsnormquant_f16_case3(self):
        '''
            基础场景
            开放尾块非32位对齐场景,
            shape: (50, 12288) / (1, 12288)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        shape0 = (50, 12288)
        shape1 = (1, 12288)
        input0 = torch.empty(shape0).normal_(0, HALF_FLOAT_MAX).clamp(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        input1 = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        input2 = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        input3 = torch.empty((1,)).uniform_(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
        if input3[0] == 0:
            input3 = torch.scalar_tensor(epsi).to(torch.float16)
        input4 = torch.randint(-128, 128, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([input0, input1, input2, input3, input4],
                     [torch.zeros(shape0, dtype=torch.int8)])
    
    @op_test.only_910b
    def test_rmsnormquant_bf16_case0(self):
        '''
            基础场景
            非32位对齐场景
            shape: (65, 31071) / (31071,)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        shape0 = (65, 31071)
        shape1 = (31071,)
        input0 = torch.empty(shape0, dtype=torch.bfloat16).uniform_(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input1 = torch.empty(shape1, dtype=torch.bfloat16).uniform_(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input2 = torch.empty(shape1, dtype=torch.bfloat16).uniform_(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input3 = torch.empty((1,)).uniform_(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.bfloat16)
        if input3[0] == 0:
            input3 = torch.scalar_tensor(epsi).to(torch.bfloat16)
        input4 = torch.randint(-128, 128, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([input0, input1, input2, input3, input4],
                     [torch.zeros(shape0, dtype=torch.int8)])
    
    @op_test.only_910b
    def test_rmsnormquant_bf16_case1(self):
        '''
            基础场景
            32位对齐场景
            shape: (28, 5, 8192) / (8192,)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        shape0 = (5, 28, 8192)
        shape1 = (8192,)
        input0 = torch.empty(shape0, dtype=torch.bfloat16).uniform_(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input1 = torch.empty(shape1, dtype=torch.bfloat16).uniform_(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input2 = torch.empty(shape1, dtype=torch.bfloat16).uniform_(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input3 = torch.empty((1,)).uniform_(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.bfloat16)
        if input3[0] == 0:
            input3 = torch.scalar_tensor(epsi).to(torch.bfloat16)
        input4 = torch.randint(-128, 128, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([input0, input1, input2, input3, input4],
                     [torch.zeros(shape0, dtype=torch.int8)])
    
    @op_test.only_910b
    def test_rmsnormquant_bf16_case2(self):
        '''
            基础场景
            开放尾块 对齐32位场景
            shape: (7, 25600) / (1, 25600)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        shape0 = (253, 25600)
        shape1 = (1, 25600)
        input0 = torch.empty(shape0).normal_(0, HALF_FLOAT_MAX).clamp(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.bfloat16)
        input1 = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.bfloat16)
        input2 = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.bfloat16)
        input3 = torch.empty((1,)).uniform_(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.bfloat16)
        if input3[0] == 0:
            input3 = torch.scalar_tensor(epsi).to(torch.bfloat16)
        input4 = torch.randint(-128, 128, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([input0, input1, input2, input3, input4],
                     [torch.zeros(shape0, dtype=torch.int8)])
    
    @op_test.only_910b
    def test_rmsnormquant_bf16_case3(self):
        '''
            基础场景
            非32位对齐场景
            shape: (28, 5, 15315) / (15315,)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        shape0 = (28, 5, 15315)
        shape1 = (15315,)
        input0 = torch.empty(shape0, dtype=torch.bfloat16).uniform_(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input1 = torch.empty(shape1, dtype=torch.bfloat16).uniform_(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input2 = torch.empty(shape1, dtype=torch.bfloat16).uniform_(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input3 = torch.empty((1,)).uniform_(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.bfloat16)
        if input3[0] == 0:
            input3 = torch.scalar_tensor(epsi).to(torch.bfloat16)
        input4 = torch.randint(-128, 128, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([input0, input1, input2, input3, input4],
                     [torch.zeros(shape0, dtype=torch.int8)])

    def test_rmsnormquant_f16_caserandd(self):
        '''
            随机大小场景
            shape0: (x, y)
            shape1: (1, y)
            x: [1, 256), y:[1, 8191)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        for i in range(5):
            Shape0 = torch.randint(1, 256, (1,))
            randVal = torch.randint(32, 8191, (1,))
            shape0 = (Shape0[0], randVal[0])
            shape1 = (1, randVal[0])
            logging.info(f"shape 0 is {shape0}, shape 1 is {shape1}")
            input0 = torch.empty(shape0).normal_(0, HALF_FLOAT_MAX).clamp(
                HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
            input1 = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(
                HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
            input2 = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(
                HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
            input3 = torch.empty((1,)).uniform_(
                HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.float16)
            if input3[0] == 0:
                input3 = torch.scalar_tensor(epsi).to(torch.float16)
            input4 = torch.randint(-128, 128, (1,), dtype=torch.int8)
            print(input3)
            self.set_param(OP_NAME, OP_PARAM0)
            self.execute([input0, input1, input2, input3, input4],
                         [torch.zeros(shape0, dtype=torch.int8)])
    
    @op_test.only_910b
    def test_rmsnormquant_bf16_caserandd(self):
        '''
            随机大小场景
            shape0: (x, y)
            shape1: (1, y)
            x: [1, 256), y:[1, 8191)
            取值范围
            fp16: [-2, 2]; int8: [-128, 128)
        '''
        Shape0 = torch.randint(1, 256, (1,))
        randVal = torch.randint(1, 8191, (1,))
        shape0 = (Shape0[0], randVal[0])
        shape1 = (1, randVal[0])
        logging.info(f"shape 0 is {shape0}, shape 1 is {shape1}")
        input0 = torch.empty(shape0).normal_(0, HALF_FLOAT_MAX).clamp(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.bfloat16)
        input1 = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.bfloat16)
        input2 = torch.empty(shape1).normal_(0, HALF_FLOAT_MAX).clamp(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.bfloat16)
        input3 = torch.empty((1,)).uniform_(
            HALF_FLOAT_MIN, HALF_FLOAT_MAX).to(torch.bfloat16)
        if input3[0] == 0:
            input3 = torch.scalar_tensor(epsi).to(torch.bfloat16)
        input4 = torch.randint(-128, 128, (1,), dtype=torch.int8)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([input0, input1, input2, input3, input4],
                     [torch.zeros(shape0, dtype=torch.int8)])


if __name__ == '__main__':
    unittest.main()
