import logging
import unittest
import numpy as np
import torch
import torch_npu
import op_test

OP_NAME = "NormOperation"
OP_PARAM0 = {"normType": 5, "epsilon": 1e-8}
OP_PARAM1 = {"normType": 5, "epsilon": 1e-5}

HALF_FLOAT_MIN = -5.0
HALF_FLOAT_MAX = 5.0

class TestGatherPreRmsNorm(op_test.OpTest):
    def golden_calc(self, intensors):
        data_type = intensors[0].dtype
        epsilon = self.op_desc["specificParam"]["epsilon"]
        #x, res_in, indices, g
        input0 = torch.tensor(intensors[0]).float()
        input1 = torch.tensor(intensors[1]).float()
        input2 = torch.tensor(intensors[2]).to(torch.int64)
        input3 = torch.tensor(intensors[3]).float()
        # use gather to obtain indexed input_x
        indices_brcb = input2.unsqueeze(1)
        index = indices_brcb.repeat(1, input1.shape[1])
        try:
            res_in_after_gather = input1.gather(0, index)
        except RuntimeError as e:
            print("some index values are out of range", e)
        # rmsnorm compute
        input_sum = input0 + res_in_after_gather
        square_sum = torch.sum(torch.square(input_sum), axis=-1, keepdims=True)
        factor = 1.0 / torch.sqrt(square_sum / input_sum.shape[-1] + epsilon)
        output = input_sum * factor * input3
        return [torch.tensor(output), torch.tensor(input_sum).to(data_type)]

    def golden_compare(self, out_tensors, golden_out_tensors):
        # golden compare for output0
        diff0 = torch.abs(torch.subtract(out_tensors[0].float(), golden_out_tensors[0].float()))
        tensor_max0 = torch.maximum(torch.ones(golden_out_tensors[0].shape, dtype=golden_out_tensors[0].dtype),
                                   torch.abs(golden_out_tensors[0])).float()
        factor0_1, factor0_2 = 2**(-11), 2**(-14)  # precision standard for float dtype
        if torch.any(torch.greater(diff0, factor0_1 * tensor_max0)):
            print("[new standards] output0 accuracy failed")
            return False
        if torch.any(torch.greater(torch.abs(torch.mean(torch.div(diff0, tensor_max0))), factor0_2)):
            print("[new standards] output0 eb failed")
            return False

        # golden compare for output1
        diff1 = torch.abs(torch.subtract(out_tensors[1].float(), golden_out_tensors[1].float()))
        tensor_max1 = torch.maximum(torch.ones(golden_out_tensors[1].shape, dtype=golden_out_tensors[1].dtype),
                                   torch.abs(golden_out_tensors[1])).float()
        factor1_1, factor1_2 = 2**(-8), 2**(-10)  # precision standard for float16 dtype
        if golden_out_tensors[1].dtype == torch.bfloat16: 
            factor1_1, factor1_2 = 2**(-7), 2**(-7)  # precision standard for bfloat16 dtype
        if torch.any(torch.greater(diff1, factor1_1 * tensor_max1)):
            print("[new standards] output1 accuracy failed")
            return False
        if torch.any(torch.greater(torch.abs(torch.mean(torch.div(diff1, tensor_max1))), factor1_2)):
            print("[new standards] output1 eb failed")
            return False
        return True

    @op_test.only_910b
    def test_gatherprermsnorm_f16_small_indices_case0(self):
        res_tokens = 3
        ind_tokens = 4
        hidden_size = 16
        input0 = torch.empty((ind_tokens, hidden_size), dtype=torch.half).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input1 = torch.empty((res_tokens, hidden_size), dtype=torch.half).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input2 = torch.randint(0, res_tokens, (ind_tokens,), dtype=torch.int32)
        input3 = torch.empty((hidden_size), dtype=torch.half).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM1)
        self.execute([torch.tensor(input0), torch.tensor(input1), torch.tensor(input2),
                      torch.tensor(input3)], [torch.zeros((ind_tokens, hidden_size)).float(),
                      torch.zeros((ind_tokens, hidden_size)).half()])

    @op_test.only_910b
    def test_gatherprermsnorm_f16_small_indices_case1(self):
        res_tokens = 97
        ind_tokens = 101
        hidden_size = 7680
        input0 = torch.empty((ind_tokens, hidden_size), dtype=torch.half).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input1 = torch.empty((res_tokens, hidden_size), dtype=torch.half).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input2 = torch.randint(0, res_tokens, (ind_tokens,), dtype=torch.int32)
        input3 = torch.empty((1, hidden_size), dtype=torch.half).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM1)
        self.execute([torch.tensor(input0), torch.tensor(input1), torch.tensor(input2),
                      torch.tensor(input3)], [torch.zeros((ind_tokens, hidden_size)).float(),
                      torch.zeros((ind_tokens, hidden_size)).half()])

    @op_test.only_910b
    def test_gatherprermsnorm_f16_small_indices_case2(self):
        res_tokens = 300
        ind_tokens = 533
        hidden_size = 7168
        input0 = torch.empty((ind_tokens, hidden_size), dtype=torch.half).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input1 = torch.empty((res_tokens, hidden_size), dtype=torch.half).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input2 = torch.randint(0, res_tokens, (ind_tokens,), dtype=torch.int32)
        input3 = torch.empty((1, hidden_size), dtype=torch.half).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0), torch.tensor(input1), torch.tensor(input2),
                      torch.tensor(input3)], [torch.zeros((ind_tokens, hidden_size)).float(),
                      torch.zeros((ind_tokens, hidden_size)).half()])

    @op_test.only_910b
    def test_gatherprermsnorm_f16_large_indices_case(self):
        res_tokens = 99999
        ind_tokens = 240000
        hidden_size = 2560
        input0 = torch.empty((ind_tokens, hidden_size), dtype=torch.half).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input1 = torch.empty((res_tokens, hidden_size), dtype=torch.half).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input2 = torch.randint(0, res_tokens, (ind_tokens,), dtype=torch.int32)
        input3 = torch.empty((hidden_size), dtype=torch.half).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0), torch.tensor(input1), torch.tensor(input2),
                      torch.tensor(input3)], [torch.zeros((ind_tokens, hidden_size)).float(),
                      torch.zeros((ind_tokens, hidden_size)).half()])

    @op_test.only_910b
    def test_gatherprermsnorm_bf16_small_indices_case0(self):
        res_tokens = 16
        ind_tokens = 16
        hidden_size = 7168
        input0 = torch.empty((ind_tokens, hidden_size), dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input1 = torch.empty((res_tokens, hidden_size), dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input2 = torch.randint(0, res_tokens, (ind_tokens,), dtype=torch.int32)
        input3 = torch.empty((1, hidden_size), dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM1)
        self.execute([torch.tensor(input0), torch.tensor(input1), torch.tensor(input2),
                    torch.tensor(input3)], [torch.zeros((ind_tokens, hidden_size)).float(),
                    torch.zeros((ind_tokens, hidden_size)).bfloat16()])

    @op_test.only_910b
    def test_gatherprermsnorm_bf16_small_indices_case1(self):
        res_tokens = 233
        ind_tokens = 334
        hidden_size = 5120
        input0 = torch.empty((ind_tokens, hidden_size), dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input1 = torch.empty((res_tokens, hidden_size), dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        input2 = torch.randint(0, res_tokens, (ind_tokens,), dtype=torch.int32)
        input3 = torch.empty((hidden_size), dtype=torch.bfloat16).uniform_(HALF_FLOAT_MIN, HALF_FLOAT_MAX)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(input0), torch.tensor(input1), torch.tensor(input2),
                    torch.tensor(input3)], [torch.zeros((ind_tokens, hidden_size)).float(),
                    torch.zeros((ind_tokens, hidden_size)).bfloat16()])

if __name__ == '__main__':
    unittest.main()
