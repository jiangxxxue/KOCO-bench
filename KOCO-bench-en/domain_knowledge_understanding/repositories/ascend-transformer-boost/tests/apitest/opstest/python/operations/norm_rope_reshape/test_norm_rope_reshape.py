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
import operation_test
OP_NAME = "NormRopeReshapeOperation"

class TestGatherPreRmsNormOperation(operation_test.OperationTest):
    def rmsnorm_golden(self, x, gamma, epsilon):
        x_float32 = x.astype(np.float32)
        square_sum = np.sum(np.square(x_float32), axis=-1, keepdims=True)
        rms = 1.0 / np.sqrt(square_sum / x_float32.shape[-1] + epsilon)
        gamma_float32 = gamma.astype(np.float32)
        rms_norm = rms * x_float32 * gamma_float32
        result = rms_norm.astype(np.float16)
        return result
 
    def rotate_half(self, k_temp):
        first_half, second_half = np.split(k_temp, 2, axis=1)
        processed_k_split = np.concatenate((-second_half, first_half), axis=1)
        return processed_k_split
    
    def rope_golden(self, key_rope, sin, cos):
        key_rope = key_rope.astype(np.float32)
        sin = sin.astype(np.float32)
        cos = cos.astype(np.float32)
        res = key_rope*cos + self.rotate_half(key_rope)*sin
        result = res.astype(np.float16)
        return result
    def rac_golden(self, block_size, key_rac, slot_mapping, keycacheout_golden):
        for i, slot in enumerate(slot_mapping):
            if slot < 0:
                continue
            block_index = slot // block_size
            block_offset = slot % block_size
            token_key = key_rac[i]
        
            keycacheout_golden[block_index][block_offset] = token_key
        return keycacheout_golden
 
    def golden_calc(self, in_tensors):
        data_type = in_tensors[6].dtype
        eps = 1e-5
        # precision_mode = 0
        # rotary_coeff = 2
 
        input0 = np.array(in_tensors[0].cpu()).astype(np.float16)  # x
        input1 = np.array(in_tensors[1].cpu()).astype(np.float16)  # gamma
        input2 = np.array(in_tensors[2].cpu()).astype(np.float16)  # keyRope
        input3 = np.array(in_tensors[3].cpu()).astype(np.float16)  # cos
        input4 = np.array(in_tensors[4].cpu()).astype(np.float16)  # sin
        input5 = np.array(in_tensors[5].cpu()).astype(np.int32)    # slotMapping
        input6 = np.array(in_tensors[6].cpu()).astype(np.float16)  # keycachein
 
        keycacheout_golden = np.zeros_like(input6)                 # keycacheout
 
        rmsnorm_output = self.rmsnorm_golden(input0, input1, eps)
        rope_output = self.rope_golden(input2, input4, input3)
        rope_reshape = rope_output.reshape(input4.shape[0], -1, input4.shape[-1])
        key_rac = np.concatenate((rmsnorm_output, rope_reshape), axis=-1)
        output = self.rac_golden(input6.shape[1], key_rac, input5, keycacheout_golden)
        return [torch.tensor(output).to(data_type)]
 
 
    def golden_compare(self, out_tensors, golden_out_tensors):
        temp = out_tensors.cpu()
        golden_out = torch.tensor(golden_out_tensors)
        diff = torch.abs(torch.subtract(temp.float(), golden_out.float()))
        tensor_max = torch.maximum(torch.ones(golden_out.shape, dtype=golden_out.dtype), torch.abs(golden_out)).float()
        factor1, factor2 = 2**(-8), 2**(-10)  # precision standard for float16 dtype
        if torch.any(torch.greater(diff, factor1 * tensor_max)):
            print("[new standards] output accuracy failed")
            return False
        if torch.any(torch.greater(torch.abs(torch.mean(torch.div(diff, tensor_max))), factor2)):
            print("[new standards] output eb failed")
            return False
        return True
 
    def test1(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        input_token_num = 64
        head_num = 1
        head_dim_x = 512
        head_dim_rope = 64
        head_dim_RAC = 576
        rope_hidden_size = head_num * head_dim_rope
        block_num = 192
        block_size = 128
 
        x = torch.rand((input_token_num, head_num, head_dim_x), dtype=torch.float16)
        gamma = torch.rand((head_dim_x,), dtype=torch.float16)
        keyRope = torch.rand((input_token_num, rope_hidden_size), dtype=torch.float16)
        cos = torch.rand((input_token_num, head_dim_rope), dtype=torch.float16)
        sin = torch.rand((input_token_num, head_dim_rope), dtype=torch.float16)
        slotMapping = torch.tensor(np.random.choice(100, input_token_num, replace=False), dtype=torch.int32)
        keycachein = torch.zeros((block_num, block_size, head_num, head_dim_RAC), dtype=torch.float16)

        in_tensors = [x.npu(), gamma.npu(), keyRope.npu(), cos.npu(), sin.npu(), slotMapping.npu(), keycachein.npu()]
 
        self.execute(OP_NAME, {"precision_mode": 0, "epsilon": 1e-5, "rotary_coeff": 2},
                     in_tensors)
 
if __name__ == '__main__':
    unittest.main()