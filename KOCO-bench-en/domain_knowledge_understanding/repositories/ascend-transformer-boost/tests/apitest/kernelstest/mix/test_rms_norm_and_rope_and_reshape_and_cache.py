# 
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
 
OP_NAME = "RmsNormAndRopeAndReshapeAndCacheOperation"
 
class TestRmsNormAndRopeAndReshapeAndCacheOperation(op_test.OpTest):
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
        eps = self.op_desc["specificParam"]["epsilon"]
        precision_mode = self.op_desc["specificParam"]["precisionMode"]
        rotary_coeff = self.op_desc["specificParam"]["rotaryCoeff"]
 
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
        return [self.rac_golden(input6.shape[1], key_rac, input5, keycacheout_golden)]
 
    def golden_compare(self, out_tensors, golden_out_tensors):
        golden_out = torch.tensor(golden_out_tensors[0])
        diff = torch.abs(torch.subtract(out_tensors[0].float(), golden_out.float()))
        # import pdb;pdb.set_trace()
        tensor_max = torch.maximum(torch.ones(golden_out.shape, dtype=golden_out.dtype), torch.abs(golden_out)).float()
        factor1, factor2 = 2**(-8), 2**(-10)  # precision standard for float16 dtype
        if torch.any(torch.greater(diff, factor1 * tensor_max)):
            print("[new standards] output accuracy failed")
            return False
        if torch.any(torch.greater(torch.abs(torch.mean(torch.div(diff, tensor_max))), factor2)):
            print("[new standards] output eb failed")
            return False
        return True
 
    @op_test.only_910b
    def test_rms_norm_and_rope_and_reshape_and_cache(self):
        for i in range(100):
            input_token_num = 65536
            head_num = 1
            head_dim_x = 512
            head_dim_rope = 64
            head_dim_RAC = 576
            rope_hidden_size = head_num * head_dim_rope
            block_num = 193
            block_size = 512
    
            precision_mode = 0
            epsilon = 1e-5
            rotary_coeff = 2

            OP_PARAM = {"precisionMode": precision_mode, "epsilon": epsilon, "rotaryCoeff": rotary_coeff}
    
            x = np.random.uniform(-100, 100, [input_token_num, head_num, head_dim_x]).astype(np.float16)
            gamma = np.random.uniform(-100, 100, [head_dim_x]).astype(np.float16)
            keyRope = np.random.uniform(-100, 100, [input_token_num, rope_hidden_size]).astype(np.float16)
            cos = np.random.uniform(-100, 100, [input_token_num, head_dim_rope]).astype(np.float16)
            sin = np.random.uniform(-100, 100, [input_token_num, head_dim_rope]).astype(np.float16)
            slotMapping = np.random.choice(input_token_num+100, input_token_num, replace=False).astype(np.int32)
            keycachein = np.zeros((block_num, block_size, head_num, head_dim_RAC), dtype=np.float16)
            keycacheout = np.zeros((block_num, block_size, head_num, head_dim_RAC), dtype=np.float16)
            self.set_param(OP_NAME, OP_PARAM)
            self.set_input_formats([self.format_nd] * 8)
            self.set_output_formats([self.format_nd] * 1)

            self.execute([torch.tensor(x).half(), torch.tensor(gamma).half(), torch.tensor(keyRope).half(),
                    torch.tensor(cos).half(), torch.tensor(sin).half(), torch.tensor(slotMapping).int(),
                    torch.tensor(keycachein).half()], [torch.zeros(keycacheout.shape).half()])
 
if __name__ == '__main__':
    unittest.main()