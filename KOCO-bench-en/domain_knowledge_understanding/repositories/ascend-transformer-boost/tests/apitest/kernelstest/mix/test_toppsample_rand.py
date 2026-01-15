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
import numpy as np
import torch
import sys
import logging
import torch_npu
import random
import op_test
from tensor_file import read_tensor

OP_NAME = "ToppsampleRandOperation"
OP_PARAM = {}

def generate_random_values(first_dim):
    arrayRand_ = np.arange(0, first_dim)

    rand_seed = arrayRand_.tolist()
    random_values = []
    
    if len(rand_seed) == 1:
        random.seed(rand_seed[0])
        for i in range(first_dim):
            random_values.append(random.random())
    elif len(rand_seed) == 0:
        random.seed(0)
        for i in range(first_dim):
            random_values.append(random.random())
    else:
        for i in range(first_dim):
            if i > len(rand_seed) - 1:
                random.seed(0)
            else:
                random.seed(rand_seed[i])
            random_values.append(random.random())
    
    return torch.tensor(np.array(random_values).reshape(-1, 1).astype(np.float32))

class TestToppSampleRand(op_test.OpTest):
    def golden_calc(self, in_tensors):
        cumsumed = np.array(in_tensors[0].cpu().float()).astype(np.float32)
        topp = np.array(in_tensors[1].cpu().float()).astype(np.float32)
        randnp =np.array(in_tensors[2].cpu().float()).reshape(-1, 1).astype(np.float32)
        bool_judge = (cumsumed < topp)
        res_select_range = np.sum(bool_judge, axis=-1, keepdims=True)
        sum_val = res_select_range - 1 # 要判断是否为0，为0直接返回0
        sum_val[sum_val < 0] = 0
        topp_v = np.take_along_axis(cumsumed, sum_val, axis=-1)
        randnp_new = np.zeros(cumsumed.shape[0]).astype(np.float32)
        for i in range(cumsumed.shape[0]):
            randnp_new[i] = randnp[i % 512]
        randnp_new = randnp_new.reshape(-1, 1)
        topp_v_new = randnp_new[0:cumsumed.shape[0]] * topp_v
        bool_judge_one = (cumsumed <= topp_v_new)
        res_index = np.sum(bool_judge_one, axis=-1, keepdims=True) # 要判断是否为0，为0直接返回0
        res_index[res_index < 0] = 0

        return [torch.tensor(res_index).int(), torch.tensor(res_select_range).int()]

    def golden_compare(self, out_tensors, golden_out_tensors):
        

        if len(out_tensors) != len(golden_out_tensors):
            logging.error(f"Out size [{len(out_tensors)}] not equal to golden size [{len(golden_out_tensors)}]")
            return False
        for i in range(len(out_tensors)):
            precision_standard = torch.abs(out_tensors[i] - golden_out_tensors[i]) <= (1 / 128) * golden_out_tensors[i]
            precision_standard = np.array(precision_standard)
            precision_sum = np.sum(precision_standard)
            if precision_sum != out_tensors[i].shape[0]:
                return False
        return True

    @op_test.skip_310b
    @op_test.skip_910a    
    def test_topp_rand_case65535(self):
        a = 512
        shape1 = (a, 1)
        torch.manual_seed(0)
        probs = torch.randn(a, 65535).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).numpy()
        cumsumed = np.cumsum(probs, axis=-1).astype(np.float16)
        topp = torch.empty(shape1, dtype=torch.half).uniform_(0, 1)
        rand = generate_random_values(a)

        self.set_param(OP_NAME, OP_PARAM)
        self.execute([torch.from_numpy(cumsumed), topp, rand], 
                     [torch.zeros(shape1).int(), torch.zeros(shape1).int()])

    @op_test.skip_310b
    @op_test.skip_910a    
    def test_topp_rand_case254208(self):
        a = 200
        shape1 = (a, 1)
        torch.manual_seed(0)
        probs = torch.randn(a, 254208).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).numpy()
        cumsumed = np.cumsum(probs, axis=-1).astype(np.float16)
        topp = torch.empty(shape1, dtype=torch.half).uniform_(0, 1)
        rand = generate_random_values(a)

        self.set_param(OP_NAME, OP_PARAM)
        self.execute([torch.from_numpy(cumsumed), topp, rand], 
                     [torch.zeros(shape1).int(), torch.zeros(shape1).int()])

    @op_test.skip_310b
    @op_test.skip_910a    
    def test_topp_rand_case65024(self):
        a = 33
        shape1 = (a, 1)
        torch.manual_seed(0)
        probs = torch.randn(a, 65024).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).numpy()
        cumsumed = np.cumsum(probs, axis=-1).astype(np.float16)
        topp = torch.empty(shape1, dtype=torch.half).uniform_(0, 1)
        rand = generate_random_values(a)

        self.set_param(OP_NAME, OP_PARAM)
        self.execute([torch.from_numpy(cumsumed), topp, rand], 
                     [torch.zeros(shape1).int(), torch.zeros(shape1).int()])

    @op_test.skip_310b
    @op_test.skip_910a    
    def test_topp_rand_case128000(self):
        a = 330
        shape1 = (a, 1)
        torch.manual_seed(0)
        probs = torch.randn(a, 128000).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).numpy()
        cumsumed = np.cumsum(probs, axis=-1).astype(np.float16)
        topp = torch.empty(shape1, dtype=torch.half).uniform_(0, 1)
        rand = generate_random_values(a)

        self.set_param(OP_NAME, OP_PARAM)
        self.execute([torch.from_numpy(cumsumed), topp, rand], 
                     [torch.zeros(shape1).int(), torch.zeros(shape1).int()])


    @op_test.only_910b   
    def test_topp_rand_bf16_case92367(self):
        a = 512
        shape1 = (a, 1)
        torch.manual_seed(0)
        probs = torch.randn(a, 92367).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).numpy()
        cumsumed = np.cumsum(probs, axis=-1)
        topp = torch.empty(shape1, dtype=torch.bfloat16).uniform_(0, 1)
        rand = generate_random_values(a)

        self.set_param(OP_NAME, OP_PARAM)
        self.execute([torch.from_numpy(cumsumed).bfloat16(), topp, rand], 
                     [torch.zeros(shape1).int(), torch.zeros(shape1).int()])


    @op_test.only_910b      
    def test_topp_rand_bf16_case254208(self):
        a = 200
        shape1 = (a, 1)
        torch.manual_seed(0)
        probs = torch.randn(a, 254208).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).numpy()
        cumsumed = np.cumsum(probs, axis=-1)
        topp = torch.empty(shape1, dtype=torch.bfloat16).uniform_(0, 1)
        rand = generate_random_values(a)

        self.set_param(OP_NAME, OP_PARAM)
        self.execute([torch.from_numpy(cumsumed).bfloat16(), topp, rand], 
                     [torch.zeros(shape1).int(), torch.zeros(shape1).int()])

    @op_test.only_910b  
    def test_topp_rand_bf16_case65024(self):
        a = 33
        shape1 = (a, 1)
        torch.manual_seed(0)
        probs = torch.randn(a, 65024).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).numpy()
        cumsumed = np.cumsum(probs, axis=-1)
        topp = torch.empty(shape1, dtype=torch.bfloat16).uniform_(0, 1)
        rand = generate_random_values(a)

        self.set_param(OP_NAME, OP_PARAM)
        self.execute([torch.from_numpy(cumsumed).bfloat16(), topp, rand], 
                     [torch.zeros(shape1).int(), torch.zeros(shape1).int()])

    @op_test.only_910b    
    def test_topp_rand_bf16_case128000(self):
        a = 330
        shape1 = (a, 1)
        torch.manual_seed(0)
        probs = torch.randn(a, 128000).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).numpy()
        cumsumed = np.cumsum(probs, axis=-1)
        topp = torch.empty(shape1, dtype=torch.bfloat16).uniform_(0, 1)
        rand = generate_random_values(a)

        self.set_param(OP_NAME, OP_PARAM)
        self.execute([torch.from_numpy(cumsumed).bfloat16(), topp, rand], 
                     [torch.zeros(shape1).int(), torch.zeros(shape1).int()])


if __name__ == '__main__':
    unittest.main()
