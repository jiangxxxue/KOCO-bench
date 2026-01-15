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
# np.set_printoptions(threshold=np.inf)

OP_NAME = "ToppsampleOperation"

# 生成一个0到255的数组
arrayRand_ = np.arange(0, 256)

# 将数组转换为列表
listRand_ = arrayRand_.tolist()
OP_PARAM0 = {'randSeed':listRand_}

rand = [0.840188, 0.840188, 0.700976, 0.56138, 0.916458, 0.274746, 0.135439, 0.486904, 0.352761, 0.206965, 0.565811, 
0.926345, 0.7856, 0.632643, 0.999498, 0.354973, 0.215437, 0.571794, 0.929073, 0.290233, 0.148812, 0.5059,
0.865027, 0.727582, 0.0877142, 0.939356, 0.795546, 0.659833, 0.517155, 0.875989, 0.229967, 0.592119, 0.449676,
 0.307949, 0.16897, 0.52449, 0.38126, 0.239412, 0.598226, 0.957148, 0.819423, 0.670246, 0.0334699, 0.392149,
  0.749737, 0.608924, 0.469696, 0.82568, 0.683866, 0.0418116, 0.894321, 0.76058, 0.615916, 0.977703, 0.329538,
   0.193523, 0.0520213, 0.911151, 0.76655, 0.126698, 0.985316, 0.843173, 0.699551, 0.557263, 0.419794, 0.278591, 
   0.133239, 0.488706, 0.854142, 0.710721, 0.570227, 0.424855, 0.283544, 0.137892, 0.998458, 0.356983, 0.218768, 
   0.0764689, 0.431105, 0.293164, 0.650124, 0.510737, 0.864335, 0.725284, 0.0856771, 0.942078, 0.798407, 0.163865, 
   0.0179958, 0.879517, 0.736922, 0.589669, 0.954449, 0.811225, 0.17209, 0.525153, 0.386471, 0.74499, 0.100015, 
   0.460092, 0.315598, 0.67758, 0.0377366, 0.390759, 0.250252, 0.609578, 0.971368, 0.329668, 0.686308, 0.542572, 
   0.404475, 0.758984, 0.119068, 0.474643, 0.8359, 0.196087, 0.053463, 0.910053, 0.774065, 0.12442, 0.488597, 
   0.847162, 0.700283, 0.0600514, 0.9206, 0.272758, 0.134877, 0.491615, 0.352014, 0.712666, 0.0740151, 0.927864, 
   0.285351, 0.650377, 0.505641, 0.863653, 0.223241, 0.079943, 0.437205, 0.298454, 0.154515, 0.011271, 0.872241, 
   0.730818, 0.0833345, 0.945243, 0.300876, 0.164505, 0.0154912, 0.382255, 0.237096, 0.0990944, 0.456223, 0.317408,
    0.672289, 0.533509, 0.888445, 0.747989, 0.606889, 0.961031, 0.321871, 0.178723, 0.540503, 0.894323, 0.253086, 
    0.614068, 0.471639, 0.329893, 0.188979, 0.0470938, 0.406147, 0.267296, 0.117794, 0.478868, 0.84048, 0.202001, 
    0.0554726, 0.414112, 0.770353, 0.132036, 0.493066, 0.347837, 0.703756, 0.56828, 0.422414, 0.276724, 0.639273, 
    0.996807, 0.35981, 0.212645, 0.0706023, 0.930278, 0.294288, 0.650826, 0.00552087, 0.365868, 0.223861, 0.0812373, 
    0.438431, 0.804498, 0.154288, 0.0176732, 0.870124, 0.729689, 0.589615, 0.946746, 0.807445, 0.66769, 0.525735, 
    0.378417, 0.739382, 0.598199, 0.960287, 0.314484, 0.679149, 0.532567, 0.892711, 0.247067, 0.605991, 0.96133, 
    0.324312, 0.182571, 0.0385279, 0.398662, 0.759372, 0.617733, 0.47376, 0.333983, 0.689455, 0.0539994, 0.908891, 
    0.261659, 0.620741, 0.986825, 0.839882, 0.200503, 0.0623456, 0.414299, 0.276184, 0.134478, 0.995795, 0.350289, 
    0.70555, 0.567943, 0.924842, 0.284541, 0.640393, 0.999927, 0.359922, 0.717201, 0.568329, 0.932895, 0.794168, 
    0.153389, 0.506864, 0.364243, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 
    0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 
    0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 
    0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 
    0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 
    0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 
    0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 
    0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 
    0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 
    0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 
    0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 
    0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 
    0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 
    0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 
    0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 
    0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 
    0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 
    0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 
    0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 
    0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 
    0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 
    0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 
    0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 
    0.840188, 0.840188, 0.840188, 0.840188, 0.840188, 0.840188
]
class TestToppSample(op_test.OpTest):
    def golden_calc(self, in_tensors):
        cumsumed = np.array(in_tensors[0].cpu().float()).astype(np.float32)
        topp = np.array(in_tensors[1].cpu().float()).astype(np.float32)
        randnp =np.array(rand).reshape(-1, 1).astype(np.float32)
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
    def test_topp_case65535(self):
        shape1 = (512, 1)
        torch.manual_seed(0)
        probs = torch.randn(512, 92367).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).numpy()
        cumsumed = np.cumsum(probs, axis=-1).astype(np.float16)
        topp = torch.empty(shape1, dtype=torch.half).uniform_(0, 1)

        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.from_numpy(cumsumed), topp], 
                     [torch.zeros(shape1).int(), torch.zeros(shape1).int()])

    @op_test.skip_310b
    @op_test.skip_910a        
    def test_topp_case254208(self):
        shape1 = (200, 1)
        torch.manual_seed(0)
        probs = torch.randn(200, 254208).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).numpy()
        cumsumed = np.cumsum(probs, axis=-1).astype(np.float16)
        topp = torch.empty(shape1, dtype=torch.half).uniform_(0, 1)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.from_numpy(cumsumed), topp], 
                     [torch.zeros(shape1).int(), torch.zeros(shape1).int()])

    @op_test.skip_310b
    @op_test.skip_910a        
    def test_topp_case65024(self):
        shape1 = (33, 1)
        torch.manual_seed(0)
        probs = torch.randn(33, 65024).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).numpy()
        cumsumed = np.cumsum(probs, axis=-1).astype(np.float16)

        topp = torch.empty(shape1, dtype=torch.half).uniform_(0, 1)

        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.from_numpy(cumsumed), topp], 
                     [torch.zeros(shape1).int(), torch.zeros(shape1).int()])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_topp_case128000(self):
        shape1 = (330, 1)
        torch.manual_seed(0)
        probs = torch.randn(330, 128000).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).numpy()
        cumsumed = np.cumsum(probs, axis=-1).astype(np.float16)

        topp = torch.empty(shape1, dtype=torch.half).uniform_(0, 1)

        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.from_numpy(cumsumed), topp], 
                     [torch.zeros(shape1).int(), torch.zeros(shape1).int()])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_topp_case32000(self):
        shape1 = (330, 1)
        torch.manual_seed(0)
        probs = torch.randn(330, 32000).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).numpy()
        cumsumed = np.cumsum(probs, axis=-1).astype(np.float16)

        topp = torch.empty(shape1, dtype=torch.half).uniform_(0, 1)

        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.from_numpy(cumsumed), topp], 
                     [torch.zeros(shape1).int(), torch.zeros(shape1).int()])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_topp_case158464(self):
        shape1 = (330, 1)
        torch.manual_seed(0)
        probs = torch.randn(330, 158464).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).numpy()
        cumsumed = np.cumsum(probs, axis=-1).astype(np.float16)

        topp = torch.empty(shape1, dtype=torch.half).uniform_(0, 1)

        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.from_numpy(cumsumed), topp], 
                     [torch.zeros(shape1).int(), torch.zeros(shape1).int()])

    @op_test.skip_310b
    @op_test.skip_910a                     
    def test_topp_case100352(self):
        shape1 = (512, 1)
        torch.manual_seed(0)
        probs = torch.randn(512, 100352).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).numpy()
        cumsumed = np.cumsum(probs, axis=-1).astype(np.float16)

        topp = torch.empty(shape1, dtype=torch.half).uniform_(0, 1)

        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.from_numpy(cumsumed), topp], 
                     [torch.zeros(shape1).int(), torch.zeros(shape1).int()])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_topp_case_bug_fix_182744(self):
        shape1 = (249, 1)
        torch.manual_seed(0)
        probs = torch.randn(249, 182744).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).numpy()
        cumsumed = np.cumsum(probs, axis=-1).astype(np.float16)
        topp = torch.empty(shape1, dtype=torch.half).uniform_(0, 1)

        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.from_numpy(cumsumed), topp], 
                     [torch.zeros(shape1).int(), torch.zeros(shape1).int()])

    @op_test.only_910b  
    def test_topp_bf16_case65535(self):
        shape1 = (512, 1)
        torch.manual_seed(0)
        probs = torch.randn(512, 92367).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).numpy()
        cumsumed = np.cumsum(probs, axis=-1)
        topp = torch.empty(shape1, dtype=torch.bfloat16).uniform_(0, 1)

        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.from_numpy(cumsumed).bfloat16(), topp], 
                     [torch.zeros(shape1).int(), torch.zeros(shape1).int()])

    @op_test.only_910b      
    def test_topp_bf16_case254208(self):
        shape1 = (200, 1)
        torch.manual_seed(0)
        probs = torch.randn(200, 254208).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).numpy()
        cumsumed = np.cumsum(probs, axis=-1)
        topp = torch.empty(shape1, dtype=torch.bfloat16).uniform_(0, 1)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.from_numpy(cumsumed).bfloat16(), topp], 
                     [torch.zeros(shape1).int(), torch.zeros(shape1).int()])

    @op_test.only_910b      
    def test_topp_bf16_case65024(self):
        shape1 = (33, 1)
        torch.manual_seed(0)
        probs = torch.randn(33, 65024).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).numpy()
        cumsumed = np.cumsum(probs, axis=-1)
        topp = torch.empty(shape1, dtype=torch.bfloat16).uniform_(0, 1)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.from_numpy(cumsumed).bfloat16(), topp], 
                     [torch.zeros(shape1).int(), torch.zeros(shape1).int()])

    @op_test.only_910b
    def test_topp_bf16_case128000(self):
        shape1 = (330, 1)
        torch.manual_seed(0)
        probs = torch.randn(330, 128000).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).numpy()
        cumsumed = np.cumsum(probs, axis=-1)
        topp = torch.empty(shape1, dtype=torch.bfloat16).uniform_(0, 1)

        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.from_numpy(cumsumed).bfloat16(), topp], 
                     [torch.zeros(shape1).int(), torch.zeros(shape1).int()])

    @op_test.only_910b
    def test_topp_bf16_case32000(self):
        shape1 = (330, 1)
        torch.manual_seed(0)
        probs = torch.randn(330, 32000).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).numpy()
        cumsumed = np.cumsum(probs, axis=-1)
        topp = torch.empty(shape1, dtype=torch.bfloat16).uniform_(0, 1)

        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.from_numpy(cumsumed).bfloat16(), topp], 
                     [torch.zeros(shape1).int(), torch.zeros(shape1).int()])

    @op_test.only_910b
    def test_topp_bf16_case158464(self):
        shape1 = (330, 1)
        torch.manual_seed(0)
        probs = torch.randn(330, 158464).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).numpy()
        cumsumed = np.cumsum(probs, axis=-1)
        topp = torch.empty(shape1, dtype=torch.bfloat16).uniform_(0, 1)

        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.from_numpy(cumsumed).bfloat16(), topp], 
                     [torch.zeros(shape1).int(), torch.zeros(shape1).int()])

    @op_test.only_910b                   
    def test_topp_bf16_case100352(self):
        shape1 = (512, 1)
        torch.manual_seed(0)
        probs = torch.randn(512, 100352).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).numpy()
        cumsumed = np.cumsum(probs, axis=-1)
        topp = torch.empty(shape1, dtype=torch.bfloat16).uniform_(0, 1)

        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.from_numpy(cumsumed).bfloat16(), topp], 
                     [torch.zeros(shape1).int(), torch.zeros(shape1).int()])

    @op_test.only_910b
    def test_topp_case_bf16_bug_fix_182744(self):
        shape1 = (249, 1)
        torch.manual_seed(0)
        probs = torch.randn(249, 182744).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs).numpy()
        cumsumed = np.cumsum(probs, axis=-1)
        topp = torch.empty(shape1, dtype=torch.bfloat16).uniform_(0, 1)

        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.from_numpy(cumsumed).bfloat16(), topp], 
                     [torch.zeros(shape1).int(), torch.zeros(shape1).int()])


if __name__ == '__main__':
    unittest.main()
