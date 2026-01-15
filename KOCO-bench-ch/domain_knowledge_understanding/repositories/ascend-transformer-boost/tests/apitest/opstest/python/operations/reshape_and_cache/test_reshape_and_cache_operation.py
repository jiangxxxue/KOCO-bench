#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import json
import math
import os
import random
import sys
import unittest
 
import numpy as np
import torch
import torch_npu
 
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402
 
MAX_SEQ_LEN = 1024

def generate_data(
        num_tokens=2,
        num_heads=32,
        head_size=128,
        block_size=128,
        num_blocks=512,
 
):
    soc_version = operation_test.get_soc_version()
    dtype = "float16"
    head_size_k = np.random.randint(1, 256)
    head_size_v = np.random.randint(1, 256)
    key = np.random.uniform(-1.0, 1.0, size=(num_tokens, num_heads, head_size_k)).astype(dtype)
    value = np.random.uniform(-1.0, 1.0, size=(num_tokens, num_heads, head_size_v)).astype(dtype)
 
    num_slots = block_size * num_blocks
    slot_list = random.sample(range(num_slots), num_tokens)
    slot_mapping = np.array(slot_list).astype(np.int32)
 
    
    key_cache = np.ones((num_blocks, block_size, num_heads, head_size_k)).astype(dtype)
    value_cache = np.ones((num_blocks, block_size, num_heads, head_size_v)).astype(dtype)
    key_expect = np.ones((num_blocks, block_size, num_heads, head_size_k)).astype(dtype)
    value_expect = np.ones((num_blocks, block_size, num_heads, head_size_v)).astype(dtype)
    
    
    for i, slot in enumerate(slot_mapping):
        if slot < 0:
            continue 
        block_index = slot // block_size
        block_offset = slot % block_size
 
        token_key = key[i]
        token_v = value[i]
        key_expect[block_index][block_offset] = token_key
        value_expect[block_index][block_offset] = token_v

 
 
    
    ret_data = key, value, key_cache, value_cache, slot_mapping, key_expect, value_expect
    return ret_data
 
 
OP_NAME = "ReshapeAndCacheOperation"
PARAM = json.dumps({})
 
 
data = generate_data()
in_tensors = [torch.from_numpy(tensor) for tensor in data]
in_tensors = [tensor.npu() for tensor in in_tensors]
a = [print(tensor.dtype, tensor.device) for tensor in in_tensors]
 
data1 = generate_data()
in_tensors1 = [torch.from_numpy(tensor).bfloat16() for tensor in data1]
in_tensors1 = [tensor.npu() for tensor in in_tensors1]
 
class TestReshapeAndCacheOperation(operation_test.OperationTest):
    soc_version = operation_test.get_soc_version()
    def golden_calc(self, input_tensors):
        return [in_tensors[5], in_tensors[6]]
 
    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.equal(out_tensor, golden_out_tensor)
 
    def test(self):
        if not TestReshapeAndCacheOperation.soc_version == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.execute_out(OP_NAME, PARAM, [in_tensors[0], in_tensors[1], in_tensors[2],
                                              in_tensors[3], in_tensors[4]], [in_tensors[2], in_tensors[3]])
 
class TestReshapeAndCacheOperationBF16(operation_test.OperationTest):
    def golden_calc(self, input_tensors):
        return [in_tensors1[5], in_tensors1[6]]
 
    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.equal(out_tensor, golden_out_tensor)
 
    def test(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.execute_out(OP_NAME, PARAM, [in_tensors1[0], in_tensors1[1], in_tensors1[2],
                                              in_tensors1[3], in_tensors[4]], [in_tensors1[2], in_tensors1[3]])
 
 
if __name__ == '__main__':
    unittest.main()