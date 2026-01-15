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
        dtype = "float16"
):
    soc_version = operation_test.get_soc_version()
    if dtype == "bfloat16":
        head_size_k = np.random.randint(1, 256)
        key = np.random.uniform(-1.0, 1.0, size=(num_tokens, num_heads, head_size_k)).astype(np.float32)
    
        num_slots = block_size * num_blocks
        slot_list = random.sample(range(num_slots), num_tokens)
        slot_mapping = np.array(slot_list).astype(np.int32)
    
        
        key_cache = np.ones((num_blocks, block_size, num_heads, head_size_k)).astype(np.float32)
        key_expect = np.ones((num_blocks, block_size, num_heads, head_size_k)).astype(np.float32)
    else:
        head_size_k = np.random.randint(1, 256)
        key = np.random.uniform(-1.0, 1.0, size=(num_tokens, num_heads, head_size_k)).astype(dtype)
    
        num_slots = block_size * num_blocks
        slot_list = random.sample(range(num_slots), num_tokens)
        slot_mapping = np.array(slot_list).astype(np.int32)
    
        
        key_cache = np.ones((num_blocks, block_size, num_heads, head_size_k)).astype(dtype)
        key_expect = np.ones((num_blocks, block_size, num_heads, head_size_k)).astype(dtype)
    
    
    for i, slot in enumerate(slot_mapping):
        if slot < 0:
            continue 
        block_index = slot // block_size
        block_offset = slot % block_size
 
        token_key = key[i]
        key_expect[block_index][block_offset] = token_key 
    
    ret_data = key, key_cache, slot_mapping, key_expect
    return ret_data
 
class TestReshapeAndCacheOperation(operation_test.OperationTest):
    soc_version = operation_test.get_soc_version()
    def golden_calc(self, input_tensors):
        return [self.in_tensors[3]]
 
    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.equal(out_tensor, golden_out_tensor)
 
    def test(self):
        if not TestReshapeAndCacheOperation.soc_version == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        
        batch = 2
        seq_len = 1
        num_tokens = batch * seq_len
        num_heads = 32
        head_size_k = 128
        block_size = 128
        num_blocks = 512
        dtype = "float16"
        data = generate_data(num_tokens,num_heads,head_size_k,block_size,num_blocks,dtype)

        self.in_tensors = [torch.from_numpy(tensor) for tensor in data]
        self.in_tensors = [tensor.npu() for tensor in self.in_tensors]
        a = [print(tensor.dtype, tensor.device) for tensor in self.in_tensors]

        OP_NAME = "ReshapeAndCacheOperation"
        PARAM = json.dumps({"kvCacheCfg" : 1})
        self.execute_out(OP_NAME, PARAM, [self.in_tensors[0], self.in_tensors[1], self.in_tensors[2]], [self.in_tensors[1]])
 
class TestReshapeAndCacheOperationBF16(operation_test.OperationTest):
    soc_version = operation_test.get_soc_version()
    def golden_calc(self, input_tensors):
        return [self.in_tensors[3]]
 
    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.equal(out_tensor, golden_out_tensor)
 
    def test(self):
        if not TestReshapeAndCacheOperation.soc_version == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        
        batch = 2
        seq_len = 1
        num_tokens = batch * seq_len
        num_heads = 32
        head_size_k = 128
        block_size = 128
        num_blocks = 512
        dtype = "bfloat16"
        data = generate_data(num_tokens,num_heads,head_size_k,block_size,num_blocks,dtype)

        self.in_tensors = [torch.from_numpy(tensor).bfloat16() for tensor in data]
        self.in_tensors[2] = torch.from_numpy(data[2])
        self.in_tensors = [tensor.npu() for tensor in self.in_tensors]
        a = [print(tensor.dtype, tensor.device) for tensor in self.in_tensors]

        OP_NAME = "ReshapeAndCacheOperation"
        PARAM = json.dumps({"kvCacheCfg" : 1})
        self.execute_out(OP_NAME, PARAM, [self.in_tensors[0], self.in_tensors[1], self.in_tensors[2]], [self.in_tensors[1]])

class TestReshapeAndCacheOperationINT8(operation_test.OperationTest):
    soc_version = operation_test.get_soc_version()
    def golden_calc(self, input_tensors):
        return [self.in_tensors[3]]
 
    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.equal(out_tensor, golden_out_tensor)
 
    def test(self):
        if not TestReshapeAndCacheOperation.soc_version == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        
        batch = 2
        seq_len = 1
        num_tokens = batch * seq_len
        num_heads = 32
        head_size_k = 128
        block_size = 128
        num_blocks = 512
        dtype = "int8"
        data = generate_data(num_tokens,num_heads,head_size_k,block_size,num_blocks,dtype)

        self.in_tensors = [torch.from_numpy(tensor) for tensor in data]
        self.in_tensors = [tensor.npu() for tensor in self.in_tensors]
        a = [print(tensor.dtype, tensor.device) for tensor in self.in_tensors]

        OP_NAME = "ReshapeAndCacheOperation"
        PARAM = json.dumps({"kvCacheCfg" : 1})
        self.execute_out(OP_NAME, PARAM, [self.in_tensors[0], self.in_tensors[1], self.in_tensors[2]], [self.in_tensors[1]])
 
 
if __name__ == '__main__':
    unittest.main()