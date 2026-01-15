#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

def genKVData(a, b, c): # a: BS, b: N, c: D
    tensor_list = [torch.full((1, c), i) for i in range(b*a)]
    conbined_list = []
    for i in range (a):
        left = b*i
        right = b*(i+1)
        combined_tensor = torch.cat(tensor_list[left:right], dim=0)
        conbined_list.append(combined_tensor)
    combined_tensor = torch.stack(conbined_list, dim=0)
    return combined_tensor.numpy()

def generate_data(
        num_heads=1,
        head_size=256,
        block_size=256,
        num_blocks=2048,
        dtype="half",
        batch=1,
):

    len = random.randint(1024,4096)
    win = random.randint(0, len)
    sub = len - win
    offset = random.randint(-1, sub)
    seqLen = torch.randint(low=len,high=len+1,size =[batch]).to(torch.int32)
    num_tokens = torch.sum(seqLen)
    wins = torch.randint(low=win, high=win+1, size=[batch*(num_heads)]).to(torch.int32)
    idxOffset = torch.randint(low=offset, high=offset+1, size=[batch*num_heads]).to(torch.int32)
 
    k = np.zeros((num_tokens, num_heads, head_size)).astype(np.float32)
    v = np.zeros((num_tokens, num_heads, head_size)).astype(np.float32)
    if dtype == "bfloat16":
        key = torch.from_numpy(k).to(torch.bfloat16)
        value = torch.from_numpy(v).to(torch.bfloat16)
        key_expect = torch.zeros(size = (num_blocks, block_size, 1, head_size), dtype=torch.bfloat16)
        value_expect = torch.zeros(size = (num_blocks, block_size, 1, head_size), dtype=torch.bfloat16)
    if dtype == "half":
        key = torch.from_numpy(k).to(torch.half)
        value = torch.from_numpy(v).to(torch.half)
        key_expect = torch.zeros(size = (num_blocks, block_size, 1, head_size), dtype=torch.half)
        value_expect = torch.zeros(size = (num_blocks, block_size, 1, head_size), dtype=torch.half)

    cacheStart = []
    cacheOffset = 0
    for i in range(batch * num_heads):
        cacheStart.append(cacheOffset)
        if (idxOffset[i] != -1) :
            cacheOffset += seqLen[int(i/num_heads)].item() - wins[i].item()
        else :
            cacheOffset += seqLen[int(i/num_heads)].item()
    slot_mapping = torch.Tensor(cacheStart).to(torch.int32)

    key_expect_fp32 = key_expect.clone().to(torch.float32)
    value_expect_fp32 = value_expect.clone().to(torch.float32)
 
    new_seq = seqLen.clone()
    new_seq[0] = seqLen[0]
    for n in range(1, batch):
        new_seq[n] = seqLen[n] + new_seq[n-1]
    new_seq = torch.cat((torch.zeros(1, dtype=torch.int32), new_seq), dim=0)
 
    for i, slot in enumerate(slot_mapping):
        if slot < 0:
            continue
        win = wins[i].clone()
        curIdxOffset = idxOffset[i]
 
        bsID = i // num_heads
        headID = i % num_heads
        headStartIdx = new_seq[bsID]
        headEndIdx = curIdxOffset + wins[i]
        bs = new_seq[bsID]
 
        sum_key = torch.zeros(head_size, dtype = torch.float32)
        sum_value = torch.zeros(head_size, dtype = torch.float32)
        for j in range(seqLen[bsID]):
            block_index = torch.div(slot, block_size, rounding_mode='trunc')
            block_offset = slot % block_size

            token_key = key[bs+j][headID]
            token_v = value[bs+j][headID]
            if curIdxOffset == -1 or (j < curIdxOffset and curIdxOffset != -1):
                key_expect_fp32[block_index][block_offset] = token_key
                value_expect_fp32[block_index][block_offset] = token_v
                slot+=1

            if j >= headEndIdx and curIdxOffset != -1:
                key_expect_fp32[block_index][block_offset] = token_key.to(torch.float32)
                value_expect_fp32[block_index][block_offset] = token_v.to(torch.float32)
                slot+=1


    ret_data = key, value, key_expect, value_expect, slot_mapping, wins, seqLen, idxOffset, key_expect_fp32, value_expect_fp32
    return ret_data
 
 
OP_NAME = "ReshapeAndCacheOmniOperation"
PARAM = json.dumps({})
 
 
in_tensors = generate_data()
in_tensors = [tensor.npu() for tensor in in_tensors]
a = [print(tensor.dtype, tensor.device) for tensor in in_tensors]
 

 
class TestReshapeAndCacheOmniOperation(operation_test.OperationTest):
    soc_version = operation_test.get_soc_version()
    def golden_calc(self, input_tensors):
        return [in_tensors[8], in_tensors[9]]
 
    def golden_compare(self, out_tensor, golden_out_tensor):
        err = 2**(-8)
        if out_tensor.dtype == torch.bfloat16:
            err = 2**(-7)
        elif out_tensor.dtype == torch.float32:
            err = 2**(-14)
        
        out_tensor = out_tensor.to(torch.float32)
        kdiff = torch.abs(torch.subtract(out_tensor, golden_out_tensor))
        tensor_max = torch.maximum(torch.ones(golden_out_tensor.shape, dtype=golden_out_tensor.dtype).npu(), torch.abs(golden_out_tensor))
        if torch.any(torch.greater(kdiff, err * tensor_max)):
            return False
        return True
 
    def test(self):
        if not TestReshapeAndCacheOmniOperation.soc_version == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        print(torch.equal(in_tensors[2], torch.zeros(in_tensors[2].shape).npu()))
        self.execute_out(OP_NAME, PARAM, [in_tensors[0], in_tensors[1], in_tensors[2],
            in_tensors[3], in_tensors[4], in_tensors[5], in_tensors[6], in_tensors[7]], [in_tensors[2], in_tensors[3]])
 
 
 
if __name__ == '__main__':
    unittest.main()