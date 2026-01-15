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

class ReshapeAndCacheDataGeneratorRopeCompress():

    def genKVData(self, a, b, c): # a: BS, b: N, c: D
        tensor_list = [torch.full((1, c), i) for i in range(b*a)]
        conbined_list = []
        for i in range (a):
            left = b*i
            right = b*(i+1)
            combined_tensor = torch.cat(tensor_list[left:right], dim=0)
            conbined_list.append(combined_tensor)
        combined_tensor = torch.stack(conbined_list, dim=0)
        return combined_tensor.numpy()

    def set_reshape_and_cache_param(self, num_heads, head_size, block_size, num_blocks, dtype, batch):
        self.num_heads = num_heads
        self.head_size = head_size
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.dtype = dtype
        self.batch = batch

        len = random.randint(1000, 4096)
        win = random.randint(0, len)
        sub = len - win
        offset = random.randint(-1, sub)
        self.seqLen = torch.randint(low=len,high=len+1,size =[self.batch]).to(torch.int32)
        self.num_tokens = torch.sum(self.seqLen)
        self.wins = torch.randint(low=win, high=win+1, size=[self.batch * self.num_heads]).to(torch.int32)
        self.idxOffset = torch.randint(low = offset, high = offset+1, size=[self.batch * self.num_heads]).to(torch.int32)

        k = np.random.uniform(low=-5, high=5, size=(self.num_tokens, self.num_heads, self.head_size)).astype(np.float32)
        v = np.random.uniform(low=-5, high=5, size=(self.num_tokens, self.num_heads, self.head_size)).astype(np.float32)
        if dtype == "bfloat16":
            self.key = torch.from_numpy(k).to(torch.bfloat16)
            self.value = torch.from_numpy(v).to(torch.bfloat16)
            self.key_expect = torch.zeros(size = (self.num_blocks, self.block_size, 1, self.head_size), dtype=torch.bfloat16)
            self.value_expect = torch.zeros(size = (self.num_blocks, self.block_size, 1, self.head_size), dtype=torch.bfloat16)
        if dtype == "float16":
            self.key = torch.from_numpy(k).to(torch.half)
            self.value = torch.from_numpy(v).to(torch.half)
            self.key_expect = torch.zeros(size = (self.num_blocks, self.block_size, 1, self.head_size), dtype=torch.half)
            self.value_expect = torch.zeros(size = (self.num_blocks, self.block_size, 1, self.head_size), dtype=torch.half)

        cacheStart = []
        cacheOffset = 0
        for i in range(self.batch * self.num_heads):
            cacheStart.append(cacheOffset)
            if (self.idxOffset[i] != -1) :
                cacheOffset += self.seqLen[int(i/self.num_heads)].item() - self.wins[i].item() + 1
            else :
                cacheOffset += self.seqLen[int(i/self.num_heads)].item()
        self.slot_mapping = torch.Tensor(cacheStart).to(torch.int32)

    def generate_test_data(self):
        key_expect_fp32 = self.key_expect.clone().to(torch.float32)
        value_expect_fp32 = self.value_expect.clone().to(torch.float32)

        self.new_seq = self.seqLen.clone()
        self.new_seq[0] = self.seqLen[0]
        for n in range(1, len(self.seqLen)):
            self.new_seq[n] = self.seqLen[n] + self.new_seq[n-1]
        self.new_seq = torch.cat((torch.zeros(1, dtype=torch.int32), self.new_seq), dim=0)

        for i, slotConst in enumerate(self.slot_mapping):
            slot = slotConst.clone().detach()
            if slot < 0:
                continue
            win = self.wins[i].clone()
            idxOffset = self.idxOffset[i]

            bsID = i // self.num_heads
            headID = i % self.num_heads
            headStartIdx = self.new_seq[bsID]
            headEndIdx = idxOffset + self.wins[i]
            bs = self.new_seq[bsID]

            sum_key = torch.zeros(self.head_size, dtype = torch.float32)
            sum_value = torch.zeros(self.head_size, dtype = torch.float32)
            for j in range(self.seqLen[bsID]):
                block_index = slot // self.block_size
                block_offset = slot % self.block_size

                token_key = self.key[bs+j][headID]
                token_v = self.value[bs+j][headID]
                if idxOffset == -1 or (j < idxOffset and idxOffset != -1):
                    key_expect_fp32[block_index][block_offset] = token_key
                    value_expect_fp32[block_index][block_offset] = token_v
                    slot+=1

                if j >= idxOffset and idxOffset != -1 and win > 0:
                    k = bs+j
                    while (win > 0):
                        rope_key = torch.clone(self.key[k][headID]).to(torch.float32)
                        rope_v = torch.clone(self.value[k][headID]).to(torch.float32)
                        sum_key += rope_key
                        sum_value += rope_v
                        win -= 1
                        k += 1

                    key_expect_fp32[block_index][block_offset] = (sum_key / (self.wins[i]))
                    value_expect_fp32[block_index][block_offset] = (sum_value / (self.wins[i]))
                    slot+=1

                if j >= headEndIdx and idxOffset != -1:
                    key_expect_fp32[block_index][block_offset] = token_key.to(torch.float32)
                    value_expect_fp32[block_index][block_offset] = token_v.to(torch.float32)
                    slot+=1
        return self.key, self.value, self.key_expect, self.value_expect, self.slot_mapping, self.wins, self.seqLen, self.idxOffset, key_expect_fp32, value_expect_fp32

    def golden_compare(self, out_tensor, golden_out_tensor):
        err = 2**(-8)
        if out_tensor[0].dtype == torch.bfloat16:
            err = 2**(-7)
        elif out_tensor[0].dtype == torch.float32:
            err = 2**(-14)     
        out_tensor[0] = out_tensor[0].to(torch.float32)
        out_tensor[1] = out_tensor[1].to(torch.float32)
        diff = torch.abs(torch.subtract(out_tensor[0].npu(), golden_out_tensor[0].npu()))
        tensor_max = torch.maximum(torch.ones(golden_out_tensor[0].shape, dtype=golden_out_tensor[0].dtype).npu(), torch.abs(golden_out_tensor[0]).npu())
        flag = True
        if torch.any(torch.greater(diff, err * tensor_max)):
            return False
        diff = torch.abs(torch.subtract(out_tensor[1], golden_out_tensor[1]))
        tensor_max = torch.maximum(torch.ones(golden_out_tensor[1].shape, dtype=golden_out_tensor[1].dtype).npu(), torch.abs(golden_out_tensor[1]).npu())
        if torch.any(torch.greater(diff, err * tensor_max)):
            return False
        return True

data_generator = ReshapeAndCacheDataGeneratorRopeCompress()
batch = 1
num_heads = 1
head_size = 128
block_size = 128
num_blocks = 512

dtype = "float16"
data_generator.set_reshape_and_cache_param(num_heads, head_size, block_size, num_blocks, dtype, batch)
in_tensors_fp16 = data_generator.generate_test_data()
in_tensors_fp16 = [tensor.npu() for tensor in in_tensors_fp16]

dtype = "bfloat16"
data_generator.set_reshape_and_cache_param(num_heads, head_size, block_size, num_blocks, dtype, batch)
in_tensors_bf16 = data_generator.generate_test_data()
in_tensors_bf16 = [tensor.npu() for tensor in in_tensors_bf16]


OP_NAME = "ReshapeAndCacheOperation"
PARAM = json.dumps({"compressType": 2})

class TestReshapeAndCacheOperationCompressRopefp16(operation_test.OperationTest):
    soc_version = operation_test.get_soc_version()
    def golden_calc(self, input_tensors):
        return [in_tensors_fp16[8], in_tensors_fp16[9]]
 
    def golden_compare(self, out_tensor, golden_out_tensor):
        return data_generator.golden_compare(out_tensor, golden_out_tensor)
 
    def test(self):
        if not TestReshapeAndCacheOperationCompressRopefp16.soc_version == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.execute_out(OP_NAME, PARAM, [in_tensors_fp16[0], in_tensors_fp16[1], in_tensors_fp16[2],\
                in_tensors_fp16[3], in_tensors_fp16[4], in_tensors_fp16[5], in_tensors_fp16[6], in_tensors_fp16[7]],
                [in_tensors_fp16[2], in_tensors_fp16[3]])
 
class TestReshapeAndCacheOperationCompressRopebf16(operation_test.OperationTest):
    soc_version = operation_test.get_soc_version()
    def golden_calc(self, input_tensors):
        return [in_tensors_bf16[8], in_tensors_bf16[9]]
 
    def golden_compare(self, out_tensor, golden_out_tensor):
        return data_generator.golden_compare(out_tensor, golden_out_tensor)
 
    def test(self):
        if not TestReshapeAndCacheOperationCompressRopebf16.soc_version == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.execute_out(OP_NAME, PARAM, [in_tensors_bf16[0].to(torch.bfloat16), in_tensors_bf16[1].to(torch.bfloat16), in_tensors_bf16[2].to(torch.bfloat16),\
                in_tensors_bf16[3].to(torch.bfloat16), in_tensors_bf16[4], in_tensors_bf16[5], in_tensors_bf16[6], in_tensors_bf16[7]],
                [in_tensors_bf16[2].to(torch.bfloat16), in_tensors_bf16[3].to(torch.bfloat16)])

if __name__ == '__main__':
    unittest.main()