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

    def genKVData(self, a,b,c):
        tensor_list = [torch.full((1, c), 0) for i in range(b*a)]
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
        self.seqLen = np.random.randint(10, 11, size=(self.batch)).astype(np.int32)
        self.num_tokens = np.sum(self.seqLen)
        self.wins = np.random.randint(5, 6, size=(self.batch * self.num_heads)).astype(np.int32)

        self.idxOffset = np.random.randint(3, 4, size=(self.batch * self.num_heads)).astype(np.int32)
        print(f'self.idxOffset: {self.idxOffset}')

        if self.dtype == "bfloat16":
            self.key_compress = np.zeros((self.num_tokens, self.num_heads, self.head_size)).astype(np.float32)
            self.value_compress = np.zeros((self.num_tokens, self.num_heads, self.head_size)).astype(np.float32)
            self.key_expect_compress = np.zeros((self.num_blocks, self.block_size, 1, self.head_size)).astype(np.float32)
            self.value_expect_compress = np.zeros((self.num_blocks, self.block_size, 1, self.head_size)).astype(np.float32)
            
            self.key_plain = np.zeros((self.num_tokens, self.num_heads, self.head_size)).astype(np.float32)
            self.value_plain = np.zeros((self.num_tokens, self.num_heads, self.head_size)).astype(np.float32)
            self.key_expect_plain = np.zeros((num_blocks, block_size, num_heads, head_size)).astype(np.float32)
            self.value_expect_plain = np.zeros((num_blocks, block_size, num_heads, head_size)).astype(np.float32)
        else:
            self.key_compress = self.genKVData(self.num_tokens, self.num_heads, self.head_size).astype(self.dtype)
            self.value_compress = self.genKVData(self.num_tokens, self.num_heads, self.head_size).astype(self.dtype)
            self.key_expect_compress = np.zeros((self.num_blocks, self.block_size, 1, self.head_size)).astype(self.dtype)
            self.value_expect_compress = np.zeros((self.num_blocks, self.block_size, 1, self.head_size)).astype(self.dtype)

            self.key_plain = np.zeros((self.num_tokens, self.num_heads, self.head_size)).astype(self.dtype)
            self.value_plain = np.zeros((self.num_tokens, self.num_heads, self.head_size)).astype(self.dtype)
            self.key_expect_plain = np.zeros((num_blocks, block_size, num_heads, head_size)).astype(self.dtype)
            self.value_expect_plain = np.zeros((num_blocks, block_size, num_heads, head_size)).astype(self.dtype)
        value = []
        sumX = 0
        for i in range(self.batch*self.num_heads):
            value.append(sumX * self.block_size)
            sumX += 1
        if self.dtype == "bfloat16":
            self.key_cache_compress = np.zeros((self.num_blocks, self.block_size, 1, self.head_size)).astype(np.float32)
            self.value_cache_compress = np.zeros((self.num_blocks, self.block_size, 1, self.head_size)).astype(np.float32)

            self.key_cache_plain = np.zeros((num_blocks, block_size, num_heads, head_size)).astype(np.float32)
            self.value_cache_plain = np.zeros((num_blocks, block_size, num_heads, head_size)).astype(np.float32)
        else:
            self.key_cache_compress = np.zeros((self.num_blocks, self.block_size, 1, self.head_size)).astype(self.dtype)
            self.value_cache_compress = np.zeros((self.num_blocks, self.block_size, 1, self.head_size)).astype(self.dtype)

            self.key_cache_plain = np.zeros((num_blocks, block_size, num_heads, head_size)).astype(self.dtype)
            self.value_cache_plain = np.zeros((num_blocks, block_size, num_heads, head_size)).astype(self.dtype)
        self.slot_mapping_compress = np.array(value).astype(np.int32)
        num_slots = block_size * num_blocks
        slot_list = random.sample(range(num_slots), self.num_tokens)
        self.slot_mapping_plain = np.array(slot_list).astype(np.int32)

    def generate_test_data(self, isRopeCompress): # 单Batch 单Head idx>0
        if isRopeCompress:
            key_expect_compress = self.key_expect_compress
            value_expect_compress = self.key_expect_compress

            self.new_seq = self.seqLen.copy()
            for n in range(1, len(self.seqLen)):
                self.new_seq[n] = self.seqLen[n] + self.seqLen[n-1]
            self.new_seq = np.hstack((np.array([0], dtype=np.int32), self.new_seq))

            for i, slot in enumerate(self.slot_mapping_compress):
                if slot < 0:
                    continue
                win = self.wins[i]
                idxOffset = self.idxOffset[i]
                
                bsID = i // self.num_heads
                headID = i % self.num_heads
                headStartIdx = self.new_seq[bsID]
                bs = self.new_seq[bsID]

                sum_key, sum_value = 0, 0
                for j in range(self.seqLen[bsID]):
                    block_index = slot // self.block_size
                    block_offset = slot % self.block_size

                    token_key = self.key_compress[bs+j][headID]
                    token_v = self.value_compress[bs+j][headID]
                    if idxOffset == -1 or (j < idxOffset and idxOffset != -1):
                        key_expect_compress[block_index][block_offset] = token_key
                        value_expect_compress[block_index][block_offset] = token_v
                        slot+=1

                    if j >= idxOffset and idxOffset != -1 and win > 0:
                        k = bs+j
                        while (win > 0):
                            token_key = self.key_compress[k][headID]
                            token_v = self.value_compress[k][headID]
                            sum_key+=token_key
                            sum_value+=token_v
                            win-=1
                            k+=1
                        key_expect_compress[block_index][block_offset] = sum_key/(self.wins[i])
                        value_expect_compress[block_index][block_offset] = sum_value/(self.wins[i])
                        slot+=1
                    headEndIdx = idxOffset + self.wins[i]
                    if j >= headEndIdx and idxOffset != -1:
                        key_expect_compress[block_index][block_offset] = token_key
                        value_expect_compress[block_index][block_offset] = token_v
                        slot+=1
            return self.key_compress, self.value_compress, self.key_cache_compress, self.value_cache_compress,\
                self.slot_mapping_compress, self.wins, self.seqLen, self.idxOffset, key_expect_compress, value_expect_compress
        else:
            for i, slot in enumerate(self.slot_mapping_plain):
                if slot < 0:
                    continue 
                block_index = slot // block_size
                block_offset = slot % block_size
 
                token_key = self.key_plain[i]
                token_v = self.value_plain[i]
                self.key_expect_plain[block_index][block_offset] = token_key
                self.value_expect_plain[block_index][block_offset] = token_v
            return self.key_plain, self.value_plain, self.key_cache_plain, self.value_cache_plain,\
                self.slot_mapping_plain, self.key_expect_plain, self.value_expect_plain

    def golden_compare(self, out_tensor, golden_out_tensor):
        expect = golden_out_tensor
        actual = out_tensor
        abs_diff = torch.abs(expect-actual)
        print((abs_diff > 0).nonzero().size(0))
        return (not (abs_diff > 0).nonzero().size(0) > 0)

data_generator = ReshapeAndCacheDataGeneratorRopeCompress()

batch = 1
num_heads = 1
head_size = 128
block_size = 128
num_blocks = 512
dtype = "bfloat16"
data_generator.set_reshape_and_cache_param(num_heads, head_size, block_size, num_blocks, dtype, batch)
data_compress = data_generator.generate_test_data(isRopeCompress=True)
data_plain = data_generator.generate_test_data(isRopeCompress=False)

in_tensors_compress = [torch.from_numpy(tensor) for tensor in data_compress]
in_tensors_compress = [tensor.npu() for tensor in in_tensors_compress]

in_tensors_plain = [torch.from_numpy(tensor) for tensor in data_plain]
in_tensors_plain = [tensor.npu() for tensor in in_tensors_plain]
if dtype == "bfloat16":
    in_tensors_compress[0] = in_tensors_compress[0].to(torch.bfloat16)
    in_tensors_compress[1] = in_tensors_compress[1].to(torch.bfloat16)
    in_tensors_compress[2] = in_tensors_compress[2].to(torch.bfloat16)
    in_tensors_compress[3] = in_tensors_compress[3].to(torch.bfloat16)

    in_tensors_plain[0] = in_tensors_plain[0].to(torch.bfloat16)
    in_tensors_plain[1] = in_tensors_plain[1].to(torch.bfloat16)
    in_tensors_plain[2] = in_tensors_plain[2].to(torch.bfloat16)
    in_tensors_plain[3] = in_tensors_plain[3].to(torch.bfloat16)



OP_NAME = "ReshapeAndCacheOperation"
PARAM_ROPE = json.dumps({"compressType": 2})
PARAM_PLAIN = json.dumps({})

class TestReshapeAndCacheOperationCompressRope(operation_test.OperationTest):
    soc_version = operation_test.get_soc_version()
    def golden_calc(self, input_tensors):
        return [in_tensors_compress[8], in_tensors_compress[9]]
 
    def golden_compare(self, out_tensor, golden_out_tensor):
        return data_generator.golden_compare(out_tensor, golden_out_tensor)
 
    def test(self):
        if not TestReshapeAndCacheOperationCompressRope.soc_version == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.execute_out(OP_NAME, PARAM_ROPE, [in_tensors_compress[0], in_tensors_compress[1], in_tensors_compress[2],\
                in_tensors_compress[3], in_tensors_compress[4], in_tensors_compress[5], in_tensors_compress[6], in_tensors_compress[7]],
                [in_tensors_compress[2], in_tensors_compress[3]])

class TestReshapeAndCacheOperationPlain(operation_test.OperationTest):
    soc_version = operation_test.get_soc_version()
    def golden_calc(self, input_tensors):
        return [in_tensors_plain[5], in_tensors_plain[6]]
 
    def golden_compare(self, out_tensor, golden_out_tensor):
        return data_generator.golden_compare(out_tensor, golden_out_tensor)
 
    def test(self):
        if not TestReshapeAndCacheOperationPlain.soc_version == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.execute_out(OP_NAME, PARAM_PLAIN, [in_tensors_plain[0], in_tensors_plain[1], in_tensors_plain[2],\
                in_tensors_plain[3], in_tensors_plain[4]],
                [in_tensors_plain[2], in_tensors_plain[3]])
 

if __name__ == '__main__':
    unittest.main()