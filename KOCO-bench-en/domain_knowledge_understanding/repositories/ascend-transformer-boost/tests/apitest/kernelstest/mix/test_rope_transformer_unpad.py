# 
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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
import op_test
from tensor_file import read_tensor
import subprocess

batch = 32
rotaryCoeff = 2
headDim = 128

maxseqlen=2048
seqlen =np.random.randint(1, maxseqlen, size=batch, dtype=np.int32)
ntokens = np.sum(seqlen)

hiddensizeQ = 1024
hiddensizeK = 1024
hiddensize = max(hiddensizeQ, hiddensizeK)

headNumQ = hiddensizeQ // headDim
headNumK = hiddensizeK // headDim
headNum = max(headNumQ, headNumK)
OP_NAME = "RopeOperation"
OP_PARAM0 = {"rotaryCoeff": rotaryCoeff, "cosFormat":1}
q = np.random.uniform(-1, 1, size=(np.sum(seqlen), hiddensizeQ)).astype(np.float16)
kk = np.random.uniform(-1, 1, size=(np.sum(seqlen), hiddensizeK)).astype(np.float16)
cos = np.random.uniform(-1, 1, size=(maxseqlen, headDim)).astype(np.float16)
sin = np.random.uniform(-1, 1, size=(maxseqlen, headDim)).astype(np.float16)


def rope_compute(batch, rotaryCoeff, headDim, hiddensize, 
                hiddensizeQ, hiddensizeK, seqlen, headNum, headNumQ,headNumK,
                q, kk, cos, sin):
    # GT
    rope_q = np.zeros(shape=(ntokens, hiddensizeQ)).astype(np.float16)
    rope_k = np.zeros(shape=(ntokens, hiddensizeK)).astype(np.float16)
    prefix_Ntokens = 0
    cos_list = [cos[:x, :] for x in seqlen]
    sin_list = [sin[:x, :] for x in seqlen]
    cos=np.squeeze(np.concatenate(cos_list,axis=0))
    sin=np.squeeze(np.concatenate(sin_list,axis=0))
    cosTable = np.zeros(shape=(ntokens, hiddensize)).astype(np.float16)
    for i in range(ntokens):
        for j in range(headNum):
            cosTable[i][j*headDim:(j+1)*headDim] = cos[i][:]
    for i in range(batch):
        curr_seqLen = seqlen[i]
        q1 = np.zeros(shape=(curr_seqLen, hiddensizeQ)).astype(np.float16)
        k1 = np.zeros(shape=(curr_seqLen, hiddensizeK)).astype(np.float16)

        for i in range(prefix_Ntokens, prefix_Ntokens + curr_seqLen):
            q1[i-prefix_Ntokens] = q[i] * cosTable[i][:hiddensizeQ]
            k1[i-prefix_Ntokens] = kk[i] * cosTable[i][:hiddensizeK] 
        q2 = np.zeros(shape=(curr_seqLen, hiddensizeQ)).astype(np.float16)
        k2 = np.zeros(shape=(curr_seqLen, hiddensizeK)).astype(np.float16)        
        for k in range(headNum):
            src_ = k * headDim
            dst_ = (k + 1) * headDim
            strdie = headDim // 2
            rotaryStrdie = headDim // rotaryCoeff
            rotaryTimesPerHead = rotaryCoeff / 2
            for cycle in range(int(rotaryTimesPerHead)):
                src =  src_ + cycle * rotaryStrdie * 2
                dst = src + rotaryStrdie * 2
                for curr_seqLeni in range(curr_seqLen):
                    if k < headNumQ:
                        q2[curr_seqLeni][src:src + rotaryStrdie] = q[prefix_Ntokens + curr_seqLeni][src+ rotaryStrdie:dst] * (-1)
                        q2[curr_seqLeni][src + rotaryStrdie:dst] = q[prefix_Ntokens + curr_seqLeni][src:src+rotaryStrdie]
                        q2[curr_seqLeni][src:dst] = q2[curr_seqLeni][src:dst] * sin[prefix_Ntokens + curr_seqLeni][cycle * rotaryStrdie * 2: (cycle +1) * rotaryStrdie * 2]
                    if k < headNumK:
                        k2[curr_seqLeni][src:src + rotaryStrdie] = kk[prefix_Ntokens + curr_seqLeni][src+ rotaryStrdie:dst] * (-1)
                        k2[curr_seqLeni][src + rotaryStrdie:dst] = kk[prefix_Ntokens + curr_seqLeni][src:src+rotaryStrdie]
                        k2[curr_seqLeni][src:dst] = k2[curr_seqLeni][src:dst] * sin[prefix_Ntokens + curr_seqLeni][cycle * rotaryStrdie * 2: (cycle +1) * rotaryStrdie * 2]
        rope_q[prefix_Ntokens:prefix_Ntokens + curr_seqLen] += q1 + q2
        rope_k[prefix_Ntokens:prefix_Ntokens + curr_seqLen] += k1 + k2      
        
        prefix_Ntokens += curr_seqLen
    return rope_q, rope_k

class TestRopeOperation(op_test.OpTest):
    def golden_calc(self, in_tensors):
        rope_q, rope_k = rope_compute(batch, rotaryCoeff, headDim, hiddensize, 
                            hiddensizeQ, hiddensizeK,
                            np.array(in_tensors[4].cpu()).astype(np.int32), 
                            headNum, headNumQ,headNumK,
                            np.array(in_tensors[0].cpu()).astype(np.float16), 
                            np.array(in_tensors[1].cpu()).astype(np.float16), 
                            np.array(in_tensors[2].cpu()).astype(np.float16),
                            np.array(in_tensors[3].cpu()).astype(np.float16))
        return [torch.tensor(rope_q), torch.tensor(rope_k)]

    def golden_compare(self, out_tensors, golden_out_tensors):
        return torch.allclose(out_tensors[0], golden_out_tensors[0], rtol=0.001, atol=0.001) and torch.allclose(out_tensors[1], golden_out_tensors[1], rtol=0.001, atol=0.001)
    
    def test_2d_half(self):
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.tensor(q).half(), torch.tensor(kk).half(), torch.tensor(cos).half(), torch.tensor(sin).half(), torch.tensor(seqlen).int()],
                     [torch.zeros(q.shape).half(), torch.zeros(kk.shape).half()])

if __name__ == '__main__':
    unittest.main()
