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
import random
import logging
import numpy as np
import torch
import sys
import op_test
from tensor_file import read_tensor
import subprocess
# torch.set_printoptions(threshold=torch.inf)
OP_NAME = "RopeOperation"

class TestRopeOperation(op_test.OpTest):
    def golden_calc(self, in_tensors):
        q = np.array(in_tensors[0].cpu().float()).astype(np.float32)
        kk = np.array(in_tensors[1].cpu().float()).astype(np.float32)
        cos = np.array(in_tensors[2].cpu().float()).astype(np.float32)
        sin = np.array(in_tensors[3].cpu().float()).astype(np.float32)
        seqlen = np.array(in_tensors[4].cpu()).astype(np.int32)
        batch = q.shape[0]
        rotaryCoeff = self.op_desc["specificParam"]["rotaryCoeff"]
        headDim = cos.shape[-1]
        hiddensizeQ = q.shape[-1]
        hiddensizeK = kk.shape[-1]
        hiddensize = max(hiddensizeQ, hiddensizeK)
        headNumQ = hiddensizeQ // headDim
        headNumK = hiddensizeK // headDim
        headNum = max(headNumQ, headNumK)
        ntokens = np.sum(seqlen)

        q = q.astype(np.float32)
        kk = kk.astype(np.float32)

        rope_q = np.zeros(shape=(ntokens, hiddensizeQ)).astype(np.float32)
        rope_k = np.zeros(shape=(ntokens, hiddensizeK)).astype(np.float32)
        prefix_Ntokens = 0
        cosTable = np.zeros(shape=(ntokens, hiddensize)).astype(np.float32)
        for i in range(ntokens):
            for j in range(headNum):
                cosTable[i][j*headDim:(j+1)*headDim] = cos[i][:]
        for i in range(batch):
            curr_seqLen = seqlen[i][0]
            q1 = np.zeros(shape=(curr_seqLen, hiddensizeQ)).astype(np.float32)
            k1 = np.zeros(shape=(curr_seqLen, hiddensizeK)).astype(np.float32)

            for z in range(prefix_Ntokens, prefix_Ntokens + curr_seqLen):
                q1[z-prefix_Ntokens] = q[z] * cosTable[z][:hiddensizeQ]
                k1[z-prefix_Ntokens] = kk[z] * cosTable[z][:hiddensizeK] 
            q2 = np.zeros(shape=(curr_seqLen, hiddensizeQ)).astype(np.float32)
            k2 = np.zeros(shape=(curr_seqLen, hiddensizeK)).astype(np.float32)        
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
        return [torch.tensor(rope_q).bfloat16(), torch.tensor(rope_k).bfloat16()]

    def golden_compare(self, out_tensors, golden_out_tensors):
        return torch.allclose(out_tensors[0], golden_out_tensors[0], rtol=0.0001, atol=0.0001) and torch.allclose(out_tensors[1], golden_out_tensors[1], rtol=0.0001, atol=0.0001)
        
    @op_test.only_910b
    def test_2d_half(self):
        '''
            基础场景
            shape: (3, 16640) / (3, 9216) / (3, 256) / (3, 256)
        '''
        batch = 3
        rotaryCoeff = 2
        headDim = 256
        OP_PARAM0 = {"rotaryCoeff": rotaryCoeff}

        hiddensizeQ = 16640
        hiddensizeK = 9216
        hiddensize = max(hiddensizeQ, hiddensizeK)

        headNumQ = hiddensizeQ // headDim
        headNumK = hiddensizeK // headDim
        headNum = max(headNumQ, headNumK)
        q = torch.rand(batch, hiddensizeQ).bfloat16()
        kk = torch.rand(batch, hiddensizeK).bfloat16()
        cos = torch.rand(batch, headDim).bfloat16()
        sin = torch.rand(batch, headDim).bfloat16()
        seqlen =torch.randint(1, 2, size=(batch,1), dtype=torch.int32)
        ntokens = torch.sum(seqlen)
        for i in range(ntokens):
            for j in range(headDim//2):
                cos[i][(2*j+1):(2*j+2)] =  cos[i][(2*j):(2*j+1)]
                sin[i][(2*j+1):(2*j+2)] =  sin[i][(2*j):(2*j+1)]
        
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([q, kk,cos, sin, seqlen],
                     [torch.zeros(q.shape).bfloat16(), torch.zeros(kk.shape).bfloat16()])
    
    @op_test.only_910b
    def test_2d_half2(self):
        for _ in range(5):
            batch = np.random.randint(1, 200)
            headDim = random.randint(1, 5) * 16
            rotaryCoeff = random.choice([2, 4, headDim])
            highPrecision = 1
            OP_PARAM0 = {"rotaryCoeff": rotaryCoeff}
            seqlen =np.random.randint(1, 2, size=batch, dtype=np.int32)

            hiddensizeQ = random.randint(51, 100) * headDim
            hiddensizeK = random.randint(1, 50) * headDim

            hiddensize = max(hiddensizeQ, hiddensizeK)

            headNumQ = hiddensizeQ // headDim
            headNumK = hiddensizeK // headDim
            headNum = max(headNumQ, headNumK)

            q = torch.rand(batch, hiddensizeQ).bfloat16()
            kk = torch.rand(batch, hiddensizeK).bfloat16()
            cos = torch.rand(batch, headDim).bfloat16()
            sin = torch.rand(batch, headDim).bfloat16()
            seqlen =torch.randint(1, 2, size=(batch,1), dtype=torch.int32)

            logging.debug("q shape is %s", q.shape)
            logging.debug("kk shape is %s", kk.shape)
            logging.debug("cos shape is %s", cos.shape)
            logging.debug("sin shape is %s", sin.shape)

            ntokens = torch.sum(seqlen)
            for i in range(ntokens):
                for j in range(headDim//2):
                    cos[i][(2*j+1):(2*j+2)] =  cos[i][(2*j):(2*j+1)]
                    sin[i][(2*j+1):(2*j+2)] =  sin[i][(2*j):(2*j+1)]
            
            self.set_param(OP_NAME, OP_PARAM0)
            self.execute([q, kk,cos, sin, seqlen],
                        [torch.zeros(q.shape).bfloat16(), torch.zeros(kk.shape).bfloat16()])

if __name__ == '__main__':
    unittest.main()
