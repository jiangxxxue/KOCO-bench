# 
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# 
import unittest
import numpy as np
import torch
import torch.nn.functional as F
import sys, os
import op_test
import random


OP_NAME = "RopeGradOperation"
OP_PARAM = {"batch":0,"qSeqLen":None}


class TestRope(op_test.OpTest):
    def golden_calc(self, in_tensors):
        # x,128*32-->reshape x,32,128
        cos_list=[in_tensors[2][:x, :] for x in OP_PARAM['qSeqLen']]
        sin_list=[in_tensors[3][:x, :] for x in OP_PARAM['qSeqLen']]
        cos = torch.cat(cos_list, dim=0)
        sin = torch.cat(sin_list, dim=0)
        sin1=sin[:,:64]
        sin2=sin[:,64:]
        rohqgsin=torch.cat((sin2,-sin1),dim=-1)
        
        q_grad=torch.zeros_like(in_tensors[0])
        bs=int(in_tensors[0].shape[1]/128)
        for i in range(bs):
            q_grad[:,i*128:(i+1)*128] = in_tensors[0][:,i*128:(i+1)*128] *(cos+rohqgsin)
    
        k_grad=torch.zeros_like(in_tensors[1])
        for i in range(bs):
            k_grad[:,i*128:(i+1)*128] = in_tensors[1][:,i*128:(i+1)*128] *(cos+rohqgsin)
        return [q_grad,k_grad]

    def golden_compare(self, out_tensors, golden_out_tensors):
        return torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001) and torch.allclose(out_tensors[1].float(), golden_out_tensors[1].float(), rtol=0.001, atol=0.001)

    @op_test.only_910b
    def test_add_float(self):
        maxSeqLen=2048
        OP_PARAM['batch']=16
        OP_PARAM['qSeqLen']=torch.randint(1,maxSeqLen,(OP_PARAM['batch'],),dtype=torch.int32).cpu().numpy().tolist()
        self.set_param(OP_NAME, OP_PARAM)
        hiddenSize=1280
        headSize= 128
        sumSeqLen=sum(OP_PARAM['qSeqLen'])
        q=torch.rand((sumSeqLen, hiddenSize)).half()
        k=torch.rand((sumSeqLen, hiddenSize)).half()
        cos=torch.rand((maxSeqLen, headSize)).half()
        sin=torch.rand((maxSeqLen, headSize)).half()

        input_tensor = [q,k,cos,sin]
        output_tensor = [torch.zeros_like(q,dtype=torch.float16),torch.zeros_like(k,dtype=torch.float16)]
        self.execute(input_tensor, output_tensor)


if __name__ == '__main__':
    unittest.main()
