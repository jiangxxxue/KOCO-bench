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
import subprocess

OP_NAME = "RopeQConcatOperation"
def padHeadNum(headdim_sin_cos):
    head_num = 32
    return np.tile(headdim_sin_cos, (1, head_num))

def rotateHalf(q_temp):
    head_num = 32
    # 拆分成 head_num 个 [n,head_dim] 的二维向量
    q_splits = np.split(q_temp, head_num, axis=1)
    # 对每个 [n,head_dim] 向量的第二维进行分割，并对第二块乘以 -1再拼回到第一块前面
    processed_q_splits = []
    for q_split in q_splits:
        # 分割第二维
        first_half, second_half = np.split(q_split, 2, axis=1)
        # 拼接回 [n,head_dim] 的二维向量
        processed_q_split = np.concatenate((-second_half, first_half), axis=1)
        processed_q_splits.append(processed_q_split)

    # 将所有处理后的 [n,head_dim] 向量拼回 [n,head_num*head_dim] 的二维向量
    return np.concatenate(processed_q_splits, axis=1)

def RopeConcatGolden(q, cos,sin, concatInput):
    input_token_num = 64
    head_dim = 64
    head_num = 32
    
    pad_sin = padHeadNum(sin)
    pad_cos = padHeadNum(cos)

    rope_res = q*pad_cos + rotateHalf(q)*pad_sin
    rope_res = rope_res.reshape(input_token_num, head_num, head_dim)
    return np.concatenate((rope_res, concatInput), axis=2)


class TestRopeQConcatOperation(op_test.OpTest):
    def golden_calc(self, in_tensors):
        q = np.array(in_tensors[0].cpu().float()).astype(np.float32)
        cos = np.array(in_tensors[1].cpu().float()).astype(np.float32)
        sin = np.array(in_tensors[2].cpu().float()).astype(np.float32)
        concatInput = np.array(in_tensors[3].cpu().float()).astype(np.float32)
        res = RopeConcatGolden(q,cos,sin,concatInput)
        res = res.astype(np.float32)
        res = torch.tensor(res, dtype=torch.bfloat16)
        return [res]

    def golden_compare(self, out_tensors, golden_out_tensors):
        return torch.allclose(out_tensors[0], golden_out_tensors[0], rtol=0.001, atol=0.001)
    
    @op_test.only_910b
    def test_rope_larger_qk(self):
        OP_PARAM0 = {}
        input_token_num = 64
        head_dim = 64
        head_num = 32
        q_hidden_size = head_dim * head_num
        concat_hidden_size = 512

        q_golden = torch.rand(input_token_num, q_hidden_size).bfloat16()
        cos_golden = torch.rand(input_token_num, head_dim).bfloat16()
        sin_golden = torch.rand(input_token_num, head_dim).bfloat16()
        concatInput_golden = torch.rand(input_token_num, head_num , concat_hidden_size ).bfloat16()
        
        outputshape = torch.rand(input_token_num, head_num , head_dim + concat_hidden_size ).bfloat16()
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([q_golden, cos_golden, sin_golden, concatInput_golden],
                     [torch.zeros(outputshape.shape).bfloat16()])

if __name__ == '__main__':
    unittest.main()
