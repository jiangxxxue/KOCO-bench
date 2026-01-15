# -*- coding: UTF-8 -*-
#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import sys
import os
import unittest
import math
import json
import torch
import torch_npu
import torch.nn.functional as F
from ctypes import CDLL
import numpy as np
from torch import nn
from torch.nn.parameter import Parameter

sys.path.append(os.path.join(os.path.dirname(__file__), "../../operations"))
import operation_test

LAYER_NAME = "PostProcessOperation"

PARAM = {
    "temperature": 2.0,
    "topK": 4,
    "randSeed": 128,
}
libc = CDLL("libc.so.6")
rand_list = [libc.rand() / 0x7fffffff for i in range(64)]

intensor0 = torch.rand(4, 128, dtype=torch.float16).npu().half() # in_score
intensor1 = torch.rand(4, 128, dtype=torch.float16).npu().half() # in_temperature
intensor2 = torch.HalfTensor([[0.9]]).npu()

outtensor0 = torch.rand(4, 1, dtype=torch.float16).npu().half()
outtensor1 = torch.rand(4, 1, dtype=torch.float16).npu().half()

class TestPostProcessOperation(operation_test.OperationTest):
    def golden_topk_topp(self, in_tensors):
        probs = in_tensors[0].cpu().numpy()
        topp = in_tensors[1].cpu().numpy()
        topk = 4
        if topk > 0 and topk % 256 == 0:
            probs_sorted = np.sort(probs, axis=-1)[..., ::-1][..., :topk]
            indices_sorted = np.argsort(-probs, kind='mergesort', axis=-1)[..., :topk]
            probs_sorted_sumed = np.cumsum(probs_sorted, axis=-1, dtype=np.float16).astype(np.float16)
            mask = probs_sorted_sumed < topp
            topp_idx = np.sum(mask, axis=-1, keepdims=True) - 1
            topp_idx[topp_idx < 0] = 0
            topp_value = np.take_along_axis(probs_sorted_sumed, topp_idx, axis=-1)
            topp_value *= np.array(rand_list).reshape(-1, 1)[0:probs.shape[0]]
            mask1 = probs_sorted_sumed < topp_value
            sample_idx = np.sum(mask1, axis=-1, keepdims=True) - 1
            sample_idx[sample_idx < 0] = 0
            indices_sampled = np.take_along_axis(indices_sorted, sample_idx, axis=-1)
            probs_sampled = np.take_along_axis(probs_sorted, sample_idx, axis=-1)
            return [torch.from_numpy(indices_sampled.astype(np.int32)),
                    torch.from_numpy(probs_sampled.astype(np.float16))]
        else:
            if topk == 0:
                topk = probs.shape[-1]
            probs_sorted = np.sort(probs, axis=-1)[..., ::-1][..., :topk]
            probs_div_sorted = probs_sorted / topp
            indices_sorted = np.argsort(-probs, kind='mergesort', axis=-1)[..., :topk]
            probs_sorted_sumed = np.cumsum(probs_sorted, axis=-1, dtype=np.float16).astype(np.float16)
            mask = np.zeros_like(probs_sorted_sumed, dtype=np.int32)
            mask[probs_sorted_sumed <= topp] = 1
            probs_div_sorted *= mask
            probs_div_sorted_sumed = np.cumsum(probs_div_sorted, axis=-1, dtype=np.float16).astype(np.float16)
            multinomial_probs = probs_div_sorted_sumed.astype(np.float32) < rand_list[0]
            first_true_indeces = np.argmax(~multinomial_probs, axis=-1)
            for i in range(probs.shape[0]):
                multinomial_probs[i, first_true_indeces[i]:] = False
            indices_sorted_sampled = np.sum(multinomial_probs, axis=-1, keepdims=True)
            indices_sorted_sampled[indices_sorted_sampled >= topk] = 0
            indices_sampled = np.take_along_axis(indices_sorted, indices_sorted_sampled, axis=-1)
            probs_sampled = np.take_along_axis(probs_sorted, indices_sorted_sampled, axis=-1)
            return [torch.from_numpy(indices_sampled.astype(np.int32)),
                    torch.from_numpy(probs_sampled.astype(np.float16))]

    def golden_calc(self, in_tensors):
        # div
        intermeadiate_div_out = in_tensors[0] / in_tensors[1]  # (4, 128)
        # softmax
        intermediate_softmax_out = F.softmax(intermeadiate_div_out, dim=-1) # (4, 128)
        # topkTopp
        topk_intensors = [intermediate_softmax_out, in_tensors[2]]
        outtensor0, outtensor1 = self.golden_topk_topp(topk_intensors)
        return [outtensor0, outtensor1]

    def test(self):
        in_tensors = [intensor0, intensor1, intensor2]
        self.execute(LAYER_NAME, PARAM, in_tensors)

if __name__ == '__main__':
    unittest.main()