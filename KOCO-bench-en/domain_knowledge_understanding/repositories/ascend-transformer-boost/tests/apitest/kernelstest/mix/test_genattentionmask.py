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
import random
import logging
import numpy as np
import torch
import sys, os
import torch.nn.functional as F
import op_test

OP_NAME = "GenAttentionMaskOperation"
OP_PARAM = {"headNum": 0, "qSeqLen": None}

class  TestGenAttentionMask(op_test.OpTest):
    def golden_calc(self, in_tensors):
        out = []
        for i, s in enumerate(OP_PARAM['qSeqLen']):
            for _ in range(OP_PARAM["headNum"]):
                out.append(in_tensors[0][i, :, :s, :s].flatten())
        logging.debug(f"torch.hstack(out): {torch.hstack(out)}")
        return [torch.hstack(out)]

    def golden_compare(self, out_tensors, golden_out_tensors):
        return torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)

    @op_test.only_910b
    def test_genattentionmask_half1(self):
        OP_PARAM["headNum"] = 2
        OP_PARAM["qSeqLen"] = [2, 1]
        self.set_param(OP_NAME, OP_PARAM)
        batch = 2
        maxseqlen = 4
        shapeOut = sum(map(lambda x: x**2, OP_PARAM['qSeqLen'])) * OP_PARAM["headNum"]
        input_tensor = torch.randint(1, 10, (batch, 1, maxseqlen, maxseqlen)).half()
        input_list = [input_tensor]
        output_list = [torch.zeros(shapeOut).half()]
        self.execute(input_list, output_list)

if __name__ == '__main__':
    unittest.main()
