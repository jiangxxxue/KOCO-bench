#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import torch
import torch.nn as nn
import torch_atb
import sys
import os
from torch_atb_test.utils import check_float, run_perf_test
import unittest
import logging

def run_test():
    print("----------- rope test begin ------------")
    rope_param = torch_atb.RopeParam()
    rope_param.rotary_coeff = 4
    rope = torch_atb.Operation(rope_param)
    logging.info(rope_param)

    def gen_inputs():
        ntoken = 4
        seqlen = 4
        hidden_size = 4096
        head_size = 128
        intensor0 = torch.rand(ntoken, hidden_size).half()
        intensor1 = torch.rand(ntoken, hidden_size).half()
        intensor2 = torch.rand(ntoken, head_size).half()
        intensor3 = torch.rand(ntoken, head_size).half()
        intensor4 = torch.tensor([seqlen], dtype=torch.int32)
        return [intensor0, intensor1, intensor2, intensor3, intensor4]

    in_tensors = gen_inputs()
    in_tensors_npu = [tensor.npu() for tensor in in_tensors]

    def rope_run():
        rope_outputs = rope.forward(in_tensors_npu)
        return rope_outputs

    def golden():
        def rotate_half(x):
            x0, x1 = x.chunk(2, -1)
            return torch.cat((-x1, x0), dim=x0.ndim - 1)

        ntoken = in_tensors[0].size()[0]
        seqlen = int(in_tensors[4][0])
        batch = ntoken // seqlen
        hidden_size = in_tensors[0].size()[1]
        head_size = in_tensors[2].size()[1]
        head_num = hidden_size // head_size
        qlayer = in_tensors[0].view(seqlen, batch, head_num, head_size)
        q0, q1 = qlayer.chunk(2, -1)
        klayer = in_tensors[1].view(seqlen, batch, head_num, head_size)
        k0, k1 = klayer.chunk(2, -1)
        cos0, cos1 = in_tensors[2].unsqueeze(1).unsqueeze(1).chunk(2, -1)
        sin0, sin1 = in_tensors[3].unsqueeze(1).unsqueeze(1).chunk(2, -1)
        q0 = (q0 * cos0) + (rotate_half(q0) * sin0)
        k0 = (k0 * cos0) + (rotate_half(k0) * sin0)
        q1 = (q1 * cos1) + (rotate_half(q1) * sin1)
        k1 = (k1 * cos1) + (rotate_half(k1) * sin1)
        q = torch.concat([q0, q1], dim=(q0.ndim - 1)).view(ntoken, hidden_size)
        k = torch.concat([k0, k1], dim=(k0.ndim - 1)).view(ntoken, hidden_size)
        return [q, k]

    cpu_goldens = golden()
    logging.info("cpu_goldens: ", cpu_goldens)

    npu_outputs = rope_run()
    logging.info("npu_outputs: ", npu_outputs)
    
    assert check_float(npu_outputs, cpu_goldens), "Test failed"
    run_perf_test(rope, in_tensors_npu)
    print("----------- rope test success ------------")

class TestRope(unittest.TestCase):
    def test(self):
        run_test()

if __name__ == "__main__":
    unittest.main()