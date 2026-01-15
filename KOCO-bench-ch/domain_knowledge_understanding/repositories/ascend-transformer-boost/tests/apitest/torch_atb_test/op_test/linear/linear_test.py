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
    print("----------- linear test begin ------------")
    m, k, n = 3, 4, 5
    input_tensor = torch.randn(m, k, dtype=torch.float16)
    weight_tensor = torch.randn(k, n, dtype=torch.float16)
    input = input_tensor.npu()
    weight = weight_tensor.npu()

    linear_param = torch_atb.LinearParam()
    linear_param.has_bias = False
    linear_param.transpose_b = False
    linear = torch_atb.Operation(linear_param)
    logging.info(linear_param)

    def linear_run():
        linear_outputs = linear.forward([input, weight])
        return [linear_outputs[0].to(torch.float32)]
    
    def golden():
        cpu_golden = torch.matmul(input_tensor.to(torch.float32), weight_tensor.to(torch.float32))
        return [cpu_golden]

    cpu_goldens = golden()
    logging.info("cpu_goldens: ", cpu_goldens)

    npu_outputs = linear_run()
    logging.info("npu_outputs: ", npu_outputs)

    assert check_float(npu_outputs, cpu_goldens), "Test failed"

    run_perf_test(linear, [input, weight])
    print("----------- linear test success ------------")

class TestLinear(unittest.TestCase):
    def test(self):
        run_test()

if __name__ == "__main__":
    unittest.main()