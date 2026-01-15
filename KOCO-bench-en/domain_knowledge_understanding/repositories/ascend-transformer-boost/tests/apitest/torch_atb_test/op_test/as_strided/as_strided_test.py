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
import torch_atb
import sys
import os
from torch_atb_test.utils import check_float, run_perf_test
import unittest
import logging

def run_test():
    print("----------- as_strided test begin ------------")
    size = [2, 2]
    stride = [2, 3]
    offset = [0]
    intensor_cpu = torch.randn(3, 3, dtype=torch.float16)
    intensor_npu = intensor_cpu.npu()
    as_strided_param = torch_atb.AsStridedParam()
    as_strided_param.size = size
    as_strided_param.stride = stride
    as_strided_param.offset = offset
    as_strided = torch_atb.Operation(as_strided_param)
    logging.info(as_strided_param)

    def as_strided_run():
        as_strided_outputs = as_strided.forward([intensor_npu])
        return as_strided_outputs

    def golden():
        return [torch.as_strided(intensor_cpu, size, stride, offset[0])]

    cpu_goldens = golden()
    logging.info("cpu_goldens: ", cpu_goldens)

    npu_outputs = as_strided_run()
    logging.info("npu_outputs: ", npu_outputs)
    
    assert check_float(npu_outputs, cpu_goldens), "Test failed"
    run_perf_test(as_strided, [intensor_npu])
    print("----------- as_strided test success ------------")

class TestAsStrided(unittest.TestCase):
    def test(self):
        run_test()

if __name__ == "__main__":
    unittest.main()