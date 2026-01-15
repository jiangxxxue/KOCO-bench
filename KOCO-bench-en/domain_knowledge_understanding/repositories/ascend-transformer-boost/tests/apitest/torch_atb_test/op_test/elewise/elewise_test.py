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
    print("----------- elewise test begin ------------")
    input1_cpu = torch.randn(2, 3, dtype=torch.float16)
    input2_cpu = torch.randn(2, 3, dtype=torch.float16)
    input1_npu = input1_cpu.npu()
    input2_npu = input2_cpu.npu()
    elewise_param = torch_atb.ElewiseParam()
    elewise_param.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_ADD
    elewise = torch_atb.Operation(elewise_param)
    logging.info(elewise_param)

    def elewise_run():
        elewise_outputs = elewise.forward([input1_npu, input2_npu])
        return elewise_outputs

    def golden():
        return [input1_cpu+input2_cpu]

    cpu_goldens = golden()
    logging.info("cpu_goldens: ", cpu_goldens)

    npu_outputs = elewise_run()
    logging.info("npu_outputs: ", npu_outputs)
    
    assert check_float(npu_outputs, cpu_goldens), "Test failed"

    run_perf_test(elewise, [input1_npu, input2_npu])
    print("----------- elewise test success ------------")

class TestElewise(unittest.TestCase):
    def test(self):
        run_test()

if __name__ == "__main__":
    unittest.main()