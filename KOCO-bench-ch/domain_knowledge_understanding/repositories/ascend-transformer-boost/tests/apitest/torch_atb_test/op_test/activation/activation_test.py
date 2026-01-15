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
import re
from torch_atb_test.utils import check_float, run_perf_test
import unittest
import logging

def is910B():
    device_name = torch.npu.get_device_name()
    return (re.search("Ascend910B", device_name, re.I) and len(device_name) > 10)

def run_test():
    if not is910B():
        print("This test case only supports 910B")
        return True
    print("----------- activation test begin ------------")
    activation_param = torch_atb.ActivationParam()
    activation_param.activation_type = torch_atb.ActivationType.ACTIVATION_SWISH
    activation_param.scale = 1.0
    activation = torch_atb.Operation(activation_param)
    logging.info(activation_param)

    intensor = torch.rand(2, 3, 5).bfloat16()
    intensor_npu = intensor.npu()

    def activation_run():
        activation_outputs = activation.forward([intensor_npu])
        return activation_outputs

    def golden():
        intensor_float = intensor.float()
        outtensor = intensor_float / (1 + torch.exp(-intensor_float * 1.0))
        return [outtensor.bfloat16()]

    cpu_goldens = golden()
    logging.info("cpu_goldens: ", cpu_goldens)

    npu_outputs = activation_run()
    logging.info("npu_outputs: ", npu_outputs)
    
    assert check_float(npu_outputs, cpu_goldens), "Test failed"

    run_perf_test(activation, [intensor_npu])
    print("----------- activation test success ------------")

class TestActivation(unittest.TestCase):
    def test(self):
        run_test()

if __name__ == "__main__":
    unittest.main()