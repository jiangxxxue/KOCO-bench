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
import re
from torch_atb_test.utils import run_perf_test
import unittest
import logging

def is910B():
    device_name = torch.npu.get_device_name()
    return (re.search("Ascend910B", device_name, re.I) and len(device_name) > 10)

def run_test():
    if not is910B():
        print("This test case only supports 910B")
        return True
    print("----------- multinomial test begin ------------")
    rand_seed = 123
    intensor = torch.rand(3, 3, dtype=torch.float16)
    normalized_tensor = intensor / intensor.sum()
    normalized_tensor_npu = normalized_tensor.npu()
    multinomial_param = torch_atb.MultinomialParam(rand_seed = rand_seed)
    multinomial = torch_atb.Operation(multinomial_param)
    logging.info(multinomial_param)

    def multinomial_run():
        multinomial_outputs = multinomial.forward([normalized_tensor_npu])
        return multinomial_outputs

    npu_outputs = multinomial_run()
    logging.info("npu_outputs: ", npu_outputs)

    run_perf_test(multinomial, [normalized_tensor_npu])
    print("----------- multinomial test success ------------")

class TestMultinomial(unittest.TestCase):
    def test(self):
        run_test()

if __name__ == "__main__":
    unittest.main()