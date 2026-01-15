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
from torch_atb_test.utils import run_perf_test, check_float
import unittest
import logging

def run_test():
    print("----------- concat test begin ------------")
    dim = 1
    intensor1_npu = torch.ones((3, 1, 3), dtype=torch.float16).npu()
    intensor2_npu = torch.ones((3, 1, 3), dtype=torch.float16).npu()
    concat_param = torch_atb.ConcatParam(concat_dim = dim)
    concat = torch_atb.Operation(concat_param)
    logging.info(concat_param)

    def concat_run():
        concat_outputs = concat.forward([intensor1_npu, intensor2_npu])
        return concat_outputs

    def golden():
        return [torch.concat((intensor1_npu, intensor2_npu), dim = dim).cpu()]

    cpu_goldens = golden()
    logging.info("cpu_goldens: ", cpu_goldens)

    npu_outputs = concat_run()
    logging.info("npu_outputs: ", npu_outputs)
    
    assert check_float(npu_outputs, cpu_goldens), "Test failed"
    run_perf_test(concat, [intensor1_npu, intensor2_npu])
    print("----------- concat test success ------------")

class TestConcat(unittest.TestCase):
    def test(self):
        run_test()

if __name__ == "__main__":
    unittest.main()