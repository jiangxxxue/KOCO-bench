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
from torch_atb_test.utils import check_move, run_perf_test
import unittest
import logging

def run_test():
    print("----------- split test begin ------------")
    splitDim = 0
    splitNum = 2
    input_cpu = torch.randn(6, 6, dtype=torch.float16)
    input_npu = input_cpu.npu()
    split_param = torch_atb.SplitParam()
    split_param.split_dim = splitDim
    split_param.split_num = splitNum
    logging.info(split_param)
    split = torch_atb.Operation(split_param)
    
    def split_run():
        split_outputs = split.forward([input_npu])
        return split_outputs

    split_output = torch.chunk(input_cpu, chunks=splitNum, dim=splitDim)
    cpu_goldens = torch.stack(split_output)
    logging.info("cpu_goldens: ", cpu_goldens)

    npu_outputs = split_run()
    logging.info("npu_outputs: ", npu_outputs)
    
    assert check_move(npu_outputs, cpu_goldens), "Test failed"
    run_perf_test(split, [input_npu])
    print("----------- split test success ------------")

class TestSplit(unittest.TestCase):
    def test(self):
        run_test()

if __name__ == "__main__":
    unittest.main()