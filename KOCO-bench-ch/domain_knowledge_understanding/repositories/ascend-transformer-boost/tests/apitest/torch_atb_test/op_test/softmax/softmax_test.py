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
    print("----------- softmax test begin ------------")
    axes = [0]
    input_cpu = torch.randn(2, 16, 256, dtype=torch.float32)
    input_npu = input_cpu.npu()
    softmax_param = torch_atb.SoftmaxParam()
    softmax_param.axes = axes
    softmax = torch_atb.Operation(softmax_param)
    logging.info(softmax_param)
    def softmax_run():
        softmax_outputs = softmax.forward([input_npu])
        return softmax_outputs

    def golden():
        in_tensor_dim_num = input_cpu.dim()
        axes2 = [ i % in_tensor_dim_num for i in axes ]
        target_shape = input_cpu.shape[axes2[0]:axes2[-1] + 1]
        in_tensor_flatten = input_cpu.flatten(start_dim=axes2[0], end_dim=axes2[-1])
        softmax0 = torch.nn.Softmax(dim=axes2[0])
        out_tensor = softmax0(in_tensor_flatten)
        out_tensor = out_tensor.unflatten(axes2[0], target_shape)
        return [out_tensor]

    cpu_goldens = golden()
    logging.info("cpu_goldens: ", cpu_goldens)

    npu_outputs = softmax_run()
    logging.info("npu_outputs: ", npu_outputs)
    
    assert check_float(npu_outputs, cpu_goldens), "Test failed"
    run_perf_test(softmax, [input_npu])
    print("----------- softmax test success ------------")

class TestSoftmax(unittest.TestCase):
    def test(self):
        run_test()

if __name__ == "__main__":
    unittest.main()