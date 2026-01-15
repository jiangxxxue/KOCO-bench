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
import numpy as np
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
    print("----------- rms_norm test begin ------------")
    rms_norm_param = torch_atb.RmsNormParam()
    rms_norm_param.layer_type = torch_atb.RmsNormParam.RmsNormType.RMS_NORM_NORM
    rms_norm_param.norm_param.rstd = True
    rms_norm = torch_atb.Operation(rms_norm_param)
    logging.info(rms_norm_param)
    epsilon = 1e-5
    shape=[8, 8, 8]
    shape_gamma=[8]
    shape_rstd=[8, 8, 1]
    x = torch.from_numpy(np.random.uniform(low=0, high=100, size=shape).astype(np.float32))
    gamma = torch.from_numpy(np.random.uniform(low=0, high=100, size=shape_gamma).astype(np.float32))
    in_tensors = [x.npu(), gamma.npu()]

    def rms_norm_run():
        rms_norm_outputs = rms_norm.forward(in_tensors)
        return rms_norm_outputs

    def golden():
        reduceDims=[]
        edim = x.dim()-gamma.dim()
        for i in range(gamma.dim()):
            reduceDims.append(edim + i)
        rstd = torch.rsqrt(x.pow(2).mean(reduceDims, keepdim=True) + epsilon)
        result = x * rstd
        result = result * gamma
        return [result, rstd]

    cpu_goldens = golden()
    logging.info("cpu_goldens: ", cpu_goldens)
    npu_outputs = rms_norm_run()
    logging.info("npu_outputs: ", npu_outputs)

    assert check_float(npu_outputs, cpu_goldens), "Test failed"
    run_perf_test(rms_norm, in_tensors)
    print("----------- rms_norm test success ------------")

class TestRMSNorm(unittest.TestCase):
    def test(self):
        run_test()

if __name__ == "__main__":
    unittest.main()