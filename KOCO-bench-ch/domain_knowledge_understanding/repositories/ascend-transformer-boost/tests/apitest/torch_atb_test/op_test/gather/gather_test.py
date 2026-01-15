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
    print("----------- gather test begin ------------")
    axis = 1
    gather_param = torch_atb.GatherParam()
    gather_param.axis = axis
    gather = torch_atb.Operation(gather_param)
    logging.info(gather_param)
    intensor0 = torch.randn([3,5],dtype=torch.float16)
    intensor0_npu = intensor0.npu()
    intensor1 = torch.randint(0, 5, [3,4],dtype=torch.int64)
    intensor1_npu = intensor1.npu()

    def gather_run():
        gather_outputs = gather.forward([intensor0_npu, intensor1_npu])
        return gather_outputs

    def golden():
        outputSize = []
        dim0 = 1
        for i in range(0,axis):
            outputSize.append(intensor0.shape[i])
            dim0 *= intensor0.shape[i]
        dim1 = intensor0.shape[axis]
        for i in range(0,intensor1.ndim):
            outputSize.append(intensor1.shape[i])
        dim2 = 1
        for i in range(axis + 1,intensor0.ndim):
            outputSize.append(intensor0.shape[i])
            dim2 *= intensor0.shape[i]
        inputFlatten = intensor0.clone().reshape(-1)
        indicesFlatten = intensor1.clone().reshape(-1)
        golden_result_np = torch.zeros(outputSize,dtype=torch.float16).reshape(-1).numpy()
        idx = 0
        for i in range(0,dim0):
            inputIdx = i * dim1 * dim2
            for indice in indicesFlatten:
                for k in range(0,dim2):
                    golden_result_np[idx] = inputFlatten[inputIdx + indice * dim2 + k]
                    idx+=1
        golden_result = torch.from_numpy(golden_result_np).reshape(outputSize)
        logging.info(intensor0.dtype)
        if intensor0.dtype == torch.bfloat16:
            golden_result = golden_result.bfloat16()
        return [golden_result]

    cpu_goldens = golden()
    logging.info("cpu_goldens: ", cpu_goldens)

    npu_outputs = gather_run()
    logging.info("npu_outputs: ", npu_outputs)
    
    assert check_move(npu_outputs, cpu_goldens), "Test failed"

    run_perf_test(gather, [intensor0_npu, intensor1_npu])
    print("----------- gather test success ------------")

class TestGather(unittest.TestCase):
    def test(self):
        run_test()

if __name__ == "__main__":
    unittest.main()