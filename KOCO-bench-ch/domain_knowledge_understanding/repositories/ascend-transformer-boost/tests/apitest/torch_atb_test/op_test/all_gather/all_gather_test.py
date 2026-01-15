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
import torch_npu
import torch_atb
import torch.multiprocessing as mp
import sys
import os
import re
from torch_atb_test.utils import check_float, run_perf_test
import unittest
import logging

rank_size = 2
rank_root = 0
backend = "hccl"
random_seed = 123

def all_gather_run(rank, tensor):
    all_gather_param = torch_atb.AllGatherParam(
    rank = rank, rank_size = rank_size, rank_root = rank_root, backend = backend)
    all_gather = torch_atb.Operation(all_gather_param)
    result = all_gather.forward([tensor])
    logging.info(all_gather_param)
    run_perf_test(all_gather, [tensor])
    return result

def is910B():
    device_name = torch.npu.get_device_name()
    return (re.search("Ascend910B", device_name, re.I) and len(device_name) > 10)

def run_test(rank, size):
    torch_npu.npu.set_device(rank)
    logging.info(f'Process {rank} started, using device npu:{rank}.')
    torch.manual_seed(random_seed)

    inTensors = []
    for _ in range(rank_size):
        inTensor = torch.rand(1, 2, dtype=torch.float16)
        inTensors.append(inTensor)

    cpu_goldens = [torch.stack(inTensors, dim=0)]
    logging.info("cpu_goldens: ", cpu_goldens)

    npu_outputs = all_gather_run(rank, inTensors[rank].npu())
    torch.npu.synchronize()
    logging.info("npu_outputs: ", npu_outputs)
    
    assert check_float(npu_outputs, cpu_goldens), "Test failed"

class TestAllGather(unittest.TestCase):
    def test(self):
        if not is910B():
            print("This test case only supports 910B")
            return True
        print("----------- all_gather test begin ------------")
        mp.spawn(run_test, nprocs=rank_size, args=(rank_size,))
        print("----------- all_gather test success ------------")

if __name__ == "__main__":
    unittest.main()