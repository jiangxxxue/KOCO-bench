#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import builtins
import os
import json
import unittest
import sys
import numpy as np
import torch
import torch_npu
import torch.multiprocessing as mp

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from precision_calcu import *

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../python/operations/"))
import operation_test  # NOQA: E402

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../python/"))

ATB_HOME_PATH = os.environ.get("ATB_HOME_PATH")
if ATB_HOME_PATH is None:
    raise RuntimeError(
        "env ATB_HOME_PATH not exist, source set_env.sh")
LIBTORCH_PATH = os.path.join(ATB_HOME_PATH, "lib/libatb_test_framework.so")
LIB_PATH = os.path.join(ATB_HOME_PATH, "lib/libatb.so")
torch.classes.load_library(LIBTORCH_PATH)

DEV_NUM = 2

M = 2
K = 256
N = 2

DATA_TYPE = torch.float16

def load_tensor(data_size,data_type,data_path):
    with open(data_path, 'rb') as f:
        data=f.read()
    if data_type == torch.float16:
        np_data = np.frombuffer(data, dtype=np.float16).copy()
        tensor = torch.from_numpy(np_data)
    elif data_type == torch.bfloat16:
        tensor  = torch.frombuffer(bytearray(data), dtype=torch.bfloat16)
    else:
        tensor = torch.zeros(data_size)

    tensor = tensor.view(data_size)
    
    return tensor


def main_worker(rank, world_size, data_type, data_size):
    torch_npu.npu.set_device(rank)
    print(f'Process {rank} started, using device npu:{rank}.')
    golden_out_tensor_high = None
    golden_out_tensor_low = None

    for i in range(world_size):
        input_tensor = load_tensor(data_size=data_size[0],data_type=data_type,data_path=f"rank{i}_inTensor{0}.bin")
        weight_tensor = load_tensor(data_size=data_size[1],data_type=data_type,data_path=f"rank{i}_inTensor{1}.bin")
        out_single_tensor = torch.matmul(input_tensor.to(torch.float), weight_tensor.to(torch.float))
        if golden_out_tensor_high is None:
            golden_out_tensor_high = out_single_tensor.clone()
            golden_out_tensor_low = out_single_tensor.clone().to(data_type)
            in_tensors_desc = [input_tensor.shape, weight_tensor.shape]
        else:
            golden_out_tensor_high = torch.add(golden_out_tensor_high,out_single_tensor)
            golden_out_tensor_low = torch.add(golden_out_tensor_low,out_single_tensor.to(data_type))
    chunks_size = int(data_size[0][0] // world_size)
    chunks_high = torch.split(golden_out_tensor_high, chunks_size)
    chunks_low = torch.split(golden_out_tensor_low, chunks_size)
    golden_result_high = chunks_high[rank]
    golden_result_low = chunks_low[rank]
    
    acl_out_tensor = load_tensor(data_size=data_size[2],data_type=data_type,data_path=f"rank{rank}_outTensor{0}.bin")

    assert check_precision_new(in_tensors_desc, acl_out_tensor.float(), golden_result_high.float(), golden_result_low.float(), rank)

def check_precision_new(in_tensors_desc, out_tensor, golden_out_tensor_high, golden_out_tensor_low, rank):
    if rank == 0:
        print(in_tensors_desc)
        print(out_tensor)
    result_double = compare_cv(golden_out_tensor_high, golden_out_tensor_low, out_tensor)
    return result_double

class LinearParallelCoverOperationTest(operation_test.OperationTest):

    def test_linear_parallel(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            return
        print(f"———————— LinearParallelCoverOp test start ————————")

        world_size = DEV_NUM

        data_type = DATA_TYPE

        data_size = [[M, K], [K, N], [M // DEV_NUM, N]]
        
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, data_type, data_size))
        

if __name__ == '__main__':
    unittest.main()
