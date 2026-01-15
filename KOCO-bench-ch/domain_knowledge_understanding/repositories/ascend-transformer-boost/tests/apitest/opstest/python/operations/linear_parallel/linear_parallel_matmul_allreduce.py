#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import os
import json
import unittest
import sys
import torch
import torch_npu
import torch.multiprocessing as mp
from precision_calcu import *

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

ATB_HOME_PATH = os.environ.get("ATB_HOME_PATH")
if ATB_HOME_PATH is None:
    raise RuntimeError(
        "env ATB_HOME_PATH not exist, source set_env.sh")
LIBTORCH_PATH = os.path.join(ATB_HOME_PATH, "lib/libatb_test_framework.so")
LIB_PATH = os.path.join(ATB_HOME_PATH, "lib/libatb.so")
torch.classes.load_library(LIBTORCH_PATH)

torch.manual_seed(0)



def main_worker(rank, world_size, data_types, data_sizes, data_gen_ranges):
    torch_npu.npu.set_device(rank)
    print(f'Process {rank} started, using device npu:{rank}.')

    acl_matmul_allreduce_operation = torch.classes.OperationTorch.OperationTorch(
        "LinearParallelOperation")

    acl_param = json.dumps({"type": 0, "rank": rank, "rankSize": world_size,
                            "rankRoot": 0, "transWeight": False, "backend": "lcoc",
                            "quantType": -1,"outDataType": -1})

    acl_matmul_allreduce_operation.set_param(acl_param)
    # exec
    for data_type in data_types:
        for data_size in data_sizes:
            for data_gen_range in data_gen_ranges:
                input_tensor = torch.clamp(torch.randn(data_size[0]), data_gen_range[0], data_gen_range[1]).to(data_type)
                weight_tensor = torch.clamp(torch.randn(data_size[1]), data_gen_range[0], data_gen_range[1]).to(data_type)

                in_tensors_desc = [input_tensor.shape,weight_tensor.shape]

                in_tensors = [input_tensor.to(torch.device('npu')), weight_tensor.to(torch.device('npu'))]

                acl_out_tensor = acl_matmul_allreduce_operation.execute(in_tensors)[0]
                torch.npu.synchronize()

                matmul_result_high = torch.matmul(input_tensor.to(torch.float), weight_tensor.to(torch.float))
                matmul_result_low = torch.matmul(input_tensor, weight_tensor)

                golden_one_high = matmul_result_high.clone()
                golden_one_low = matmul_result_low.clone()

                golden_out_tensor_high = golden_one_high.clone()
                golden_out_tensor_low = golden_one_low.clone()

                # all reduce

                for i in range(world_size-1):
                    golden_out_tensor_high += golden_one_high
                    golden_out_tensor_low += golden_one_low

                assert check_precision_new(in_tensors_desc, acl_out_tensor.float(), golden_out_tensor_high.npu().float(), golden_out_tensor_low.npu().float(), rank)

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

        world_size = 8

        data_types = [torch.bfloat16]

        data_sizes = [[[256, 1024], [1024, 8192]],
                      [[256, 3696], [3696, 8192]]]

        data_gen_ranges = [[-10, 10]]
        
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, data_types, data_sizes, data_gen_ranges))

if __name__ == '__main__':
    unittest.main()
