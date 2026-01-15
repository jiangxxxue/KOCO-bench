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


def main_worker(rank, world_size, trans_weight, d_types, sizes, bias):
    torch_npu.npu.set_device(rank)
    print(f'Process {rank} started, using device npu:{rank}.')

    acl_matmul_allreduce_operation = torch.classes.OperationTorch.OperationTorch(
        "LinearParallelOperation")

    acl_param = json.dumps({"rank": rank, "rankSize": world_size, "bias": bias, "parallelType": "RowParallel",
                            "rankRoot": 0, "transWeight": trans_weight, "backend": "lcoc"})

    acl_matmul_allreduce_operation.set_param(acl_param)
    # exec
    for d_type in d_types:
        for size in sizes:
            input_tensor = torch.clamp(torch.randn(size[0]), -1, 1).to(d_type)
            weight_tensor = torch.clamp(torch.randn(size[1]), -1, 1).to(d_type)
            in_tensors = [input_tensor.to(torch.npu.current_device()), weight_tensor.to(torch.npu.current_device())]
            if bias != "None":
                if trans_weight:
                    bias_tensor = torch.randn([size[1][0]]).to(d_type)
                    in_tensors.append(bias_tensor.to(torch.npu.current_device()))
                else:
                    bias_tensor = torch.randn([size[1][1]]).to(d_type)
                    in_tensors.append(bias_tensor.to(torch.npu.current_device()))
            if rank == 0:
                print("sizes: ", size, "d_type: ", d_type)

            acl_out_tensor = acl_matmul_allreduce_operation.execute(in_tensors)[0]
            torch.npu.synchronize()

            if trans_weight:
                if weight_tensor.dim() == 2:
                    weight_tensor = torch.transpose(weight_tensor, 0, 1)
                if weight_tensor.dim() == 3:
                    weight_tensor = torch.transpose(weight_tensor, 1, 2)

            matmul_result = torch.matmul(input_tensor.to(torch.float), weight_tensor.to(torch.float)).to(d_type)
            golden_out_tensor = matmul_result.clone()

            for i in range(world_size - 1):
                golden_out_tensor += matmul_result

            if bias != "None":
                golden_out_tensor += bias_tensor
            assert check_precision(acl_out_tensor.cpu(), golden_out_tensor, rank)


def check_precision(out_tensor, golden_out_tensor, rank, error_threshold=0.005, tol=0.001):
    relative_error = torch.abs(out_tensor - golden_out_tensor) / (torch.abs(golden_out_tensor) + 1e-8)
    absolute_error = torch.abs(out_tensor - golden_out_tensor)
    high_error_elements = (relative_error > error_threshold) | (absolute_error > tol)
    result = torch.sum(high_error_elements).item() / out_tensor.numel()

    if rank == 0:
        print("high_error_num: ", torch.sum(high_error_elements).item())
        print("result: ", result)

    if not result < error_threshold and rank == 0:
        print("out_tensor.shape", out_tensor.shape,
              "\ngolden_out_tensor.shape:", golden_out_tensor.shape)
        print("out_tensor:", out_tensor,
              ", \ngolden_out_tensor:", golden_out_tensor)

    return result < error_threshold


class LinearParallelCoverOperationTest(operation_test.OperationTest):

    def test_linear_paraller_cover(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            return
        print(f"———————— LinearParallelCoverOp normal test start ————————")
        world_size = 8
        device_available = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        if device_available:
            device_num = len(device_available.split(","))
            if world_size > device_num:
                self.skipTest(f"Skipped because world_size {world_size} > available devices {device_num}")
        d_types = [torch.float16, torch.bfloat16]

        sizes = [[[27, 333], [333, 77]],
                 [[140, 1024], [1024, 8192]],
                 [[32, 2752], [2752, 8192]],
                 [[64, 8192], [8192, 3072]],
                 [[140, 8192], [8192, 5504]],
                 [[1024, 8192], [8192, 5504]]]
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, True, d_types, sizes, "None"))

    def test_linear_paraller_cover_trans_weight(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            return
        print(f"———————— LinearParallelCoverOp transB test start ————————")
        world_size = 8
        device_available = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        if device_available:
            device_num = len(device_available.split(","))
            if world_size > device_num:
                self.skipTest(f"Skipped because world_size {world_size} > available devices {device_num}")
        d_types = [torch.float16]
        sizes = [[[140, 1024], [8192, 1024]],
                 [[28, 5, 1024], [8192, 1024]],
                 [[28, 5, 2752], [8192, 2752]],
                 [[32, 2752], [8192, 2752]],
                 [[64, 8192], [3072, 8192]],
                 [[140, 8192], [5504, 8192]],
                 [[1024, 8192], [5504, 8192]]]
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, False, d_types, sizes, "None"))

    def test_linear_paraller_cover_bias(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            return
        print(f"———————— LinearParallelCoverOp bias test start ————————")
        world_size = 2
        d_types = [torch.float16, torch.bfloat16]
        sizes = [[[27, 333], [333, 77]],
                 [[140, 8192], [8192, 5504]],
                 [[1024, 8192], [8192, 5504]]]
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, True, d_types, sizes, "yes"))


if __name__ == '__main__':
    os.environ["ASCEND_RT_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
    unittest.main()
