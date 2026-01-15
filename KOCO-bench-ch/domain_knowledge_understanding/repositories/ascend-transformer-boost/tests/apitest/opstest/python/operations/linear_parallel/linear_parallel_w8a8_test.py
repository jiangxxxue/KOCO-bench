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
def main_worker(rank, world_size, trans_weight, d_types, sizes, out_type):
    torch_npu.npu.set_device(rank)
    print(f'Process {rank} started, using device npu:{rank}.')

    acl_matmul_allreduce_operation = torch.classes.OperationTorch.OperationTorch(
        "LinearParallelOperation")

    acl_param = json.dumps({"type": 0, "rank": rank, "rankSize": world_size,
                            "rankRoot": 0, "transWeight": trans_weight, "backend": "lcoc",
                            "quantType": 1,"outDataType": out_type})

    acl_matmul_allreduce_operation.set_param(acl_param)
    # exec
    for d_type in d_types:
        for size in sizes:
            input_tensor = torch.clamp(torch.randn(size[0]), -10, 10).to(d_type)
            weight_tensor = torch.clamp(torch.randn(size[1]), -10, 10).to(d_type)
            n = size[1][1]
            if trans_weight:
                n = size[1][0]

            in_tensors = [input_tensor.to(torch.device('npu')), weight_tensor.to(torch.device('npu'))]

            bias_tensor = torch.randn([n]).to(torch.int32)
            in_tensors.append(bias_tensor.to(torch.device('npu')))

            scale_fp16_tensor = (5 - 3) * torch.rand(n, dtype=torch.float16) + 3
            scale_fp32_tensor = scale_fp16_tensor.to(torch.float32)
            scale_tensor = torch.stack((scale_fp32_tensor, torch.zeros_like(scale_fp32_tensor)), dim=-1).flatten().view(torch.int64).contiguous()
            in_tensors.append(scale_tensor.to(torch.device('npu')))

            acl_out_tensor = acl_matmul_allreduce_operation.execute(in_tensors)[0]
            torch.npu.synchronize()

            if trans_weight:
                if weight_tensor.dim() == 2:
                    weight_tensor = torch.transpose(weight_tensor, 0, 1)
                if weight_tensor.dim() == 3:
                    weight_tensor = torch.transpose(weight_tensor, 1, 2)

            matmul_result = torch.matmul(input_tensor.to(torch.float), weight_tensor.to(torch.float))
            # quant
            golden_one = matmul_result.clone()
            golden_one += bias_tensor.to(torch.float)
            golden_one *= scale_fp32_tensor
            golden_out_tensor = golden_one.to(torch.float).clone()
            # all reduce
            for i in range(world_size - 1):
                golden_out_tensor += golden_one

            err = 2**-8 if out_type == torch.float16 else 2**-7
            assert check_precision_new(acl_out_tensor.cpu().float(), golden_out_tensor.float(), rank)

def check_precision_new(out_tensor, golden_out_tensor, rank, err = 2**-8):
    # 计算每个元素的误差阈值
    max_err = err * torch.max(torch.ones_like(golden_out_tensor), torch.abs(golden_out_tensor))

    # 计算实际误差
    error = torch.abs(out_tensor - golden_out_tensor)

    # 计算不满足条件的元素个数
    num_failures = torch.sum(error > max_err).item()
    if rank == 0:
        print("num_failures: ", num_failures)
    return num_failures == 0

class LinearParallelCoverOperationTest(operation_test.OperationTest):

    def test_linear_paraller_fp16(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            return
        print(f"———————— LinearParallelCoverOp test start ————————")
        world_size = 8
        device_available = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        if device_available:
            device_num = len(device_available.split(","))
            if world_size > device_num:
                self.skipTest(f"Skipped because world_size {world_size} > available devices {device_num}")
        d_types = [torch.int8]

        sizes = [[[27, 333], [333, 77]],
                 [[2, 16], [16, 48]],
                 [[32768,3904],[3904,12288]],
                 [[2048,1536],[1536,12288]],
                 [[16384,12288],[12288,7808]],
                 [[140, 1024], [1024, 8192]],
                 [[32, 2752], [2752, 8192]],
                 [[64, 8192], [8192, 3072]],
                 [[140, 8192], [8192, 5504]],
                 [[1024, 8192], [8192, 5504]]]

        acl_float16 = 1
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, False, d_types, sizes, acl_float16))

if __name__ == '__main__':
    unittest.main()
