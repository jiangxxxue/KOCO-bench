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
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
# usage:
# export HCCL_WHITELIST_DISABLE=1
# python3 -m unittest test_all_gather_operation.py
# Attention: when you use lccl backend, unset HCCL_MTE_ENABLE and copy lcal.o to current directory

ATB_HOME_PATH = os.environ.get("ATB_HOME_PATH")
if ATB_HOME_PATH is None:
    raise RuntimeError(
        "env ATB_HOME_PATH not exist, source set_env.sh")
LIBTORCH_PATH = os.path.join(ATB_HOME_PATH, "lib/libatb_test_framework.so")
LIB_PATH = os.path.join(ATB_HOME_PATH, "lib/libatb.so")
torch.classes.load_library(LIBTORCH_PATH)


def main_worker(rank, world_size,inTensorDtypes,sizes,random_seed):
    torch_npu.npu.set_device(rank)
    print(f'Process {rank} started, using device npu:{rank}.')
    # init all gather operation
    acl_allgather_operation1 = torch.classes.OperationTorch.OperationTorch(
        "AllGatherOperation")
    acl_param1 = json.dumps({"rank": rank, "rankSize": world_size,
                            "rankRoot": 0, "backend": "hccl"})
    acl_allgather_operation1.set_param(acl_param1)
    acl_allgather_operation2 = torch.classes.OperationTorch.OperationTorch(
        "AllGatherOperation")
    acl_param2 = json.dumps({"rank": rank, "rankSize": world_size,
                            "rankRoot": 0, "backend": "lccl"})
    acl_allgather_operation2.set_param(acl_param2)
    # exec all gather
    torch.manual_seed(random_seed)
    low = -100
    high = 100
    for inTensorDtype in inTensorDtypes:
        for size in sizes:
            inTensors = []
            for i in range(world_size):
                inTensor = ((high - low) * torch.rand(size) + low).type(inTensorDtype)
                inTensors.append(inTensor)
            inTensor = inTensors[rank].clone().npu()
            acl_out_tensor1 = acl_allgather_operation1.execute([inTensor])[0]
            torch.npu.synchronize()
            acl_out_tensor2 = acl_allgather_operation2.execute([inTensor])[0]
            torch.npu.synchronize()
            # assert result
            golden_out_tensor = torch.stack(inTensors, dim=0)
            assert golden_compare(acl_out_tensor1.cpu(), golden_out_tensor.cpu())
            assert golden_compare(acl_out_tensor2.cpu(), golden_out_tensor.cpu())

def golden_compare(out_tensor, golden_out_tensor, rtol=0.0001, atol=0.0001):
    result = torch.allclose(out_tensor, golden_out_tensor, rtol=rtol, atol=atol)
    if not result:
        print("out_tensor.shape", out_tensor.shape,
            "\ngolden_out_tensor.shape:", golden_out_tensor.shape)
        print("out_tensor:", out_tensor,
            ", \ngolden_oute_tensor:", golden_out_tensor)
    return result


class AllGatherOperationTest(operation_test.OperationTest):
    def test_all_gather(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        command = f"nm -D {LIB_PATH} | grep HcclAllGather > /dev/null"
        res = os.system(command)
        if res == 0:
            world_size = 2
            random_seed = 123
            inTensorDtypes = [torch.int8,torch.int16, torch.int32,torch.int64,torch.float32, torch.float16, torch.bfloat16]
            sizes = [[3,4,5],[10,100,512]]
            mp.spawn(main_worker, nprocs=world_size, args=(world_size,inTensorDtypes,sizes,random_seed))
        else:
            print("hccl_runner is not compiled, skip AllGatherOperationTest")
    


if __name__ == '__main__':
    unittest.main()
