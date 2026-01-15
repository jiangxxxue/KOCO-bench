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
import socket
import random
import threading
from time import sleep
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import ReduceOp


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
os.environ["LCCL_DETERMINISTIC"]="1"
os.environ["LCCL_PARALLEL"]="1"

def main_worker(rank, world_size,inTensorDtypes, sizes, random_seed,golden_cal):
    # init process group
    torch_npu.npu.set_device(rank)
    print(f'Process {rank} started, using device npu:{rank}.')
    # init all reduce operation
    acl_allreduce_operation = torch.classes.OperationTorch.OperationTorch(
        "AllReduceOperation")
    comm_domain = "11" if rank < 4 else "12"
    acl_param = json.dumps({"rank": rank % 4, "rankSize": 4,
                    "rankRoot": 0, "allReduceType": "sum", "backend": "lccl","commDomain": comm_domain})
    acl_allreduce_operation.set_param(acl_param)
    inTensor = torch.tensor([1,1,1]).float().npu()
    acl_out_tensor = acl_allreduce_operation.execute([inTensor])[0]
    print("acl_out_tensor:", acl_out_tensor,
                    ", \nrank:", rank)
    acl_param2 = json.dumps({"rank": rank % 4, "rankSize": 4,
                    "rankRoot": 0, "allReduceType": "sum", "backend": "lccl","commDomain": comm_domain})
    acl_allreduce_operation.set_param(acl_param2)
    acl_out_tensor = acl_allreduce_operation.execute([acl_out_tensor])[0]
    torch.npu.synchronize()
    golden_out_tensor = torch.tensor([16,16,16]).float().npu()
    # assert result
    assert golden_compare(golden_out_tensor.cpu(), acl_out_tensor.cpu())
    
    # 8rank commDomain
    acl_param = json.dumps({"rank": rank % 8, "rankSize": 8,
                    "rankRoot": 0, "allReduceType": "sum", "backend": "lccl","commDomain": "8"})
    acl_allreduce_operation.set_param(acl_param)
    inTensor1 = torch.tensor([1,1,1]).float().npu()
    acl_out_tensor_8 = acl_allreduce_operation.execute([inTensor1])[0]
    torch.npu.synchronize()
    golden_out_tensor = torch.tensor([8,8,8]).float().npu()
    # assert result
    assert golden_compare(golden_out_tensor.cpu(), acl_out_tensor_8.cpu())

def golden_compare(out_tensor, golden_out_tensor, rtol=0.001, atol=0.001):
    result = torch.allclose(out_tensor, golden_out_tensor, rtol=rtol, atol=atol)
    if not result:
        print("out_tensor.shape", out_tensor.shape,
            "\ngolden_out_tensor.shape:", golden_out_tensor.shape)
    print("out_tensor:", out_tensor,
        ", \ngolden_oute_tensor:", golden_out_tensor)
    return result

def sum_cal(inTensors):
    result = inTensors[0]
    for i in range(1,len(inTensors)):
        result += inTensors[i]
    return result

def max_cal(inTensors):
    result = inTensors[0]
    for i in range(1,len(inTensors)): 
        result = torch.max(result,inTensors[i])
    return result

def min_cal(inTensors):
    result = inTensors[0]
    for i in range(1,len(inTensors)): 
        result = torch.min(result,inTensors[i])
    return result

def log(out_tensor,golden_out_tensor,filename):
    # 把输出重定向到文件
    f = open(filename, 'w')
    # 之后使用print函数，都将内容打印到 screenshot.log 文件中
    sys.stdout = f
    print("diff:",out_tensor-golden_out_tensor)
    f.close()

class AllReduceOperationTest(operation_test.OperationTest):
    def test_all_reduce(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        
        device_count = torch_npu.npu.device_count() 
        if device_count < 8: 
            print("device_count is not enough") 
            return

        command = f"nm -D {LIB_PATH} | grep AllReduce > /dev/null"
        res = os.system(command)
        print("res:",res)
        if res == 0:
            world_size = 8
            device_available = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
            if device_available:
                device_num = len(device_available.split(","))
                if world_size > device_num:
                    self.skipTest(f"Skipped because world_size {world_size} > available devices {device_num}")
            random_seed = 123
            inTensorDtypes = [torch.float]
            sizes = [[2,3]]
            golden_cal = {"sum":"sum_cal"}
            mp.spawn(main_worker, nprocs=world_size, args=(world_size, inTensorDtypes, sizes, random_seed,golden_cal))
        else:
            print("lccl_runner is not compiled, skip AllReduceOperationTest")


if __name__ == '__main__':
    unittest.main()