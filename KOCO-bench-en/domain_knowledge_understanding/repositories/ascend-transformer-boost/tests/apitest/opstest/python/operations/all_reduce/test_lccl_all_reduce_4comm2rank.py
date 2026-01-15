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
    inTensor = torch.tensor([1,1,1]).float().npu()
    # 1 commDomain
    if rank < 2:
        comm_domain = "1" 
        acl_param = json.dumps({"rank": rank % 2, "rankSize": 2,
                    "rankRoot": 0, "allReduceType": "sum", "backend": "lccl","commDomain": comm_domain})
        acl_allreduce_operation.set_param(acl_param)
        acl_out_tensor = acl_allreduce_operation.execute([inTensor])[0]
        torch.npu.synchronize()
        golden_out_tensor = torch.tensor([2,2,2]).float().npu()
        # assert result
        assert golden_compare(golden_out_tensor.cpu(), acl_out_tensor.cpu())

    # 2 commDomain
    if 2<= rank < 4:
        comm_domain = "2" 
        acl_param = json.dumps({"rank": rank % 2, "rankSize": 2,
                    "rankRoot": 0, "allReduceType": "sum", "backend": "lccl","commDomain": comm_domain})
        acl_allreduce_operation.set_param(acl_param)
        acl_out_tensor = acl_allreduce_operation.execute([inTensor])[0]
        torch.npu.synchronize()
        golden_out_tensor = torch.tensor([2,2,2]).float().npu()
        # assert result
        assert golden_compare(golden_out_tensor.cpu(), acl_out_tensor.cpu())

    # 3 commDomain
    if 4<= rank < 6:
        comm_domain = "3" 
        acl_param = json.dumps({"rank": rank % 2, "rankSize": 2,
                    "rankRoot": 0, "allReduceType": "sum", "backend": "lccl","commDomain": comm_domain})
        acl_allreduce_operation.set_param(acl_param)
        acl_out_tensor = acl_allreduce_operation.execute([inTensor])[0]
        torch.npu.synchronize()
        golden_out_tensor = torch.tensor([2,2,2]).float().npu()
        # assert result
        assert golden_compare(golden_out_tensor.cpu(), acl_out_tensor.cpu())

    # 4 commDomain
    if 6<= rank < 8:
        comm_domain = "4" 
        acl_param = json.dumps({"rank": rank % 2, "rankSize": 2,
                    "rankRoot": 0, "allReduceType": "sum", "backend": "lccl","commDomain": comm_domain})
        acl_allreduce_operation.set_param(acl_param)
        acl_out_tensor = acl_allreduce_operation.execute([inTensor])[0]
        print("acl_out_tensor:", acl_out_tensor,
                    ", \nrank:", rank)
        torch.npu.synchronize()
        golden_out_tensor = torch.tensor([2,2,2]).float().npu()
        # assert result
        assert golden_compare(golden_out_tensor.cpu(), acl_out_tensor.cpu())

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
            random_seed = 123
            inTensorDtypes = [torch.float]
            sizes = [[2,3]]
            golden_cal = {"sum":"sum_cal"}
            mp.spawn(main_worker, nprocs=world_size, args=(world_size, inTensorDtypes, sizes, random_seed,golden_cal))
        else:
            print("lccl_runner is not compiled, skip AllReduceOperationTest")


if __name__ == '__main__':
    unittest.main()