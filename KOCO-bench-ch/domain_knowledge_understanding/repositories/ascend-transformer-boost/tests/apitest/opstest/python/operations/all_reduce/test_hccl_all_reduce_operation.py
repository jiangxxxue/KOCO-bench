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

# usage:
# export HCCL_WHITELIST_DISABLE=1
# python3 -m unittest test_all_reduce_operation.py
# Attention: when you use lccl backend, unset HCCL_MTE_ENABLE and copy lcal.o to current directory

ATB_HOME_PATH = os.environ.get("ATB_HOME_PATH")
if ATB_HOME_PATH is None:
    raise RuntimeError(
        "env ATB_HOME_PATH not exist, source set_env.sh")
LIBTORCH_PATH = os.path.join(ATB_HOME_PATH, "lib/libatb_test_framework.so")
LIB_PATH = os.path.join(ATB_HOME_PATH, "lib/libatb.so")
torch.classes.load_library(LIBTORCH_PATH)
os.environ["LCCL_DETERMINISTIC"]="1"
os.environ["HCCL_DETERMINISTIC"]="true"

def main_worker(rank, world_size,inTensorDtypes, sizes, random_seed,golden_cal):
    # init process group
    torch_npu.npu.set_device(rank)
    print(f'Process {rank} started, using device npu:{rank}.')
    # init all reduce operation
    acl_allreduce_operation = torch.classes.OperationTorch.OperationTorch(
        "AllReduceOperation")
    #exec all reduce
    torch.manual_seed(random_seed)
    low = -100
    high = 100
    for inTensorDtype in inTensorDtypes:
        for size in sizes:
            inTensors = []
            for i in range(world_size):
                inTensor = ((high - low) * torch.rand(size) + low).type(inTensorDtype)
                inTensors.append(inTensor)
            for key,gold in golden_cal.items():
                if key == "prod" and inTensorDtype == torch.int16:
                    continue
                if key == "prod" and inTensorDtype == torch.bfloat16:
                    continue
                acl_param = json.dumps({"rank": rank, "rankSize": world_size,
                                "rankRoot": 0, "allReduceType": key, "backend": "hccl"})
                acl_allreduce_operation.set_param(acl_param)
                inTensor = inTensors[rank].clone().npu()
                acl_out_tensor = acl_allreduce_operation.execute([inTensor])[0]
                torch.npu.synchronize()
                golden_out_tensor = globals()[gold](inTensors)
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

def prod_cal(inTensors):
    result = inTensors[0]
    for i in range(1,len(inTensors)): 
        result = torch.mul(result,inTensors[i])
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

        command = f"nm -D {LIB_PATH} | grep HcclAllReduce > /dev/null"
        res = os.system(command)
        if res == 0:
            world_size = 2
            random_seed = 123
            inTensorDtypes = [torch.int8, torch.int16, torch.int32, torch.int64,torch.float32,torch.float16, torch.bfloat16]
            sizes = [[10,100,512]]
            golden_cal = {"sum":"sum_cal","max":"max_cal","min":"min_cal","prod":"prod_cal"}
            mp.spawn(main_worker, nprocs=world_size, args=(world_size, inTensorDtypes, sizes, random_seed,golden_cal))
        else:
            print("hccl_runner is not compiled, skip AllReduceOperationTest")


if __name__ == '__main__':
    unittest.main()