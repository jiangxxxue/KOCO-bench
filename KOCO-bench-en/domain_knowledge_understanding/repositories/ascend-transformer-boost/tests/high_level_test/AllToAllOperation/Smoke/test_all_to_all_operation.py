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
from multiprocessing import  Process, set_start_method

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import operation_test  # NOQA: E402
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

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

def main_worker(rank, world_size,inTensorDtypes, sizes, random_seed):
    # init process group
    torch_npu.npu.set_device(rank)
    print(f'Process {rank} started, using device npu:{rank}.')
    # init all reduce operation
    all_to_all_operation = torch.classes.OperationTorch.OperationTorch(
        "AllToAllOperation")
    #exec all reduce
    torch.manual_seed(random_seed)
    low = -100
    high = 100
    for inTensorDtype in inTensorDtypes:
        print(inTensorDtype)
        for size in sizes:
            inTensors = []
            for i in range(world_size):
                inTensor = ((high - low) * torch.rand(size) + low).type(inTensorDtype)
                inTensors.append(inTensor)
            goldenTensors = []
            for i in range(world_size):
                golden_out = []
                for j in range(world_size):
                    golden_out_list = inTensors[j].reshape(-1).tolist()
                    split = golden_out_list[i*len(golden_out_list) // world_size:(i+1)*len(golden_out_list) // world_size]
                    golden_out += split
                golden_out_tensor = torch.tensor(golden_out,dtype=inTensorDtype).reshape(size)
                goldenTensors.append(golden_out_tensor)

            acl_param = json.dumps({"rank": rank, "rankSize": world_size,
                            "rankRoot": 0, "backend": "hccl"})
            all_to_all_operation.set_param(acl_param)
            inTensor = inTensors[rank].clone().npu()
            acl_out_tensor = all_to_all_operation.execute([inTensor])[0]
            torch.npu.synchronize()
            # assert result
            assert golden_compare(goldenTensors[rank], acl_out_tensor.cpu())

def golden_compare(out_tensor, golden_out_tensor, rtol=0.001, atol=0.001):
    result = torch.allclose(out_tensor, golden_out_tensor, rtol=rtol, atol=atol)
    if not result:
        print("out_tensor.shape", out_tensor.shape,
            "\ngolden_out_tensor.shape:", golden_out_tensor.shape)
        print("out_tensor:", out_tensor,
            ", \ngolden_oute_tensor:", golden_out_tensor)
    return result

def log(out_tensor,golden_out_tensor,filename):
    # 把输出重定向到文件
    f = open(filename, 'w')
    # 之后使用print函数，都将内容打印到 screenshot.log 文件中
    sys.stdout = f
    print("diff:",out_tensor-golden_out_tensor)
    f.close()

class all_to_all_operationTest(operation_test.OperationTest):
    def test_all_to_all_operation(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        world_size = 4
        random_seed = 123
        inTensorDtypes = [torch.int8, torch.int16, torch.int32, torch.int64,torch.float32,torch.float16, torch.bfloat16]
        sizes = [[3,4,6]]
        set_start_method('spawn', force=True)
        process_list = []
        for i in range(world_size):  
            p = Process(target=main_worker,args=(i,world_size, inTensorDtypes, sizes, random_seed))
            p.start()
            process_list.append(p)

        for i in process_list:
            p.join()    


if __name__ == '__main__':
    unittest.main()