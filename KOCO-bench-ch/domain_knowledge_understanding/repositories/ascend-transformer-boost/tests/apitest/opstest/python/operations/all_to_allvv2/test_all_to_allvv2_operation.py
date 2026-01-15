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

import numpy as np
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from multiprocessing import  Process, set_start_method


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
sendcount=[[2,4],[2,1]]
senddisp=[[0,2],[0,2]]
recvout=[[2,2],[4,1]]
recvdis=[[0,2],[0,4]]

def main_worker(rank, world_size,inTensorDtypes, sizes, random_seed):
    # init process group
    torch_npu.npu.set_device(int(rank))
    print(f'========== Process {rank} started, using device npu:{rank}.==========')
    # init all_to_allvv2_operation
    op_name = "AllToAllVV2Operation"
    all_to_allvv2_operation = torch.classes.OperationTorch.OperationTorch(
        "AllToAllVV2Operation")
    torch.manual_seed(random_seed)
    low = -100
    high = 100
    for inTensorDtype in inTensorDtypes:
        inTensors=[torch.tensor([[0,1,2,3],[4,5,6,7]],dtype=inTensorDtype),torch.tensor([[0,1,2],[3,4,6]],dtype=inTensorDtype)]
        GoldenTensors=[torch.tensor([[0,1,0,1]],dtype=inTensorDtype),torch.tensor([2,3,4,5,2],dtype=inTensorDtype)]
        acl_param = json.dumps({"rank": rank, "rankSize": world_size,"rankRoot": 0, "backend": "hccl"})
        run_param = json.dumps({"sendCounts": sendcount[rank], "sdispls": senddisp[rank], "recvCounts": recvout[rank],
                                "rdispls": recvdis[rank]})
        host_list = [sendcount[rank], senddisp[rank], recvout[rank], recvdis[rank]]
        host_tensors = [np.array(tensor) for tensor in host_list]
        host_tensors = [torch.from_numpy(tensor).to(torch.int64) for tensor in host_tensors]
        host_tensors = [tensor.npu() for tensor in host_tensors]
        all_to_allvv2_operation.set_param(acl_param)
        all_to_allvv2_operation.set_varaintpack_param(run_param)
        print(f"host_tensors={host_tensors}")
        print(f"inTensors[rank].npu()={inTensors[rank].npu()}")
        print(f"host_tensors[0]={host_tensors[0]}")
        print(f"host_tensors[1]={host_tensors[1]}")
        print(f"host_tensors[2]={host_tensors[2]}")
        print(f"host_tensors[3]={host_tensors[3]}")
        atb_input5 = compute_out_tensor_dim1(recvout[rank])
        print(f"atb_input5 = {atb_input5}")
        result1 = all_to_allvv2_operation.execute(
            [inTensors[rank].npu(), host_tensors[0], host_tensors[1], host_tensors[2], host_tensors[3], atb_input5])
        print(f"\n ==========\nresult1={result1}\n ==========\n")
        acl_out_tensor = result1[0]
        print(f"\n ==========\nacl_out_tensor={acl_out_tensor}\n ==========\n")
        torch.npu.synchronize()
        # assert result
        assert golden_compare(acl_out_tensor.cpu(), GoldenTensors[rank])


def compute_out_tensor_dim1(recvout_i):
    num = sum(recvout_i)
    tensor1 = [3] * num
    tensor1 = np.array(tensor1)
    tensor1 = torch.from_numpy(tensor1).to(torch.int8)
    tensor1 = tensor1.npu()
    return tensor1


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

class all_to_allvv2_operationTest(operation_test.OperationTest):
    def test_all_to_allvv2_operation(self):
        world_size = 2
        random_seed = 123
        sizes = [[3, 4]]
        set_start_method('spawn', force=True)
        process_list = []
        if operation_test.get_soc_version() == 'Ascend910B':
            inTensorDtypes = [torch.int8, torch.int16, torch.int32, torch.int64, torch.float32, torch.float16,
                              torch.bfloat16]
            for i in range(world_size):
                p = Process(target=main_worker,args=(i,world_size, inTensorDtypes, sizes, random_seed))
                p.start()
                process_list.append(p)

            for i in process_list:
                p.join()
        else:
            inTensorDtypes = [torch.int8, torch.float16]
            for i in range(world_size):
                p = Process(target=main_worker, args=(i, world_size, inTensorDtypes, sizes, random_seed))
                p.start()
                process_list.append(p)

            for i in process_list:
                p.join()


if __name__ == '__main__':
    unittest.main()