#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import json
import os
import random
import sys
import unittest
from multiprocessing import Process, set_start_method
import numpy as np
import torch
import torch_npu


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
sendcount = [4, 2]
recvout = [[4, 2], [4, 2]]
recvdis = [[0, 4], [0, 4]]


def main_worker(rank, world_size, inTensorDtypes, random_seed,recvout,recvdis,inTensors,tensorlist,sum,tensorafters,sendcount,dim):
    torch_npu.npu.set_device(rank)
    print(f'Process {rank} started, using device npu:{rank}.')
    acl_allgatherv_operation = torch.classes.OperationTorch.OperationTorch(
        "AllGatherVOperation")
    torch.manual_seed(random_seed)
    for inTensorDtype in inTensorDtypes:
        acl_param = json.dumps({"rank": rank, "rankSize": world_size, "rankRoot": 0, "backend": "hccl"})
        acl_allgatherv_operation.set_param(acl_param)
        run_param = json.dumps({"sendCount": sendcount[rank], "recvCounts": recvout[rank], "rdispls": recvdis[rank]})
        host_list = [[sendcount[rank]], recvout[rank], recvdis[rank]]
        host_tensors = [np.array(tensor) for tensor in host_list]
        host_tensors = [torch.from_numpy(tensor).to(torch.int64) for tensor in host_tensors]
        host_tensors = [tensor.npu() for tensor in host_tensors]
        acl_allgatherv_operation.set_varaintpack_param(run_param)
        for i in range(world_size):
            inTensors[i] = torch.tensor(tensorlist[i], dtype=inTensorDtype)
        # y用来推导outputshape，长度应为所有inputtensor的dim0之和
        y = torch.tensor([random.randint(1,100) for i in range(sum)], dtype=torch.float16)
        # #计算goldtensor
        gold_outtensor = []
        for i in range(len(sendcount)):
            gold_outtensor= gold_outtensor+(tensorafters[i][0:sendcount[i]])
        GoldenTensors = (torch.tensor(np.array(gold_outtensor+[0]*(sum*dim[1]-len(gold_outtensor))).reshape(sum,dim[1]), dtype=inTensorDtype))
        acl_out_tensor = acl_allgatherv_operation.execute([inTensors[rank].npu(), host_tensors[0],host_tensors[1], host_tensors[2], y.npu()])[0]
        print(f"********************acl_out_tensor:{acl_out_tensor}")
        torch.npu.synchronize()
        # assert result
        assert golden_compare(GoldenTensors, acl_out_tensor.cpu())


def golden_compare(golden_out_tensor, out_tensor, rtol=0.0001, atol=0.0001):
    result = torch.allclose(out_tensor, golden_out_tensor, rtol=rtol, atol=atol)
    if not result:
        print("out_tensor.shape", out_tensor.shape,
              "\ngolden_out_tensor.shape:", golden_out_tensor.shape)
        print("out_tensor:", out_tensor,
              ", \ngolden_oute_tensor:", golden_out_tensor)
    return result


class AllGatherVOperationTest(operation_test.OperationTest):
    def test_all_gather(self):
        dims=[[32,64],[128,256],[512,1024]]
        world_sizes = [1,2]
        for dim in dims:
            for world_size in world_sizes:
                sendcount= np.random.randint(1,10 ,size=[world_size]).tolist()
                recvout =[sendcount]*world_size
                recvdis = np.zeros([world_size,world_size]).tolist()
                for i in range(len(recvout)):
                    for j in range(len(recvout[i])):
                        if j == 0:
                            recvdis[i][j] = 0
                        else:
                            recvdis[i][j] = recvout[i][j - 1] + recvdis[i][j - 1]
                tensorlist = []
                for i in range(world_size):
                     tensorlist.append(np.random.randint(1, 20, size=dim).tolist())
                tensorafters = []
                for i in range(world_size):
                    tensorafters.append(np.array(tensorlist[i]).flatten().tolist())
                sum=0
                for tensor in tensorlist:
                    sum=sum+len(tensor)
                inTensors = np.zeros([world_size]).tolist()
                random_seed = 123
                inTensorDtypes =  [torch.int8]
                set_start_method('spawn', force=True)
                process_list = []
                for i in range(world_size):
                    p = Process(target=main_worker, args=(i, world_size, inTensorDtypes, random_seed,recvout,recvdis,inTensors,tensorlist,sum,tensorafters,sendcount,dim))
                    p.start()
                    process_list.append(p)
                for i in process_list:
                    p.join()


if __name__ == '__main__':
    unittest.main()

