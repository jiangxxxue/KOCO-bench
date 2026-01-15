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
import time
import unittest
import sys
from time import sleep
import torch
import torch_npu
import torch.multiprocessing as mp
from multiprocessing import  Process, set_start_method


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
high = 100
low = -100

def main_worker(rank, world_size, inTensorDtypes,sizes,random_seed):
    torch.manual_seed(random_seed)
    torch_npu.npu.set_device(rank)
    print(f'Process {rank} started, using device npu:{rank}.')
    # init send operation
    acl_send_operation = torch.classes.OperationTorch.OperationTorch(
        "SendOperation")
    # set send param
    acl_param = json.dumps({"rank": rank, "rankSize": world_size,
                                    "rankRoot": 0,"destRank": rank+2})
    acl_send_operation.set_param(acl_param)
    for inTensorDtype in inTensorDtypes:
        # inTensor1 = torch.tensor([[3,0,26],
        #                         [2,15,2]]).float().npu()
        # inTensor2 = torch.tensor([[4,5,6],
        #                         [2,1,2],
        #                         [3,4,4]]).float().npu()
        inTensor1 = ((high - low) * torch.rand(2,3,4) + low).type(inTensorDtype).npu()
        inTensor2 = ((high - low) * torch.rand(3,3,4) + low).type(inTensorDtype).npu()
        intensorList = [inTensor1,inTensor2]
        out1 = torch.zeros(inTensor1.shape,dtype=inTensorDtype).npu()
        out2 = torch.zeros(inTensor2.shape,dtype=inTensorDtype).npu()
        outtensorList = [out1,out2]
        print("setup SendOperation...")
        acl_send_operation.setup([inTensor1],[])
        # init recv operation
        acl_recv_operation = torch.classes.OperationTorch.OperationTorch(
            "RecvOperation")
        # set recv param
        acl_param2 = json.dumps({"rank": rank, "rankSize": world_size,
                                    "rankRoot": 0, "srcRank": rank-2})
        acl_recv_operation.set_param(acl_param2)

        # rank 0 1  0 send 2 ,1 send 3
        if rank//2==0:
            print("execute SendOperation...")
            acl_send_operation.execute([intensorList[rank]])
            print(f'send result={"null"}')
        else:# rank 2 3
            print("execute RecvOperation...")
            acl_out_tensor = acl_recv_operation.execute([outtensorList[rank-2]])[0]
            print(f'recv result={acl_out_tensor}')
            torch.npu.synchronize()
            # assert result
            assert golden_compare(acl_out_tensor, intensorList[rank-2])

def golden_compare(out_tensor, golden_out_tensor, rtol=0.001, atol=0.001):
    result = torch.allclose(out_tensor, golden_out_tensor, rtol=rtol, atol=atol)
    if not result:
        print("out_tensor.shape", out_tensor.shape,
            "\ngolden_out_tensor.shape:", golden_out_tensor.shape)
        print("out_tensor:", out_tensor,
            ", \ngolden_oute_tensor:", golden_out_tensor)
    return result

class SendOperationTest(operation_test.OperationTest):
    def test_send(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        world_size = 4
        device_available = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        if device_available:
            device_num = len(device_available.split(","))
            if world_size > device_num:
                self.skipTest(f"Skipped because world_size {world_size} > available devices {device_num}")
        random_seed = 123
        inTensorDtypes = [torch.int8, torch.int16, torch.int32, torch.int64,torch.float32,torch.float16, torch.bfloat16]
        sizes = [[10,100,512]]
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