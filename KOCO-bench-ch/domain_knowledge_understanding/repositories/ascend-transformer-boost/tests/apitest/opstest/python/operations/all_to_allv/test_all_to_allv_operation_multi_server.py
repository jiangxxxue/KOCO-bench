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
sendcount=[[2,4],[2,1]]
senddisp=[[0,2],[0,2]]
recvout=[[2,2],[4,1]]
recvdis=[[0,2],[0,4]]

def main_worker(local_rank, gpus, nodes,world_size,nr,inTensorDtypes, sizes, random_seed):
    torch.manual_seed(random_seed)
    rank = nr * gpus + local_rank
    torch_npu.npu.set_device(local_rank)
    print(f'Process {rank} started, using device npu:{rank}.')
    all_to_allv_operation = torch.classes.OperationTorch.OperationTorch(
        "AllToAllVOperation")
    torch.manual_seed(random_seed)
    low = -100
    high = 100
    for inTensorDtype in inTensorDtypes:
        inTensors=[torch.tensor([[0,1,2,3],[4,5,6,7]],dtype=inTensorDtype),torch.tensor([[0,1,2],[3,4,6]],dtype=inTensorDtype)]
        GoldenTensors=[torch.tensor([[0,1,0,1]],dtype=inTensorDtype),torch.tensor([2,3,4,5,2],dtype=inTensorDtype)]
        acl_param = json.dumps({"rank": rank, "rankSize": world_size,"sendCounts":sendcount[rank],"sdispls":senddisp[rank],"recvCounts":recvout[rank],
        "rdispls":recvdis[rank],"rankRoot": 0, "backend": "hccl", "rankTableFile":"/home/qsc/ascend-transformer-boost/ranktable_16_89_91.json"})
        all_to_allv_operation.set_param(acl_param)
        acl_out_tensor = all_to_allv_operation.execute([inTensors[rank].npu()])[0]
        torch.npu.synchronize()
        # assert result
        assert golden_compare(GoldenTensors[rank], acl_out_tensor.cpu())

def golden_compare(out_tensor, golden_out_tensor, rtol=0.001, atol=0.001):
    result = torch.allclose(out_tensor, golden_out_tensor, rtol=rtol, atol=atol)
    if not result:
        print("out_tensor.shape", out_tensor.shape,
            "\ngolden_out_tensor.shape:", golden_out_tensor.shape)
        print("out_tensor:", out_tensor,
            ", \ngolden_oute_tensor:", golden_out_tensor)
    return result

class all_to_allv_operationTest(operation_test.OperationTest):
    def test_all_to_allv(self):
        return
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        gpus = 2
        nodes =2
        world_size =gpus * nodes
        device_available = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        if device_available:
            device_num = len(device_available.split(","))
            if world_size > device_num:
                self.skipTest(f"Skipped because world_size {world_size} > available devices {device_num}")
        nr = 1 #节点编号
        random_seed = 123
        inTensorDtypes = [torch.int8, torch.int16, torch.int32, torch.int64,torch.float32,torch.float16, torch.bfloat16]
        sizes = [[10,100,512]]
        set_start_method('spawn', force=True)
        process_list = []
        for i in range(gpus):
            p = Process(target=main_worker,args=(i,gpus, nodes,world_size,nr, inTensorDtypes, sizes, random_seed))
            p.start()
            process_list.append(p)

        for i in process_list:
            p.join()


if __name__ == '__main__':
    unittest.main()