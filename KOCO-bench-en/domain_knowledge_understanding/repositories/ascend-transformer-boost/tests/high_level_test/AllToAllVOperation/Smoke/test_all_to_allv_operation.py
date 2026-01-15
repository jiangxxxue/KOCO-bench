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
from multiprocessing import Process, set_start_method
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import operation_test  # NOQA: E402
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

ATB_HOME_PATH = os.environ.get("ATB_HOME_PATH")
if ATB_HOME_PATH is None:
    raise RuntimeError(
        "env ATB_HOME_PATH not exist, source set_env.sh")
LIBTORCH_PATH = os.path.join(ATB_HOME_PATH, "lib/libatb_test_framework.so")
LIB_PATH = os.path.join(ATB_HOME_PATH, "lib/libatb.so")
torch.classes.load_library(LIBTORCH_PATH)


def main_worker(rank, world_size, inTensorDtypes, sizes, random_seed, sendcount, senddisp, recvout, recvdis, tensorlist,
                inTensors, tensorafters):
    # init process group
    torch_npu.npu.set_device(rank)
    print(f'Process {rank} started, using device npu:{rank}.')
    # init all_to_allv_operation
    all_to_allv_operation = torch.classes.OperationTorch.OperationTorch("AllToAllVOperation")
    torch.manual_seed(random_seed)
    low = -100
    high = 100
    print("rank", rank)
    # 计算goldTensor
    gold_outtensor = []
    for j in range(len(recvout[rank])):
        print("tensorafters[j][sendcount[j][j]:senddisp[j][j]]",
              tensorafters[j][senddisp[j][rank]:sendcount[j][rank] + senddisp[j][rank]])
        gold_outtensor.append(tensorafters[j][senddisp[j][rank]:sendcount[j][rank] + senddisp[j][rank]])
    gold_outtensor = [i for arr in gold_outtensor for i in arr]
    for k in range(len(inTensorDtypes)):
        GoldenTensors = (torch.tensor(gold_outtensor, dtype=inTensorDtypes[k]))
        print("GoldenTensors", GoldenTensors)
        #计算inTensors
        for i in range(world_size):
            inTensors[i] = torch.tensor(tensorlist[i], dtype=inTensorDtypes[k])

        acl_param = json.dumps(
            {"rank": rank, "rankSize": world_size, "sendCounts": sendcount[rank], "sdispls": senddisp[rank],
             "recvCounts": recvout[rank],
             "rdispls": recvdis[rank], "rankRoot": 0, "backend": "hccl"})
        all_to_allv_operation.set_param(acl_param)
        acl_out_tensor = all_to_allv_operation.execute([inTensors[rank].npu()])[0]
        torch.npu.synchronize()
        # assert result
        assert golden_compare(GoldenTensors, acl_out_tensor.cpu())


def golden_compare(out_tensor, golden_out_tensor, rtol=0.001, atol=0.001):
    result = torch.allclose(out_tensor, golden_out_tensor, rtol=rtol, atol=atol)
    if not result:
        print("out_tensor.shape", out_tensor.shape,
              "\ngolden_out_tensor.shape:", golden_out_tensor.shape)
        print("out_tensor:", out_tensor,
              ", \ngolden_oute_tensor:", golden_out_tensor)
    return result


def log(out_tensor, golden_out_tensor, filename):
    # 把输出重定向到文件
    f = open(filename, 'w')
    # 之后使用print函数，都将内容打印到 screenshot.log 文件中
    sys.stdout = f
    print("diff:", out_tensor - golden_out_tensor)
    f.close()


class all_to_allv_operationTest(operation_test.OperationTest):
    def test_all_to_allv_operation(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        for world_size in random.sample(range(1, 9), 3):
            print("world_size---------------------------------------------------------------------------------------------------------------------world_size", world_size)
            random_seed = 123
            # 生成param
            sendcount = np.random.randint(1, 8, size=[world_size, world_size]).tolist()
            senddisp = np.zeros([world_size, world_size]).tolist()
            recvout = np.zeros([world_size, world_size]).tolist()
            recvdis = np.zeros([world_size, world_size]).tolist()

            for i in range(len(sendcount)):
                for j in range(len(sendcount[i])):
                    if j == 0:
                        senddisp[i][j] = 0
                    else:
                        senddisp[i][j] = senddisp[i][j - 1] + sendcount[i][j - 1]
                    recvout[i][j] = sendcount[j][i]
            for i in range(len(recvout)):
                for j in range(len(recvout[j])):
                    if j == 0:
                        recvdis[i][j] = 0
                    else:
                        recvdis[i][j] = recvout[i][j - 1] + recvdis[i][j - 1]
            print("sendcount", sendcount)
            print("senddisp", senddisp)
            print("recvout", recvout)
            print("recvdis", recvdis)
            # 生成intensorlist
            tensorlist = []
            for i in range(world_size):
                tensorlist.append(np.random.randint(1, 100, size=[8, 8]).tolist())
            inTensorDtypes = [torch.int8, torch.int16, torch.int32, torch.int64, torch.float32, torch.float16,
                              torch.bfloat16]
            inTensors = np.zeros([world_size, world_size]).tolist()
            # 多维的intensor处理为一维
            tensorafters = []
            for i in range(world_size):
                tensorafters.append(np.array(tensorlist[i]).flatten().tolist())
            print("tensorafters", tensorafters)
            sizes = [[3, 4]]
            set_start_method('spawn', force=True)
            process_list = []
            for i in range(world_size):
                p = Process(target=main_worker, args=(
                i, world_size, inTensorDtypes, sizes, random_seed, sendcount, senddisp, recvout, recvdis, tensorlist,
                inTensors, tensorafters))
                p.start()
                process_list.append(p)

            for i in process_list:
                p.join()


if __name__ == '__main__':
    unittest.main()
