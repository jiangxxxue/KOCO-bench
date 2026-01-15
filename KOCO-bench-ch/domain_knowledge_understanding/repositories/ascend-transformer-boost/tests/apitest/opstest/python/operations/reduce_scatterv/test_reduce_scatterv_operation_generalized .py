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
from multiprocessing import Process, set_start_method

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import tests.high_level_test.operation_test as operation_test  # NOQA: E402
import itertools

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


def main_worker(rank, world_size, random_seed, recvout, inTensors, inTensorDtype,sendcount, senddisp,
                reduceType, intensorOperation,low,high):
        # init process group
        torch_npu.npu.set_device(rank)
        print(f'Process {rank} started, using device npu:{rank}.')
        # init reduce_scatterv_operation
        op_name = "ReduceScatterVOperation"
        reduce_scatterv_operation = torch.classes.OperationTorch.OperationTorch(
            "ReduceScatterVOperation")
        torch.manual_seed(random_seed)
        # y用来推导outputshape，recvout[rank]为几长度就为几
        y = ((high - low) * torch.rand(recvout[rank]) + low).type(torch.float16)
        # print("y",y)
        gold_outtensor = []
        # 计算goldTensor,获取到发送的值
        for j in range(len(recvout)):
            gold_outtensor.append((intensorOperation.flatten()[senddisp[j][j]:sendcount[j][j] + senddisp[j][j]]))
        # print("gold_outtensor",gold_outtensor)
        # 将goldtensor转化为（recvout[rank]*dim[1]）的二维数组
        GoldenTensors=(torch.cat((gold_outtensor[rank], torch.zeros((recvout[rank] * len(inTensors[rank][0]) - len(gold_outtensor[rank])), dtype=inTensorDtype)),dim=0)).reshape(recvout[rank], len(inTensors[rank][0]))
        # print("GoldenTensors",GoldenTensors)
        acl_param = json.dumps({"rank": rank, "rankSize": world_size, "sendCounts": sendcount[rank],
                                "sdispls": senddisp[rank], "recvCount": recvout[rank], "rankRoot": 0, "backend": "hccl",
                                "reduceType": reduceType})
        run_param = json.dumps({"sendCounts": sendcount[rank], "sdispls": senddisp[rank], "recvCount": recvout[rank]})
        host_list = [sendcount[rank], senddisp[rank], [recvout[rank]]]
        host_tensors = [np.array(tensor) for tensor in host_list]
        host_tensors = [torch.from_numpy(tensor).to(torch.int64) for tensor in host_tensors]
        host_tensors = [tensor.npu() for tensor in host_tensors]
        reduce_scatterv_operation.set_param(acl_param)
        reduce_scatterv_operation.set_varaintpack_param(run_param)
        acl_out_tensor = reduce_scatterv_operation.execute(
            [inTensors[rank].npu(), host_tensors[0], host_tensors[1], host_tensors[2], y.npu()])[0]
        flat_tensor = acl_out_tensor.flatten()
        flat_tensor[recvout[rank]:] = torch.tensor(0, dtype=inTensorDtype)
        acl_out_tensor = flat_tensor.reshape(acl_out_tensor.shape)
        # print("acl_out_tensor",acl_out_tensor)
        torch.npu.synchronize()
        # assert result
        assert golden_compare(acl_out_tensor.cpu(), GoldenTensors)


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


class reduce_scatterv_operationTest(operation_test.OperationTest):
    def test_reduce_scatterv_operation(self):
        # 指定维度和shape
        shapes = [32, 64, 128, 256, 512, 1024]
        dims = list(itertools.combinations(shapes, 2))
        # 指定卡数
        world_sizes = [2]
        # 指定reduceType
        reduceTypes = ['max', 'min', 'sum']
        # reduceTypes=['sum']
        low = -100
        high = 100
        # 指定intensor数据格式
        inTensorDtypes = [torch.int8, torch.float16]
        # inTensorDtypes=[torch.int8]
        for dim in dims:
            for world_size in world_sizes:
                for reduceType in reduceTypes:
                    for inTensorDtype in inTensorDtypes:
                            print(
                                "-----------------------------------------------" + f"dim:{dim},world_size:{world_size},reduceType:{reduceType},inTensorDtype:{inTensorDtype}")
                            # 生成param recvout、sendcount、senddisp
                            recvout = np.random.randint(1, 8, size=[world_size]).tolist()
                            print("recvout", recvout)
                            sendcount = [recvout] * world_size
                            print("sendcount", sendcount)
                            senddisp = np.zeros([world_size, world_size]).tolist()
                            for i in range(len(sendcount)):
                                for j in range(len(sendcount[i])):
                                    if j == 0:
                                        senddisp[i][j] = 0
                                    else:
                                        senddisp[i][j] = sendcount[i][j - 1] + senddisp[i][j - 1]
                            print("senddisp", senddisp)
                            # 生成inTensors
                            inTensors = []
                            for i in range(world_size):
                                inTensors.append(((high - low) * torch.rand(dim) + low).type(inTensorDtype))
                            # print("inTensors",inTensors)
                            # 根据reducetype进行计算，获取到计算后的outtensor(intensorOperation)
                            if reduceType == 'sum':
                                    intensorOperation =torch.sum(torch.stack(inTensors),dim=0)[0].to(inTensorDtype)
                            elif reduceType == 'max':
                                    intensorOperation =torch.max(torch.stack(inTensors),dim=0)[0]
                            elif reduceType == "min":
                                intensorOperation =torch.min(torch.stack(inTensors),dim=0)[0]
                            # print("intensorOperation",intensorOperation)
                            random_seed = 123
                            set_start_method('spawn', force=True)
                            process_list = []
                            for i in range(world_size):
                                p = Process(target=main_worker, args=(
                                    i, world_size, random_seed, recvout, inTensors, inTensorDtype,sendcount,
                                    senddisp,
                                    reduceType, intensorOperation,low,high))
                                p.start()
                                process_list.append(p)

                            for i in process_list:
                                p.join()


if __name__ == '__main__':
    unittest.main()
