import torch
import torch_npu
import torch_atb
import torch.multiprocessing as mp
import sys
import os
import re
import logging
import unittest
import logging

rank_size = 2
rank_root = 0
backend = "lccl"
random_seed = 123

def all_to_all_run(rank, tensor):
    all_to_all_param = torch_atb.AllToAllParam(
    rank = rank, rank_size = rank_size, rank_root = rank_root, backend = backend, transpose = True)
    all_to_all = torch_atb.Operation(all_to_all_param)
    result = all_to_all.forward([tensor])
    logging.info(all_to_all_param)
    return result

def is910B():
    device_name = torch.npu.get_device_name()
    return (re.search("Ascend910B", device_name, re.I) and len(device_name) > 10)

def run_test(rank, size):
    torch_npu.npu.set_device(rank)
    logging.info(f'Process {rank} started, using device npu:{rank}.')
    torch.manual_seed(random_seed)

    inTensors = []
    for _ in range(rank_size):
        inTensor = torch.rand(3, 4, dtype=torch.float16)
        inTensors.append(inTensor)

    npu_outputs = all_to_all_run(rank, inTensors[rank].npu())
    torch.npu.synchronize()
    logging.info("npu_outputs: ", npu_outputs)

class TestAllToAll(unittest.TestCase):
    def test(self):
        if not is910B():
            print("This test case only supports 910B")
            return True
        print("----------- all_to_all test begin ------------")
        mp.spawn(run_test, nprocs=rank_size, args=(rank_size,))
        print("----------- all_to_all test success ------------")

if __name__ == "__main__":
    unittest.main()
