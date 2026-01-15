#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import torch
import torch_atb
import acl
import unittest
import sys
import os
from torch_atb_test.utils import ret_check, check_float

def run_test():
    print("----------- graph test begin ------------")
    elewise_param = torch_atb.ElewiseParam()
    elewise_param.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_ADD
    elewise_add = torch_atb.Operation(elewise_param)
    elewise_param.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_MUL
    elewise_mul = torch_atb.Operation(elewise_param)
    graph = torch_atb.GraphBuilder("Graph") \
        .set_input_output(["x", "y", "z"], ["out"]) \
        .add_operation(elewise_add, ["x", "y"], ["add_out"]) \
        .reshape("add_out", lambda shape: [1, shape[0] * shape[1]], "add_out_") \
        .add_operation(elewise_mul, ["add_out_", "z"], ["out"]) \
        .build()
    print(graph)
    x = torch.ones(2, 3, dtype=torch.float16)
    y = torch.ones(2, 3, dtype=torch.float16)
    z = torch.ones(1, 6, dtype=torch.float16)
    tensors_npu = [tensor.npu() for tensor in [x, y, z]]

    def graph_run():
        return graph.forward(tensors_npu)

    def golden():
        sum_xy = x + y
        sum_xy_reshaped = sum_xy.view(1, 6)
        result = sum_xy_reshaped * z
        return [result]

    cpu_goldens = golden()

    npu_outputs = graph_run()
    
    assert check_float(npu_outputs, cpu_goldens), "Test failed"
    print("----------- graph test success ------------")

class TestGraph(unittest.TestCase):
    def test(self):
        ret = acl.rt.set_device(0)
        ret_check(ret)
        run_test()

if __name__ == "__main__":
    unittest.main()