#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import torch_atb
import torch_npu
import torch
import acl
import unittest
import sys
import os
from torch_atb_test.utils import ret_check

def create_streams():
    streams = []
    for i in range(2):
        stream, ret = acl.rt.create_stream()
        ret_check(ret)
        streams.append(stream)
    return streams

def graph_build_add_mul():
    print("----------- graph build begin ------------")
    builder = torch_atb.Builder("Graph")
    builder.set_execute_streams(create_streams())
    x = builder.add_input("x")
    y = builder.add_input("y")

    elewise_add = torch_atb.ElewiseParam()
    elewise_add.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_ADD

    add1 = builder.add_node([x, y], elewise_add)
    add1_out = add1.get_output(0)

    add2 = builder.add_node([x, y], elewise_add)
    add2.set_stream_id(1)

    add2_out = add2.get_output(0)
    z = builder.add_input("z")

    builder.reshape(add2_out, lambda shape: [1, shape[0] * shape[1]], "add_out_")

    elewise_mul = torch_atb.ElewiseParam()
    elewise_mul.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_MUL

    mul = builder.add_node(["add_out_", z], elewise_mul)
    mul.set_stream_id(1)

    builder.mark_output(mul.get_output(0))
    Graph = builder.build()
    print("----------- graph build success ------------")

    print(Graph)

    print("----------- graph forward begin ------------")
    x = torch.ones(2, 3, dtype=torch.float16)
    y = torch.ones(2, 3, dtype=torch.float16)
    z = torch.ones(1, 6, dtype=torch.float16)
    
    tensors_npu = [tensor.npu() for tensor in [x, y, z]]

    def graph_run():
        res = Graph.forward(tensors_npu)
        assert add1.get_stream_id() == 0
        assert add2.get_stream_id() == 1
        assert mul.get_stream_id() == 1
        return res

    npu_outputs = graph_run()
    print("----------- graph forward success ------------")

class TestMultiStream(unittest.TestCase):
    def test(self):
        torch_npu.npu.set_device(0)
        graph_build_add_mul()

if __name__ == "__main__":
    unittest.main()