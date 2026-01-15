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
import torch
  
def graph_build_add_mul():
    print("----------- graph test begin ------------")
    bigger_graph = torch_atb.Builder("Graph")
    subgraphs = []
    y = []
    z = []
    out = []
    elewise_add = []
    elewise_mul = []
    builder = torch_atb.Builder("Graph")
    for i in range(10):
        elewise_add.append(torch_atb.ElewiseParam())
        elewise_add[i].elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_ADD
        elewise_mul.append(torch_atb.ElewiseParam())
        elewise_mul[i].elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_MUL
    x = builder.add_input("x")
    y.append(builder.add_input("y0"))
    z.append(builder.add_input("z0"))
    layer1 = [builder.add_node([x, y[0]], elewise_add[0])]
    builder.reshape(layer1[0].get_output(0), lambda shape: [1, shape[0] * shape[1]], "add_out_0")
    layer2 = [builder.add_node(["add_out_0", z[0]], elewise_mul[0])]
    builder.reshape(layer2[0].get_output(0), lambda shape: [2, 3], "out_0")
    out = [layer2[0].get_output(0)]
    for i in range(1,10):
        y.append(builder.add_input("y" + str(i)))
        z.append(builder.add_input("z" + str(i)))
        layer1.append(builder.add_node(["out_"+str(i-1), y[i]], elewise_add[i]))
        builder.reshape(layer1[i].get_output(0), lambda shape: [1, shape[0] * shape[1]], "add_out_"+str(i))
        layer2.append(builder.add_node(["add_out_" + str(i), z[i]], elewise_mul[i]))
        builder.reshape(layer2[i].get_output(0), lambda shape: [2, 3], "out_"+str(i))
        out.append(layer2[i].get_output(0))
    builder.mark_output(layer2[9].get_output(0))
    Graph = builder.build()
    
    x_golden = torch.ones(2, 3, dtype=torch.float16)
    tensors_npu = [x_golden.npu()]
    for i in range(10):
        y_golden = torch.ones(2, 3, dtype=torch.float16)
        tensors_npu.append(y_golden.npu())
        z_golden = torch.ones(1, 6, dtype=torch.float16)
        tensors_npu.append(z_golden.npu())
        
    def graph_run():
        
        return Graph.forward(tensors_npu)

    def golden():
        sum_xy = x_golden + y_golden
        sum_xy_reshaped = sum_xy.view(1, 6)
        result = sum_xy_reshaped * z_golden
        result = result.view(2,3)
        for i in range(9):
            sum_xy = result + y_golden
            sum_xy_reshaped = sum_xy.view(1, 6)
            result = sum_xy_reshaped * z_golden
            result = result.view(2,3)
        return [result.view(1,6)]
    
    cpu_goldens = golden()
    print("cpu_goldens: ", cpu_goldens)

    npu_outputs = graph_run()
    print("npu_outputs: ", npu_outputs)
    
    assert torch.allclose(npu_outputs[0].cpu(), cpu_goldens[0]), "Test failed"
    print("----------- graph test success ------------")


if __name__ == "__main__":
    graph_build_add_mul()
