import torch_atb
import torch
import acl
import unittest
import numpy as np
import sys
import os
import numbers
import logging

def gen_inputs():
    ele1 = torch.ones(2, 3, dtype=torch.float16).npu()
    ele2 = torch.ones(2, 3, dtype=torch.float16).npu()

    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = torch.randn(batch, sentence_length, embedding_dim)
    normalized_shape = (embedding_dim,) if isinstance(embedding_dim, numbers.Integral) else tuple(embedding_dim)
    lay1 = embedding.npu()
    lay2 =  torch.ones(normalized_shape, dtype=torch.float32).npu()
    lay3 = torch.zeros(normalized_shape, dtype=torch.float32).npu()

    m, k, n = 3, 4, 5
    lin1 = torch.randn(m, k, dtype=torch.float16).npu()
    lin2 = torch.randn(k, n, dtype=torch.float16).npu()

    sof1 = torch.randn(2, 16, 256, dtype=torch.float32).npu()

    ntoken = 4
    seqlen = 4
    hidden_size = 4096
    head_size = 128
    rop1 = torch.rand(ntoken, hidden_size).half().npu()
    rop2 = torch.rand(ntoken, hidden_size).half().npu()
    rop3 = torch.rand(ntoken, head_size).half().npu()
    rop4 = torch.rand(ntoken, head_size).half().npu()
    rop5 = torch.tensor([seqlen], dtype=torch.int32).npu()

    sel1 = torch.ones(4096, 24, 64, dtype=torch.float16).npu()
    sel2 = torch.ones(4096, 24, 64, dtype=torch.float16).npu()
    sel3 = torch.ones(4096, 24, 64, dtype=torch.float16).npu()
    sel4 = torch.tensor([4096], dtype=torch.int32)

    spl1 = torch.randn(6, 6, dtype=torch.float16).npu()

    gat1 = torch.randn([3,5],dtype=torch.float16).npu()
    gat2 = torch.randint(0, 5, [3,4],dtype=torch.int64).npu()

    act1 = torch.rand(2, 3, 5).bfloat16().npu()

    epsilon = 1e-5
    shape=[8, 8, 8]
    shape_gamma=[8]
    shape_rstd=[8, 8, 1]
    rms1 = torch.from_numpy(np.random.uniform(low=0, high=100, size=shape).astype(np.float32)).npu()
    rms2 = torch.from_numpy(np.random.uniform(low=0, high=100, size=shape_gamma).astype(np.float32)).npu()

    inputs = [ele1, ele2, lay1, lay2, lay3, lin1, lin2, sof1, rop1, rop2, rop3, rop4, rop5, sel1, sel2, sel3, sel4, spl1, gat1, gat2, act1, rms1, rms2]
    return inputs

def graph_build():
    print("----------- graph test begin ------------")
    builder = torch_atb.Builder("Graph")

    # elewise_add
    ele1 = builder.add_input("ele1")
    ele2 = builder.add_input("ele2")
    elewise_add_param = torch_atb.ElewiseParam()
    elewise_add_param.elewise_type = torch_atb.ElewiseParam.ElewiseType.ELEWISE_ADD
    node1 = builder.add_node([ele1, ele2], elewise_add_param)
    add_out = node1.get_output(0)
    
    # layernorm
    eps=1e-05
    embedding_dim = 10
    normalized_shape = (embedding_dim,) if isinstance(embedding_dim, numbers.Integral) else tuple(embedding_dim)
    lay1 = builder.add_input("lay1")
    lay2 = builder.add_input("lay2")
    lay3 = builder.add_input("lay3")
    layer_norm_param = torch_atb.LayerNormParam()
    layer_norm_param.layer_type = torch_atb.LayerNormParam.LayerNormType.LAYER_NORM_NORM
    layer_norm_param.norm_param.epsilon = eps
    layer_norm_param.norm_param.begin_norm_axis = len(normalized_shape) * -1
    node2 = builder.add_node([lay1, lay2, lay3], layer_norm_param)
    layernorm_out = node2.get_output(0)

    # linear
    lin1 = builder.add_input("lin1")
    lin2 = builder.add_input("lin2")
    linear_param = torch_atb.LinearParam()
    linear_param.has_bias = False
    linear_param.transpose_b = False
    node3 = builder.add_node([lin1, lin2], linear_param)
    linear_out = node3.get_output(0)

    # softmax
    axes = [0]
    sof1 = builder.add_input("sof1")
    softmax_param = torch_atb.SoftmaxParam()
    softmax_param.axes = axes
    node4 = builder.add_node([sof1], softmax_param)
    softmax_out = node4.get_output(0)

    # rope
    rop1 = builder.add_input("rop1")
    rop2 = builder.add_input("rop2")
    rop3 = builder.add_input("rop3")
    rop4 = builder.add_input("rop4")
    rop5 = builder.add_input("rop5")
    rope_param = torch_atb.RopeParam()
    rope_param.rotary_coeff = 4
    node5 = builder.add_node([rop1, rop2, rop3, rop4, rop5], rope_param)
    rope_out = node5.get_output(0)

    # selfattention
    sel1 = builder.add_input("sel1")
    sel2 = builder.add_input("sel2")
    sel3 = builder.add_input("sel3")
    sel4 = builder.add_input("sel4")
    self_attention_param = torch_atb.SelfAttentionParam()
    self_attention_param.head_num = 24
    self_attention_param.kv_head_num = 24
    self_attention_param.calc_type = torch_atb.SelfAttentionParam.CalcType.PA_ENCODER
    node6 = builder.add_node([sel1, sel2, sel3, sel4], self_attention_param)
    selfattention_out = node6.get_output(0)

    # split
    splitDim = 0
    splitNum = 2
    spl1 = builder.add_input("spl1")
    split_param = torch_atb.SplitParam()
    split_param.split_dim = splitDim
    split_param.split_num = splitNum
    node7 = builder.add_node([spl1], split_param)
    split_out = node7.get_output(0)

    # gather
    axis = 1
    gat1 = builder.add_input("gat1")
    gat2 = builder.add_input("gat2")
    gather_param = torch_atb.GatherParam()
    gather_param.axis = axis
    node8 = builder.add_node([gat1, gat2], gather_param)
    gather_out = node8.get_output(0)

    # activation
    act1 = builder.add_input("act1")
    activation_param = torch_atb.ActivationParam()
    activation_param.activation_type = torch_atb.ActivationType.ACTIVATION_SWISH
    activation_param.scale = 1.0
    node9 = builder.add_node([act1], activation_param)
    activation_out = node9.get_output(0)

    # rmsnorm
    rms1 = builder.add_input("rms1")
    rms2 = builder.add_input("rms2")
    rms_norm_param = torch_atb.RmsNormParam()
    rms_norm_param.layer_type = torch_atb.RmsNormParam.RmsNormType.RMS_NORM_NORM
    rms_norm_param.norm_param.rstd = True
    node10 = builder.add_node([rms1, rms2], rms_norm_param)
    rmsnorm_out = node10.get_output(0)

    builder.mark_output(gather_out)
    Graph = builder.build()

    inputs = gen_inputs()
    npu_outputs = Graph.forward(inputs)
    logging.info(npu_outputs)

    print("----------- graph test success ------------")

class TestGraph(unittest.TestCase):
    def test(self):
        ret = acl.rt.set_device(0)
        if ret != 0:
            self.assertEqual(ret, 0, "ret should be 0")
        graph_build()
        torch.npu.synchronize()

if __name__ == "__main__":
    unittest.main()