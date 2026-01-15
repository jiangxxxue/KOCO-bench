import torch
import torch.nn as nn
import torch_atb
import sys
import os
import unittest
import logging

def run_test():
    print("----------- set_buffer_size_single_op test begin ------------")
    self_attention_param = torch_atb.SelfAttentionParam()
    self_attention_param.head_num = 24
    self_attention_param.kv_head_num = 24
    self_attention_param.calc_type = torch_atb.SelfAttentionParam.CalcType.PA_ENCODER
    self_attention = torch_atb.Operation(self_attention_param)
    logging.info(self_attention_param)
    q = torch.ones(4096, 24, 64, dtype=torch.float16).npu()
    k = torch.ones(4096, 24, 64, dtype=torch.float16).npu()
    v = torch.ones(4096, 24, 64, dtype=torch.float16).npu()
    seqlen = torch.tensor([4096], dtype=torch.int32)
    intensors = [q,k,v,seqlen]

    def self_attention_run():
        outputs = self_attention.forward([q,k,v,seqlen])
        return [outputs]

    npu_outputs = self_attention_run()
    logging.info("npu_outputs: ", npu_outputs)
    print("----------- set_buffer_size_single_op test success ------------")

class TestSetBufferSize(unittest.TestCase):
    def test_single_op(self):
        torch_atb.set_buffer_size(300000000)
        run_test()

if __name__ == "__main__":
    unittest.main()