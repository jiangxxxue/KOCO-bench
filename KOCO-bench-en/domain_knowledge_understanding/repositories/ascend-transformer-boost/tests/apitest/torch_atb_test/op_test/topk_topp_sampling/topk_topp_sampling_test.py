import torch
import torch_atb
import torch_npu
import unittest
import re
from typing import List

class topktopp:
    def __init__(self):
        self.param = torch_atb.TopkToppSamplingParam()
        self.param.topk_topp_sampling_type = torch_atb.TopkToppSamplingParam.TopkToppSamplingType.BATCH_TOPK_MULTINOMIAL_LOGPROBS_SAMPLING
        self.param.log_probs_size = 15
        self.op = torch_atb.Operation(self.param)
    
    def forward(self, tensor: List[torch.Tensor]) -> List[torch.Tensor]:
        return self.op.forward(tensor)

def gen_inputs():
    t1 = (torch.rand(4, 50, dtype=torch.float16) * 200.0 - 100.0).npu()
    t2 = torch.randint(low=10, high=21, size=(4, 1), dtype=torch.int32).npu()
    t3 = torch.randn(4, 1, dtype=torch.float16).npu()
    t4 = torch.randn(4, 1, dtype=torch.float).npu()

    return [t1, t2, t3, t4]

def isValid():
    device_name = torch.npu.get_device_name()
    return (re.search("Ascend910B|Ascend910C|Ascend310P", device_name, re.I) and len(device_name) > 10)

def run_test():
    if not isValid():
        print("This test case only supports 910B|910C|310P")
        return True
    print("----------- topk_topp_sampling test begin ------------")
    topktoppsampling = topktopp()
    topktoppsampling.forward(gen_inputs())
    print("----------- topk_topp_sampling test success ------------")
    
class TestTopkToppSampling(unittest.TestCase):
    def test(self):
        run_test()

if __name__ == "__main__":
    unittest.main()