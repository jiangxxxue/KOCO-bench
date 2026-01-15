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
import re
import sys
import os
from torch_atb_test.utils import run_perf_test
import unittest

def is910B():
    device_name = torch.npu.get_device_name()
    return (re.search("Ascend910B", device_name, re.I) and len(device_name) > 10) or re.search("Ascend910_93", device_name, re.I)


def run_test():
    if not is910B():
        print("This test case only support 910B")
        return
    print("----------- razor fusion attention test begin ------------")
    rfa_param = torch_atb.RazorFusionAttentionParam()
    rfa_param.head_num = 1
    rfa_param.kv_head_num = 1
    rfa_param.qk_scale = 1
    rfa_param.razor_len = 3
    rfa_param.pre_tokens = 256
    rfa_param.next_tokens = 128
    rfa_param.tile_q = 4
    rfa_param.tile_kv = 4
    rfa_param.text_q_len = 3611
    rfa_param.text_kv_len = 3688
    self_attention = torch_atb.Operation(rfa_param)
    q = torch.ones(7246, 128, dtype=torch.bfloat16).npu()
    k = torch.ones(7400, 128, dtype=torch.bfloat16).npu()
    v = torch.ones(7400, 128, dtype=torch.bfloat16).npu()
    intensors = [q, k, v]
    run_perf_test(self_attention, intensors, 10)
    print("----------- razor fusion attention test success ------------")

class TestRazorFusionAttention(unittest.TestCase):
    def test(self):
        run_test()

if __name__ == "__main__":
    unittest.main()