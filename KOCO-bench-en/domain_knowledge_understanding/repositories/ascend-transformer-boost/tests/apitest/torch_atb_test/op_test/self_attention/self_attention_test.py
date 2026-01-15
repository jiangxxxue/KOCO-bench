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
import torch.nn as nn
import torch_atb
import sys
import os
from torch_atb_test.utils import check_float, run_perf_test
import unittest
import logging

def run_test():
    print("----------- self_attention test begin ------------")
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
    run_perf_test(self_attention, intensors)
    print("----------- self_attention test success ------------")

class TestSelfAttention(unittest.TestCase):
    def test(self):
        run_test()

if __name__ == "__main__":
    unittest.main()