#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import numpy as np
import torch
import torch.nn as nn
import torch_atb
import re
import sys
import os
from torch_atb_test.utils import run_perf_test
import unittest
import logging

def is910B():
    device_name = torch.npu.get_device_name()
    return (re.search("Ascend910B", device_name, re.I) and len(device_name) > 10) or re.search("Ascend910_93", device_name, re.I)


def run_test():
    if not is910B():
        print("This test case only support 910B")
        return
    print("----------- self_attention_prefix_encoder test begin ------------")
    self_attention_param = torch_atb.SelfAttentionParam()
    self_attention_param.head_num = 12
    self_attention_param.kv_head_num = 1
    self_attention_param.calc_type = torch_atb.SelfAttentionParam.CalcType.PREFIX_ENCODER
    self_attention_param.mask_type = torch_atb.SelfAttentionParam.MaskType.MASK_TYPE_ALIBI_COMPRESS_SQRT
    self_attention_param.kernel_type = torch_atb.SelfAttentionParam.KernelType.KERNELTYPE_HIGH_PRECISION
    self_attention_param.is_triu_mask = 1
    self_attention = torch_atb.Operation(self_attention_param)
    q = torch.ones(128, 1536, dtype=torch.float16).npu()
    k = torch.ones(1024, 128, 1, 128, dtype=torch.float16).npu()
    v = torch.ones(1024, 128, 1, 128, dtype=torch.float16).npu()
    block_tables = torch.arange(0, 4, dtype=torch.int32).view(1, 4).npu()
    mask = torch.ones(256, 256, dtype=torch.float16).npu()
    q_seqlens = torch.tensor([128], dtype=torch.int32)
    kv_seqLen = torch.tensor([512], dtype=torch.int32)
    slopes = torch.ones(12, dtype=torch.float).npu()
    intensors = [q, k, v, block_tables, mask, q_seqlens, kv_seqLen, slopes]

    def self_attention_run():
        output = self_attention.forward([q, k, v, block_tables, mask, q_seqlens, kv_seqLen, slopes])
        return output

    npu_outputs = self_attention_run()
    logging.info("npu_outputs: ", npu_outputs)
    run_perf_test(self_attention, intensors)
    print("----------- self_attention_prefix_encoder test success ------------")

class TestSelfAttentionPrefixEncoder(unittest.TestCase):
    def test(self):
        run_test()

if __name__ == "__main__":
    unittest.main()
