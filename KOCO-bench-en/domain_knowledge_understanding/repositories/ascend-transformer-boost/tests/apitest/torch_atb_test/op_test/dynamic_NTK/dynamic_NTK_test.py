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
import sys
import os
import re
from torch_atb_test.utils import check_float, run_perf_test
import logging
import unittest

def is910B():
    device_name = torch.npu.get_device_name()
    return (re.search("Ascend910B", device_name, re.I) and len(device_name) > 10)

def gen_golden(position_ids, inv_freqs, seq_lens, out_type = 0):
    off = 0
    num_tokens = position_ids.shape[0]
    dim = inv_freqs.shape[1] * 2
    batch_num = seq_lens.shape[0]
    otype = torch.float16 if out_type == 0 else torch.bfloat16
    sinOut = torch.zeros([num_tokens, dim], dtype=torch.float32)
    cosOut = torch.zeros([num_tokens, dim], dtype=torch.float32)
    for batch_id in range(batch_num):
        pos_len = seq_lens[batch_id]
        freqs = torch.einsum('i,j->ij', position_ids[off:off + pos_len].to(torch.float32), inv_freqs[batch_id])
        emb = torch.cat((freqs, freqs), dim = -1)
        cosOut[off:off + pos_len, :] = emb.cos()
        sinOut[off:off + pos_len, :] = emb.sin()
        off += pos_len
    return sinOut.to(otype), cosOut.to(otype)

def gen_test_data(batch, num_tokens, dim, max_seq_len, out_type=0):
    aux_array = torch.arange(0, dim, 2, dtype=torch.float32) / dim
    batch_base = torch.randint(10000, 50000, [batch], dtype=torch.float32)
    position_ids = torch.randint(0, max_seq_len, [num_tokens], dtype=torch.int32)
    inv_freqs = torch.zeros([batch, int(dim / 2)], dtype=torch.float32)
    for i in range(batch):
        inv_freqs[i, :] = 1.0 / batch_base[i] ** aux_array

    avg_seq_len = int(num_tokens / batch)
    seq_lens = torch.ones([batch], dtype=torch.int32) * avg_seq_len
    seq_lens[0] += num_tokens - avg_seq_len * batch
    logging.info(f"seq_lens:{seq_lens}")
    golden_sin, golden_cos = gen_golden(position_ids, inv_freqs, seq_lens, out_type)
    out_type = int(out_type)
    return position_ids.npu(), inv_freqs.npu(), seq_lens.npu()

def run_test():
    if not is910B():
        print("This test case only supports 910B")
        return True
    print("----------- dynamic_NTK test begin ------------")
    position_ids, inv_freqs, seq_lens = gen_test_data(16, 256, 128, 256000, 0)
    dynamic_NTK_param = torch_atb.DynamicNTKParam(out_data_type = torch_atb.AclDataType.ACL_FLOAT16)
    dynamic_NTK = torch_atb.Operation(dynamic_NTK_param)
    logging.info(dynamic_NTK_param)

    def dynamic_NTK_run():
        dynamic_NTK_outputs = dynamic_NTK.forward([position_ids, inv_freqs, seq_lens])
        return dynamic_NTK_outputs

    def golden():
        off = 0
        num_tokens = position_ids.shape[0]
        dim = inv_freqs.shape[1] * 2
        batch_num = seq_lens.shape[0]
        otype = torch.float16
        sinOut = torch.zeros([num_tokens, dim], dtype=torch.float16)
        cosOut = torch.zeros([num_tokens, dim], dtype=torch.float16)
        for batch_id in range(batch_num):
            pos_len = seq_lens[batch_id]
            freqs = torch.einsum('i,j->ij', position_ids[off:off + pos_len].to(torch.float32), inv_freqs[batch_id])
            emb = torch.cat((freqs, freqs), dim = -1)
            cosOut[off:off + pos_len, :] = emb.cos()
            sinOut[off:off + pos_len, :] = emb.sin()
            off += pos_len
        return [sinOut, cosOut]

    cpu_goldens = golden()
    logging.info("cpu_goldens: ", cpu_goldens)

    npu_outputs = dynamic_NTK_run()
    logging.info("npu_outputs: ", npu_outputs)
    
    assert check_float(npu_outputs, cpu_goldens), "Test failed"
    run_perf_test(dynamic_NTK, [position_ids, inv_freqs, seq_lens])
    print("----------- dynamic_NTK test success ------------")

class TestDynamicNTK(unittest.TestCase):
    def test(self):
        run_test()

if __name__ == "__main__":
    unittest.main()