#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import sys
import os
import unittest
import torch
import torch_npu
import numpy as np
import torch.nn as nn
import math
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
 
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402
 
OP_NAME = "DynamicNTKOperation"
PARAM = {"outputType": 1}

class TestDynamicNTKOperation(operation_test.OperationTest):
    def __gen_golden(self, position_ids, inv_freqs, seq_lens, out_type = 0):
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

    def __gen_test_data(self, batch, num_tokens, dim, max_seq_len, out_type=0):
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
        self.golden_sin, self.golden_cos = self.__gen_golden(position_ids, inv_freqs, seq_lens, out_type)
        self.out_type = int(out_type)
        return position_ids.npu(), inv_freqs.npu(), seq_lens.npu()

    def golden_calc(self, in_tensors):
        off = 0
        position_ids = in_tensors[0]
        num_tokens = position_ids.shape[0]
        inv_freqs = in_tensors[1]
        dim = inv_freqs.shape[1] * 2
        seq_lens = in_tensors[2]
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
    def test(self):
        if  operation_test.get_soc_version() == 'Ascend910A' or \
            operation_test.get_soc_version() == 'Ascend310B':
            logging.info("this testcase don't supports Ascend910A\Ascend310B")
            return True
        position_ids, inv_freqs, seq_lens = self.__gen_test_data(16, 256, 128, 256000, 0)
        PARAM["outDataType"] = 1
        self.execute(OP_NAME, PARAM, [position_ids, inv_freqs, seq_lens])
if __name__ == '__main__':
    unittest.main()