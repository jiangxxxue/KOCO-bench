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
import time
import json
import unittest
import torch
import torch_npu
import numpy as np



sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

def generate_data(
        layer = 28,
        layer_id = 0,
        batch = 16,
        max_seqlen = 384,
        hidden_size = 1024,
):
    seqlen = np.random.randint(1, max_seqlen // 2, size=batch, dtype=np.int32)
    token_offset = seqlen
    ntokens = np.sum(seqlen)
    newkv = np.random.uniform(-5, 5, size=(ntokens, hidden_size)).astype(np.int8)
    cache_in = np.zeros(shape=(layer, batch, max_seqlen, hidden_size)).astype(np.int8)
    layer_id = np.array([layer_id], dtype=np.int32)
    cache_out = np.zeros(shape=(layer, batch, max_seqlen, hidden_size)).astype(np.int8)
    prefix_ntokens = 0
    for i in range(batch):
            for j in range(seqlen[i]):
                cache_out[layer_id[0]][i][token_offset[i] - seqlen[i] + j][:] = newkv[prefix_ntokens + j][:]
            prefix_ntokens += seqlen[i]
    ret_data = newkv, layer_id, cache_in, token_offset, seqlen, cache_out
    return ret_data

OP_NAME = "KvCacheOperation"
OP_PARAM = {}

data = generate_data()
in_tensors = [torch.from_numpy(tensor) for tensor in data]
in_tensors = [tensor.npu() for tensor in in_tensors]
a = [print(tensor.dtype, tensor.device) for tensor in in_tensors]

class TestKvCacheOperation(operation_test.OperationTest):
    def golden_calc(self, input_tensors):
        return [in_tensors[5]]

    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.allclose(out_tensor, golden_out_tensor, rtol=1, atol=1)

    def test(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.execute_inplace(OP_NAME, OP_PARAM, [in_tensors[0], in_tensors[1], in_tensors[2],
                                              in_tensors[3], in_tensors[4]], [2])
        

if __name__ == '__main__':
    unittest.main()