#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import logging
import sys
import unittest
import math
import numpy as np
import torch
import random
import sys
import os
import math
import json
import torch_npu
from paged_attention.paged_attention_test_data_generator import PagedAttentionDataGenerator
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402
np.random.seed(1)
random.seed(1)
MAX_SEQ_LEN = 1024


data_generator = PagedAttentionDataGenerator()
data_generator.test_pa_fp16_case_norm_mask_multi_batch_compress_head()

(q, k, v, block_table, context_lens, mask, golden) = (
    data_generator.q,
    data_generator.key_cache,
    data_generator.value_cache,
    data_generator.block_tables,
    data_generator.contex_lens,
    data_generator.alib_mask,
    data_generator.golden_out
)


data = [q, k, v, torch.from_numpy(block_table), torch.from_numpy(context_lens), mask, golden]
in_tensors = [tensor.npu() for tensor in data]
_ = [print(tensor.dtype, tensor.device, tensor.shape) for tensor in in_tensors]

OP_NAME = "PagedAttentionOperation"
PARAM = json.dumps({"headNum":8, "qkScale": (1 / 128 ** 0.5), "kvHeadNum": 8, "maskType": 1, "compressType": 1})

RUN_PARAM = json.dumps({"contextLens": data_generator.contex_lens.tolist()})

class TestPagedAttentionAttentionOperationCompressHead(operation_test.OperationTest):
    def golden_calc(self, input_tensors):
        return [in_tensors[6]]

    def golden_compare(self, out_tensor, golden_out_tensor):
        ratios = [0.001, 0.001, 0.005, 0.005]
        return data_generator.compare_output_data(out_tensor.cpu(), golden_out_tensor.cpu(), ratios)

    def test(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4], in_tensors[5]])
 
if __name__ == '__main__':
    unittest.main()