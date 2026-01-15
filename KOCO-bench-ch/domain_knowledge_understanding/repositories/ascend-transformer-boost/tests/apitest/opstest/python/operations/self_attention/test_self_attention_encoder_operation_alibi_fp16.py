#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import json
import math
import os
import sys
import unittest
import random
import numpy as np
import torch
import torch_npu

np.random.seed(0)

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402
from self_attention.self_attention_test_data_generator import SelfAttentionTestDataGenerator

data_generator = SelfAttentionTestDataGenerator()

data = data_generator.test_flash_attention_case_fa_encoder_nocache_bf16_alibi()
param_seqlen = data[4]
data[4] = torch.from_numpy(np.array(data[4]).astype(np.int32))

in_tensors = [tensor.npu().contiguous() for tensor in data]

OP_NAME = "SelfAttentionOperation"
PARAM = json.dumps({"headNum": 12, "qkScale": 1, "kvHeadNum": 1,
                    "calcType": 3, "maskType": 2, "isTriuMask": 1, "kernelType": 0})
RUN_PARAM = json.dumps({"seqLen": param_seqlen})

class TestFlashAttentionEncoderOperationAlibiFp16(operation_test.OperationTest):
    def golden_calc(self, input_tensors):
        return [in_tensors[5]]

    def golden_compare(self, out_tensor, golden_out_tensor):
        ratios = [0.001, 0.001, 0.005, 0.005]
        return data_generator.compare_output_data(out_tensor.cpu(), golden_out_tensor.cpu(), ratios)

    def test(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.execute_with_param(OP_NAME, PARAM, RUN_PARAM, [
            in_tensors[0], in_tensors[1], in_tensors[2], in_tensors[3], in_tensors[4]
        ])


if __name__ == '__main__':
    unittest.main()
