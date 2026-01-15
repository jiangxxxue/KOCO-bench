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
import math
import json
import torch
import torch_npu
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

sys.path.append("../../../operations")
import operation_test

LAYER_NAME = "ChatGlm6BLayerDecoderFlashAttentionOperation"

layer_id = 0
epsilon = 1e-05
num_attention_heads = 32
head_dim = 128
num_layers = 28
qkScale = float(layer_id + 1)
qScale = 1 / (math.sqrt(head_dim) * float(layer_id + 1))
residualAddScale =  math.sqrt(2 * num_layers)
intensor_num = 22
tensorPath = os.path.join(os.getenv("ATB_TESTDATA"), "tensors/layers/ChatGlm6BLayerDecoderFlashAttentionOperation/after/")


class TestChatGlm6BDecoderFlashAttention(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        return [self.get_tensor(tensorPath + "outtensor0.bin").npu()]

    def test(self):
        layer_param = {
            "layerNormEps": epsilon,
            "headNum": num_attention_heads,
            "headDim": head_dim,
            "qScale": qScale,
            "qkScale": qkScale,
            "residualAddScale": residualAddScale
        }

        in_tensors = [self.get_tensor(tensorPath + "intensor" + str(i) + ".bin").npu() for i in range(intensor_num)]
        
        # 首token,长度为6的输入
        token_param = json.dumps({"tokenOffset": 6, "seqLen": 6})
        self.execute_with_param(LAYER_NAME, layer_param, token_param, in_tensors)


if __name__ == '__main__':
    unittest.main()