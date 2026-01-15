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
import numpy as np
from torch import nn
from torch.nn.parameter import Parameter

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../operations"))
import operation_test

LAYER_NAME = "Llama65BLayerOperation"

PARAM = {"rmsNormEps": 1e-5}

intensor0 = torch.rand(28, 8, 8192, dtype=torch.float16).npu().half()
intensor1 = torch.rand(8192, dtype=torch.float16).npu().half()
intensor2 = torch.rand(3072, 8192, dtype=torch.float16).npu().half()
intensor3 = torch.rand(8192, 1024, dtype=torch.float16).npu().half()
intensor4 = torch.rand(8192, dtype=torch.float16).npu().half()
intensor5 = torch.rand(5504, 8192, dtype=torch.float16).npu().half()
intensor6 = torch.rand(2752, 8192, dtype=torch.float16).npu().half()
intensor7 = torch.randint(1, 10, size=(28, 8), dtype=torch.int64).npu()
intensor8 = torch.rand(2, 28, 1, 8, 128, dtype=torch.float32).npu()
intensor8 = torch_npu.npu_format_cast(intensor8, 2)
intensor9 = torch.rand(28, 1, 4096, 4096, dtype=torch.float16).npu().half()
intensor10 = torch.rand(28, 8, 4096, 128, dtype=torch.float16).npu().half()
intensor11 = torch.rand(28, 8, 4096, 128, dtype=torch.float16).npu().half()
intensor12 = torch.randint(1, 10, size=(28, 1), dtype=torch.int32).npu() # token offset
intensor13 = torch.randint(1, 10, size=(28, 1), dtype=torch.int32).npu() # seq len
intensor14 = torch.randint(1, 10, size=(1,), dtype=torch.int32).npu()
intensor15 = torch.randint(1, 10, size=(28,), dtype=torch.int32).npu()

outtensor0 = torch.rand(28, 8, 8192, dtype=torch.float16).npu().half()


token_offset_data = intensor12.cpu().reshape(1, 28).tolist()[0]
seq_len_data = intensor13.cpu().reshape(1, 28).tolist()[0]

RUN_PARAM = json.dumps({"tokenOffset": token_offset_data, "seqLen": seq_len_data})

class TestLlama65BLayerOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        return [outtensor0]

    def golden_compare(self, out_tensor, golden_out_tensor):
        return True

    def test(self):
        

        in_tensors = [intensor0, intensor1, intensor2, intensor3, intensor4, intensor5, intensor6, intensor7,
                   intensor8, intensor9, intensor10, intensor11, intensor12, intensor13, intensor14, intensor15]
        
        self.execute_with_param(LAYER_NAME, PARAM, RUN_PARAM, in_tensors)


if __name__ == '__main__':
    unittest.main()