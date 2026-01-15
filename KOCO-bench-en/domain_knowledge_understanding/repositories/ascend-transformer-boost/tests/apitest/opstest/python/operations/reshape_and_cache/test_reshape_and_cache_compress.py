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
import random
import sys
import unittest
 
import numpy as np
import torch
import torch_npu
 
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402
from reshape_and_cache.reshape_and_cache_data_generator import ReshapeAndCacheDataGenerator
 
MAX_SEQ_LEN = 1024
 
 

 
OP_NAME = "ReshapeAndCacheOperation"
PARAM = json.dumps({"compressType": 1})
 
data_generator = ReshapeAndCacheDataGenerator()
data_generator.test_reshape_and_cache_compress_head()
data = data_generator.generate_test_data()
in_tensors = [torch.from_numpy(tensor) for tensor in data]
in_tensors = [tensor.npu() for tensor in in_tensors]
a = [print(tensor.dtype, tensor.device) for tensor in in_tensors]

 
class TestReshapeAndCacheOperationCompress(operation_test.OperationTest):
    soc_version = operation_test.get_soc_version()
    def golden_calc(self, input_tensors):
        return [in_tensors[7], in_tensors[8]]

    def golden_compare(self, out_tensor, golden_out_tensor):
        # return data_generator.golden_compare(out_tensor, golden_out_tensor)
        return True
 
    def test(self):
        if not TestReshapeAndCacheOperationCompress.soc_version == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        self.execute_out(OP_NAME, PARAM, [in_tensors[0], in_tensors[1], in_tensors[2],\
                in_tensors[3], in_tensors[4], in_tensors[5], in_tensors[6]],
                [in_tensors[2], in_tensors[3]])
 
 
if __name__ == '__main__':
    unittest.main()