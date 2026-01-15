#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import unittest
import numpy as np
import torch
import op_test

OP_NAME = "ZerosLikeOperation"

class TestZerosLike(op_test.OpTest):
    def golden_calc(self, in_tensors):
        return [torch.zeros_like(in_tensors[0]).float()]

    def golden_compare(self, out_tensors, golden_out_tensors):
        if (out_tensors[0].dtype == torch.float32):
            return torch.allclose(out_tensors[0].float(), golden_out_tensors[0].float(), rtol=0.001, atol=0.001)
    
    @op_test.only_910b
    def test_transpose_bf32(self):
        shape = (8192, 2752)
        input0 = np.random.uniform(low=-100, high=100, size=shape).astype(np.float32)
        self.set_param(OP_NAME, {})

        self.execute([torch.from_numpy(input0).float()],
                     [torch.ones(shape).float()])

if __name__ == '__main__':
    unittest.main()