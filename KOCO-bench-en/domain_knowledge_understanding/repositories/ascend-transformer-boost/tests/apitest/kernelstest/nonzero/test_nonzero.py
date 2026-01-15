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
import torch.nn as nn
import op_test
from tensor_file import read_tensor

OP_NAME = "NonzeroOperation"


class TestNonZero(op_test.OpTest):
    def golden_calc(self, in_tensors):
        result = torch.stack(list(torch.nonzero(in_tensors[0], as_tuple=True)))
        return [result]

    def golden_compare(self, out_tensors, golden_out_tensors):
        numTrue = out_tensors[1][0]
        out_tensors[0] = out_tensors[0][:, :numTrue]
        
        return torch.allclose(out_tensors[0], golden_out_tensors[0], rtol=0, atol=0)
    
    @op_test.only_910b
    def test(self):
        input0 = np.random.randint(0, 2, [2, 490]).astype(np.int64)
        input0 = torch.from_numpy(input0)
        self.set_param(OP_NAME, {})
        self.execute([input0], [torch.zeros(2, 980).long(), torch.tensor([0]).long()])


if __name__ == '__main__':
    unittest.main()
