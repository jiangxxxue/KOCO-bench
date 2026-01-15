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
import torch.nn.functional as F
import op_test

OP_NAME = "IndexOperation"
INDEX_ADD = 1

class TestIndex(op_test.OpTest):
    def golden_calc(self, in_tensors):
        if self.op_desc["specificParam"]["indexType"] == INDEX_ADD:
            axis = self.op_desc["specificParam"]["axis"]
            in_tensors[0].index_add_(axis, in_tensors[1], in_tensors[2], alpha=in_tensors[3].item())
            return [in_tensors[0]]
        
        return []

    def golden_compare(self, out_tensors, golden_out_tensors):     
        return torch.allclose(golden_out_tensors[0], out_tensors[0], rtol=0.001, atol=0.001)

    def atest_index_add_f16_case0(self):
        axis = 0
        n, k = 1024, 4096
        num_indices = 1024
        shape0 = (n, k)
        shape1 = (num_indices,)
        shape2 = (num_indices, k)
        shape3 = (1,)

        input0 = torch.rand(shape0, dtype=torch.half)
        input1 = torch.from_numpy(np.arange(num_indices).astype(np.int32))
        input2 = torch.rand(shape2, dtype=torch.half)
        input3 = torch.rand(shape3, dtype=torch.half)
        
        OP_PARAM = {"indexType": INDEX_ADD, "axis": axis}
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([input0, input1, input2, input3],
                     [0])

if __name__ == '__main__':
    unittest.main()
