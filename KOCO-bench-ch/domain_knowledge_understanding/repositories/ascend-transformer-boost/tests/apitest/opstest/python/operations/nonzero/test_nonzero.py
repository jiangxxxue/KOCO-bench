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

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

OP_NAME = "NonzeroOperation"
PARAM = {}

class TestNonzeroOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        num_non_negative = torch.count_nonzero(in_tensors[0])
        paddingNum = in_tensors[0].numel() - num_non_negative
        padding = torch.zeros((in_tensors[0].shape[0], paddingNum)).npu()
        result = torch.stack(list(torch.nonzero(in_tensors[0], as_tuple=True))).npu()
        result = torch.cat((result, padding), dim=-1).long()

        return [result, torch.tensor(num_non_negative).long().npu()]
    
    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.allclose(out_tensor, golden_out_tensor, rtol=0, atol=0)
    
    def test(self):
        if operation_test.get_soc_version() != 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        input0 = np.random.randint(0, 2, [2, 490]).astype(np.int64)
        input0 = torch.from_numpy(input0).npu()
        self.execute(OP_NAME, PARAM, [input0])
    
        
if __name__ == '__main__':
    unittest.main()
