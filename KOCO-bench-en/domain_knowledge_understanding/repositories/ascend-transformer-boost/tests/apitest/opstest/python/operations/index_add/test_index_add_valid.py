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

OP_NAME = "IndexAddOperation"
OP_PARAM = {"indexType":2, "axis": 0}

class TestIndexAddValidOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        if OP_PARAM["indexType"] == 2:
            axis = OP_PARAM["axis"]
            cloned_in_tensor_var = in_tensors[0].clone().cpu()
            cloned_in_tensor_indices = in_tensors[1].clone().cpu()
            cloned_in_tensor_updates = in_tensors[2].clone().cpu()
            cloned_in_tensor_valid_indices_num = in_tensors[3].clone().cpu()
            if cloned_in_tensor_valid_indices_num == 0:
                return [cloned_in_tensor_var.clone().npu() - in_tensors[0]]
            cloned_in_tensor_indices = torch.split(cloned_in_tensor_indices, cloned_in_tensor_valid_indices_num.item(), dim=0)[0]
            cloned_in_tensor_updates = torch.split(cloned_in_tensor_updates, cloned_in_tensor_valid_indices_num.item(), dim=0)[0]
            cloned_in_tensor_var.index_add_(axis, cloned_in_tensor_indices, cloned_in_tensor_updates, alpha=1)
            return [cloned_in_tensor_var.clone().npu() - in_tensors[0]]
        
        return []
    
    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.allclose(out_tensor, golden_out_tensor, rtol=0.001, atol=0.001)
    
    def test_float16(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        axis = 0
        n, k = 1024, 4096
        num_indices = 90
        shape0 = (n, k)
        shape1 = (num_indices,)
        shape2 = (num_indices, k)
        shape3 = (1,)
 
        input0 = torch.rand(shape0, dtype=torch.half).npu()
        input1 = torch.from_numpy(np.arange(num_indices).astype(np.int32)).npu()
        input2 = torch.rand(shape2, dtype=torch.half).npu()
        input3 = torch.from_numpy(np.arange(1).astype(np.int32)).npu()
 
        self.execute(OP_NAME, OP_PARAM, [input0, input1, input2, input3])
        
    def test_bfloat16(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        axis = 0
        n, k = 1024, 4096
        num_indices = 90
        shape0 = (n, k)
        shape1 = (num_indices,)
        shape2 = (num_indices, k)
        shape3 = (1,)
 
        input0 = torch.rand(shape0, dtype=torch.bfloat16).npu()
        input1 = torch.from_numpy(np.arange(num_indices).astype(np.int32)).npu()
        input2 = torch.rand(shape2, dtype=torch.bfloat16).npu()
        input3 = torch.from_numpy(np.arange(1).astype(np.int32)).npu()
 
        self.execute(OP_NAME, OP_PARAM, [input0, input1, input2, input3])
        
if __name__ == '__main__':
    unittest.main()
