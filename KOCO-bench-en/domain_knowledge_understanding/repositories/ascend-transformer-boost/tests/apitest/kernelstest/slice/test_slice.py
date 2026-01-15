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
OP_NAME = "SliceOperation"
 
class TestSlicepose(op_test.OpTest):
    def golden_calc(self, in_tensors):
        offsets = self.op_desc["specificParam"]["offsets"]
        size = self.op_desc["specificParam"]["size"]
        at_out_ref_tensor = in_tensors[0]
        for i in range(len(in_tensors[0].shape)):
            offset_value = offsets[i]
            size_value = size[i]
            if offsets[i] < 0:
                offset_value = offsets[i] + in_tensors.shape[i]
            if size[i] == -1:
                size_value = in_tensors.shape[i] - offsets[i]
            at_out_ref_tensor = at_out_ref_tensor.narrow(i, offset_value, size_value)
        return at_out_ref_tensor.float()
 
    def golden_compare(self, out_tensors, golden_out_tensors):
        if (out_tensors[0].dtype == torch.bfloat16):
            return torch.allclose(out_tensors[0].bfloat16(), golden_out_tensors.bfloat16(), rtol=0.001, atol=0.001)
        return torch.allclose(out_tensors[0].float(), golden_out_tensors.float(), rtol=0.001, atol=0.001)
    
    @op_test.only_910b
    def test_slice_bf16(self):
        shape = (16,20)
        param = {"offsets": [2,4],
                 "size": [10,10]}
        shape1 = list()
        for i in range(len(param["offsets"])):
            offset_value = param["offsets"][i]
            size_value = param["size"][i]
            if param["offsets"][i] < 0:
                offset_value = param["offsets"][i] + shape[i]
            if param["size"][i] == -1:
                size_value = shape[i] - param["offsets"][i]
            shape1.append(size_value)
        shape1 = tuple(shape1)
        input0 = np.random.uniform(low=-100, high=100, size=shape).astype(np.float16)
        self.set_param(OP_NAME, param)
 
        self.execute([torch.from_numpy(input0).bfloat16()],
                     [torch.zeros(shape1).bfloat16()])

    def test_slice_fp16(self):
        shape = (16,20)
        param = {"offsets": [2,4],
                 "size": [10,10]}
        shape1 = list()
        for i in range(len(param["offsets"])):
            offset_value = param["offsets"][i]
            size_value = param["size"][i]
            if param["offsets"][i] < 0:
                offset_value = param["offsets"][i] + shape[i]
            if param["size"][i] == -1:
                size_value = shape[i] - param["offsets"][i]
            shape1.append(size_value)
        shape1 = tuple(shape1)
        
        input0 = np.random.uniform(low=-100, high=100, size=shape).astype(np.float16)
        self.set_param(OP_NAME, param)
 
        self.execute([torch.from_numpy(input0).half()],
                     [torch.zeros(shape1).half()])
if __name__ == '__main__':
    unittest.main()
