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


sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

OP_NAME = "GatherOperation"
PARAM = {"axis":1}

class TestGatherOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        axis = PARAM["axis"]
        if axis == 0:
            if in_tensors[0].ndim == 2 and in_tensors[1].ndim == 2:
                embedding = torch.nn.Embedding(in_tensors[0].shape[0],in_tensors[0].shape[1])
                golden_result = embedding(in_tensors[1]).detach()
                return [golden_result.npu()]
        outputSize = []
        dim0 = 1
        for i in range(0,axis):
            outputSize.append(in_tensors[0].shape[i])
            dim0 *= in_tensors[0].shape[i]
        dim1 = in_tensors[0].shape[axis]
        for i in range(0,in_tensors[1].ndim):
            outputSize.append(in_tensors[1].shape[i])
        dim2 = 1
        for i in range(axis + 1,in_tensors[0].ndim):
            outputSize.append(in_tensors[0].shape[i])
            dim2 *= in_tensors[0].shape[i]
        inputFlatten = in_tensors[0].clone().reshape(-1)
        indicesFlatten = in_tensors[1].clone().reshape(-1)
        print("outputSize",outputSize)
        golden_result_np = torch.zeros(outputSize,dtype=torch.float16).reshape(-1).numpy()
        idx = 0
        for i in range(0,dim0):
            inputIdx = i * dim1 * dim2
            for indice in indicesFlatten:
                for k in range(0,dim2):
                    golden_result_np[idx] = inputFlatten[inputIdx + indice * dim2 + k]
                    idx+=1
        golden_result = torch.from_numpy(golden_result_np).reshape(outputSize)
        print(in_tensors[0].dtype)
        if in_tensors[0].dtype == torch.bfloat16:
            golden_result = golden_result.bfloat16()
        return [golden_result]

    def test_float16(self):
        intensor0=torch.randn([3,5],dtype=torch.float16)
        print(intensor0.size())
        print(intensor0.shape)
        intensor1=torch.randint(0, 5, [3,4],dtype=torch.int64)
        self.execute(OP_NAME, PARAM, [intensor0.npu(),
                                      intensor1.npu()])

    def test_bfloat16(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        intensor0=torch.randn([3,5],dtype=torch.bfloat16)
        print(intensor0)
        print(intensor0.size())
        print(intensor0.shape)
        intensor1=torch.randint(0, 5, [3,4],dtype=torch.int64)
        self.execute(OP_NAME, PARAM, [intensor0.npu(),
                                      intensor1.npu()])

if __name__ == '__main__':
    unittest.main()
