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
import operation_test
import random
OP_NAME = "StridedBatchMatmulOperation"
OP_PARAM = {"transA": None, "transB": None, "m": None, "k": None, "n": None, "lda": None, "ldb": None, "ldc": None, "strideA": None, "strideB": None, "strideC": None, "batch": None, "headNum": None}

class TestStridedBatchMatmulOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        A = in_tensors[0].flatten()
        B = in_tensors[1].flatten()

        batchStartA = 0
        batchStartB = 0
        batchStartC = 0
        C = torch.zeros(sum([OP_PARAM["m"][i] * OP_PARAM["n"][i] for i in range(OP_PARAM["batch"])]) * OP_PARAM["headNum"], dtype=torch.float16, device=A.device)

        for i in range(OP_PARAM["batch"]):
            for j in range(OP_PARAM["headNum"]):
                listA = []
                listB = []
                rowA = OP_PARAM["m"][i] if not OP_PARAM["transA"] else OP_PARAM["k"][i]
                colA = OP_PARAM["k"][i] if not OP_PARAM["transA"] else OP_PARAM["m"][i]
                for t in range(rowA):
                    startA = OP_PARAM["lda"][i] * t + OP_PARAM["strideA"][i] * j + batchStartA
                    endA = startA + colA
                    listA.append(A[startA:endA])
                rowB = OP_PARAM["k"][i] if not OP_PARAM["transB"] else OP_PARAM["n"][i]
                colB = OP_PARAM["n"][i] if not OP_PARAM["transB"] else OP_PARAM["k"][i]
                for t in range(rowB):
                    startB = OP_PARAM["ldb"][i] * t +  OP_PARAM["strideB"][i] * j + batchStartB
                    endB = startB + colB
                    listB.append(B[startB:endB])
                matA = torch.stack(listA)
                matB = torch.stack(listB)
                matA = torch.transpose(matA, 0, 1) if OP_PARAM["transA"] else matA
                matB = torch.transpose(matB, 0, 1) if OP_PARAM["transB"] else matB
                matC = torch.matmul(matA.float(), matB.float()).half()
                for t in range(matC.shape[0]):
                    startC = OP_PARAM["ldc"][i] * t + OP_PARAM["strideC"][i] * j + batchStartC
                    endC = startC + matC.shape[1]
                    C[startC:endC] = matC[t, :]
            batchStartA += OP_PARAM["m"][i] * OP_PARAM["k"][i] * OP_PARAM["headNum"]
            batchStartB += OP_PARAM["n"][i] * OP_PARAM["k"][i] * OP_PARAM["headNum"]
            batchStartC += OP_PARAM["m"][i] * OP_PARAM["n"][i] * OP_PARAM["headNum"]
        return [C]

    def golden_compare(self, out_tensor, golden_out_tensor):
        return torch.allclose(out_tensor, golden_out_tensor, rtol=0.001, atol=0.001)

    def test_add_bmm1(self): #bmm2_grad1
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True

        batch = random.randint(1, 8)
        seqlen = [random.randint(1, 512) for i in range(batch)]
        head_num = random.randint(4, 16)
        head_size = random.randint(64, 256)
        sum_seqlen = sum(seqlen)
        hidden_size = head_size * head_num
        seqlen_squared = [x**2 for x in seqlen]
        shapeC = (head_num * sum(seqlen_squared), )

        OP_PARAM["transA"] = 0
        OP_PARAM["transB"] = 1
        OP_PARAM["m"] = seqlen
        OP_PARAM["k"] = [head_size] * batch
        OP_PARAM["n"] = seqlen
        OP_PARAM["lda"] = [hidden_size] * batch
        OP_PARAM["ldb"] = [hidden_size] * batch
        OP_PARAM["ldc"] = seqlen
        OP_PARAM["strideA"] = [head_size] * batch
        OP_PARAM["strideB"] = [head_size] * batch
        OP_PARAM["strideC"] = seqlen_squared

        OP_PARAM["batch"] = batch
        OP_PARAM["headNum"] = head_num

        A = torch.randn(sum_seqlen, hidden_size).npu().half()
        B = torch.randn(sum_seqlen, hidden_size).npu().half()

        input_tensor = [A, B]

        self.execute(OP_NAME, OP_PARAM, input_tensor)

    def test_add_bmm1_grad1(self): #bmm2
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True

        batch = random.randint(1, 8)
        seqlen = [random.randint(1, 512) for i in range(batch)]
        head_num = random.randint(4, 16)
        head_size = random.randint(64, 256)

        sum_seqlen = sum(seqlen)
        hidden_size = head_size * head_num
        seqlen_squared = [x**2 for x in seqlen]
        shapeA = (head_num * sum(seqlen_squared), )

        OP_PARAM["transA"] = False
        OP_PARAM["transB"] = False
        OP_PARAM["m"] = seqlen
        OP_PARAM["k"] = seqlen
        OP_PARAM["n"] = [head_size] * batch
        OP_PARAM["lda"] = seqlen
        OP_PARAM["ldb"] = [hidden_size] * batch
        OP_PARAM["ldc"] = [hidden_size] * batch
        OP_PARAM["strideA"] = seqlen_squared
        OP_PARAM["strideB"] = [head_size] * batch
        OP_PARAM["strideC"] = [head_size] * batch

        OP_PARAM["batch"] = batch
        OP_PARAM["headNum"] = head_num

        A = torch.randn(shapeA).npu().half()
        B = torch.randn(sum_seqlen, hidden_size).npu().half()

        input_tensor = [A, B]

        self.execute(OP_NAME, OP_PARAM, input_tensor)

    def test_add_bmm2_grad2(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True

        batch = random.randint(1, 8)
        seqlen = [random.randint(1, 512) for i in range(batch)]
        head_num = random.randint(4, 16)
        head_size = random.randint(64, 256)

        sum_seqlen = sum(seqlen)
        hidden_size = head_size * head_num
        seqlen_squared = [x**2 for x in seqlen]
        shapeA = (head_num * sum(seqlen_squared), )

        OP_PARAM["transA"] = True
        OP_PARAM["transB"] = False
        OP_PARAM["m"] = seqlen
        OP_PARAM["k"] = seqlen
        OP_PARAM["n"] = [head_size] * batch
        OP_PARAM["lda"] = seqlen
        OP_PARAM["ldb"] = [hidden_size] * batch
        OP_PARAM["ldc"] = [hidden_size] * batch
        OP_PARAM["strideA"] = seqlen_squared
        OP_PARAM["strideB"] = [head_size] * batch
        OP_PARAM["strideC"] = [head_size] * batch

        OP_PARAM["batch"] = batch
        OP_PARAM["headNum"] = head_num

        A = torch.randn(shapeA).npu().half()
        B = torch.randn(sum_seqlen, hidden_size).npu().half()

        input_tensor = [A, B]

        self.execute(OP_NAME, OP_PARAM, input_tensor)

    def test_add_bmm1_grad2(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True

        batch = random.randint(1, 8)
        seqlen = [random.randint(1, 512) for i in range(batch)]
        head_num = random.randint(4, 16)
        head_size = random.randint(64, 256)

        sum_seqlen = sum(seqlen)
        hidden_size = head_size * head_num
        seqlen_squared = [x**2 for x in seqlen]
        shapeB = (head_num * sum(seqlen_squared), )

        OP_PARAM["transA"] = True
        OP_PARAM["transB"] = False
        OP_PARAM["m"] = [head_size] * batch
        OP_PARAM["k"] = seqlen
        OP_PARAM["n"] = seqlen
        OP_PARAM["lda"] = [hidden_size] * batch
        OP_PARAM["ldb"] = seqlen
        OP_PARAM["ldc"] = seqlen
        OP_PARAM["strideA"] = [head_size] * batch
        OP_PARAM["strideB"] = seqlen_squared
        OP_PARAM["strideC"] = [s*head_size for s in seqlen]

        OP_PARAM["batch"] = batch
        OP_PARAM["headNum"] = head_num

        A = torch.randn(sum_seqlen, hidden_size).npu().half()
        B = torch.randn(shapeB).npu().half()

        input_tensor = [A, B]

        self.execute(OP_NAME, OP_PARAM, input_tensor)


if __name__ == '__main__':
    unittest.main()
