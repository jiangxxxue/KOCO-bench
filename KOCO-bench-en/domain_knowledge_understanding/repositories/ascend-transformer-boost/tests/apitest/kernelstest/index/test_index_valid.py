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
import logging
OP_NAME = "IndexOperation"
INDEX_ADD_VALID = 2
class TestIndex(op_test.OpTest):
    def golden_calc(self, in_tensors):
        if self.op_desc["specificParam"]["indexType"] == INDEX_ADD_VALID:
            axis = self.op_desc["specificParam"]["axis"]
            if in_tensors[3]==0:
                return [in_tensors[0].float()]
            splits1 = torch.split(in_tensors[1], in_tensors[3].item(), dim = 0)
            golden_indices = splits1[0]
            splits2 = torch.split(in_tensors[2], in_tensors[3].item(), dim = 0)
            golden_updates = splits2[0].float()
            golden_var = in_tensors[0].float()
            golden_var = golden_var.index_add_(axis, golden_indices, golden_updates, alpha=1)
            return [golden_var]
        return []

    def golden_compare(self, out_tensors, golden_out_tensors): 
        err = 2**(-8)
        if out_tensors[0].dtype == torch.bfloat16:
            err = 2**(-7)
        elif out_tensors[0].dtype == torch.float32:
            err = 2**(-14)     
        out_tensors[0] = out_tensors[0].to(torch.float32)
        golden_out_tensors[0] = golden_out_tensors[0].to(torch.float32)
        diff = torch.abs(torch.subtract(out_tensors[0], golden_out_tensors[0]))
        tensor_max = torch.maximum(torch.ones(golden_out_tensors[0].shape, dtype=golden_out_tensors[0].dtype), torch.abs(golden_out_tensors[0]))
        if torch.any(torch.greater(diff, err * tensor_max)):
            golden_out_numpy = golden_out_tensors[0].numpy()
            out_numpy = out_tensors[0].numpy()
            logging.debug("new golden result: failed")
            logging.debug(f'golden is: {golden_out_numpy}')
            logging.debug(f'actual results: {out_numpy}')
            return False
        logging.debug("new golden result: true")
        return True

    #   indice比var小的情况
    @op_test.only_910b
    def test_index_add_valid_f16_case0(self):
        axis = 0
        indicesTotal = 512
        indicesValid = 1024
        valueSize = 2048
        validIndexLength = 256
        shape0 = (indicesValid, valueSize)
        shape1 = (indicesTotal, )
        shape2 = (indicesTotal, valueSize)
        #input0: var   input1: indices   input2: updates   input3: validIndicesNum
        input0 = torch.zeros(shape0, dtype=torch.half)
        input1 = torch.randint(0, indicesValid, shape1, dtype=torch.int32)
        input2 = np.random.uniform(low=-5, high=5, size=shape2).astype(np.float16)
        input2 = torch.from_numpy(input2)
        input3 = torch.tensor([validIndexLength], dtype=torch.int32)
        OP_PARAM = {"indexType": INDEX_ADD_VALID, "axis": axis}
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([input0, input1, input2, input3],
                     [0])

    #   模型常见场景
    @op_test.only_910b
    def test_index_add_valid_f16_case1(self):
        axis = 0
        indicesTotal = 10000
        indicesValid = 1024
        valueSize = 2048
        validIndexLength = 8192
        shape0 = (indicesValid, valueSize)
        shape1 = (indicesTotal, )
        shape2 = (indicesTotal, valueSize)
        #input0: var   input1: indices   input2: updates   input3: validIndicesNum
        input0 = torch.zeros(shape0, dtype=torch.half)
        input1 = torch.randint(0, indicesValid, shape1, dtype=torch.int32)
        input2 = np.random.uniform(low=-5, high=5, size=shape2).astype(np.float16)
        input2 = torch.from_numpy(input2)
        input3 = torch.tensor([validIndexLength], dtype=torch.int32)
        OP_PARAM = {"indexType": INDEX_ADD_VALID, "axis": axis}
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([input0, input1, input2, input3],
                     [0])
    
    #   32不对齐的情况
    @op_test.only_910b
    def test_index_add_valid_f16_case2(self):
        axis = 0
        indicesTotal = 222
        indicesValid = 111
        valueSize = 2048
        validIndexLength = 111
        shape0 = (indicesValid, valueSize)
        shape1 = (indicesTotal, )
        shape2 = (indicesTotal, valueSize)
        #input0: var   input1: indices   input2: updates   input3: validIndicesNum
        input0 = torch.zeros(shape0, dtype=torch.half)
        input1 = torch.randint(0, indicesValid, shape1, dtype=torch.int32)
        input2 = np.random.uniform(low=-5, high=5, size=shape2).astype(np.float16)
        input2 = torch.from_numpy(input2)
        input3 = torch.tensor([validIndexLength], dtype=torch.int32)
        OP_PARAM = {"indexType": INDEX_ADD_VALID, "axis": axis}
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([input0, input1, input2, input3],
                     [0])

    #   长序列场景 耗时最久
    @op_test.only_910b
    def test_index_add_valid_f16_case3(self):
        axis = 0
        indicesTotal = 270000
        indicesValid = 1024
        valueSize = 4096
        validIndexLength = 262144   # 256k=262144 
        shape0 = (indicesValid, valueSize)
        shape1 = (indicesTotal, )
        shape2 = (indicesTotal, valueSize)
        #input0: var   input1: indices   input2: updates   input3: validIndicesNum
        input0 = torch.zeros(shape0, dtype=torch.half)
        input1 = torch.randint(0, indicesValid, shape1, dtype=torch.int32)
        input2 = np.random.uniform(low=-5, high=5, size=shape2).astype(np.float16)
        input2 = torch.from_numpy(input2)
        input3 = torch.tensor([validIndexLength], dtype=torch.int32)
        OP_PARAM = {"indexType": INDEX_ADD_VALID, "axis": axis}
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([input0, input1, input2, input3],
                     [0])

    #   索引集中的情况
    @op_test.only_910b
    def test_index_add_valid_f16_case4(self):
        axis = 0
        indicesTotal = 512
        indicesValid = 2
        valueSize = 512
        validIndexLength = 512
        shape0 = (indicesValid, valueSize)
        shape1 = (indicesTotal, )
        shape2 = (indicesTotal, valueSize)
        #input0: var   input1: indices   input2: updates   input3: validIndicesNum
        input0 = torch.zeros(shape0, dtype=torch.half)
        input1 = torch.randint(0, indicesValid, shape1, dtype=torch.int32)
        input2 = np.random.uniform(low=-5, high=5, size=shape2).astype(np.float16)
        input2 = torch.from_numpy(input2)
        input3 = torch.tensor([validIndexLength], dtype=torch.int32)
        OP_PARAM = {"indexType": INDEX_ADD_VALID, "axis": axis}
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([input0, input1, input2, input3],
                     [0])

    #   8192边缘数据
    @op_test.only_910b
    def test_index_add_valid_f16_case5(self):
        axis = 0
        indicesTotal = 1024
        indicesValid = 1024
        valueSize = 8192
        validIndexLength = 1024
        shape0 = (indicesValid, valueSize)
        shape1 = (indicesTotal, )
        shape2 = (indicesTotal, valueSize)
        #input0: var   input1: indices   input2: updates   input3: validIndicesNum
        input0 = torch.zeros(shape0, dtype=torch.half)
        input1 = torch.randint(0, indicesValid, shape1, dtype=torch.int32)
        input2 = np.random.uniform(low=-5, high=5, size=shape2).astype(np.float16)
        input2 = torch.from_numpy(input2)
        input3 = torch.tensor([validIndexLength], dtype=torch.int32)
        OP_PARAM = {"indexType": INDEX_ADD_VALID, "axis": axis}
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([input0, input1, input2, input3],
                     [0])

    #   bf16数据
    #   模型常见场景
    @op_test.only_910b
    def test_index_add_valid_bf16_case0(self):
        axis = 0
        indicesTotal = 10000
        indicesValid = 1024
        valueSize = 2048
        validIndexLength = 8192
        shape0 = (indicesValid, valueSize)
        shape1 = (indicesTotal, )
        shape2 = (indicesTotal, valueSize)
        #input0: var   input1: indices   input2: updates   input3: validIndicesNum
        input0 = torch.zeros(shape0, dtype=torch.bfloat16)
        input1 = torch.randint(0, indicesValid, shape1, dtype=torch.int32)
        input2 = np.random.uniform(low=-5, high=5, size=shape2).astype(np.float16)
        input2 = torch.from_numpy(input2).bfloat16()
        input3 = torch.tensor([validIndexLength], dtype=torch.int32)
        OP_PARAM = {"indexType": INDEX_ADD_VALID, "axis": axis}
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([input0, input1, input2, input3],
                     [0])

    @op_test.only_910b
    def test_index_add_valid_bf16_case1(self):
        axis = 0
        indicesTotal = 8192
        indicesValid = 1024
        valueSize = 2048
        validIndexLength = 1
        shape0 = (indicesValid, valueSize)
        shape1 = (indicesTotal, )
        shape2 = (indicesTotal, valueSize)
        #input0: var   input1: indices   input2: updates   input3: validIndicesNum
        input0 = torch.zeros(shape0, dtype=torch.bfloat16)
        input1 = torch.randint(0, indicesValid, shape1, dtype=torch.int32)
        input2 = torch.empty(indicesTotal, valueSize, dtype=torch.bfloat16)
        torch.nn.init.uniform_(input2, a=-5, b=5)  # 填充范围为 (-5, 5)
        input3 = torch.tensor([validIndexLength], dtype=torch.int32)
        OP_PARAM = {"indexType": INDEX_ADD_VALID, "axis": axis}
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([input0, input1, input2, input3],
                     [0])
        
        #   长序列场景 耗时最久
    @op_test.only_910b
    def test_index_add_valid_bf16_case2(self):
        axis = 0
        indicesTotal = 270000
        indicesValid = 1024
        valueSize = 4096
        validIndexLength = 262144   # 256k=262144 
        shape0 = (indicesValid, valueSize)
        shape1 = (indicesTotal, )
        shape2 = (indicesTotal, valueSize)
        #input0: var   input1: indices   input2: updates   input3: validIndicesNum
        input0 = torch.empty(indicesValid, valueSize, dtype=torch.bfloat16)
        torch.nn.init.uniform_(input0, a=-5, b=5)  # 填充范围为 (-5, 5)
        input1 = torch.randint(0, indicesValid, shape1, dtype=torch.int32)
        # 生成均匀分布的张量
        input2 = torch.empty(indicesTotal, valueSize, dtype=torch.bfloat16)
        torch.nn.init.uniform_(input2, a=-5, b=5)  # 填充范围为 (-5, 5)
        input3 = torch.tensor([validIndexLength], dtype=torch.int32)
        OP_PARAM = {"indexType": INDEX_ADD_VALID, "axis": axis}
        self.set_param(OP_NAME, OP_PARAM)
        self.execute([input0, input1, input2, input3],
                     [0])
        
if __name__ == '__main__':
    unittest.main()
