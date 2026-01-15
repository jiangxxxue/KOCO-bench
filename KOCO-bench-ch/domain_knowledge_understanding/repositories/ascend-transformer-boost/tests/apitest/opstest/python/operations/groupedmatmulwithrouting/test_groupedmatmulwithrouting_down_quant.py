# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import sys
import os
import unittest
import torch
import torch_npu
import numpy as np
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402
np.random.seed(42)

def get_eb(golden: torch.Tensor, actual: torch.Tensor):
    golden = golden.to(torch.float32)
    golden_nmax = torch.clamp(torch.abs(golden), min=1)
    actual_error = actual.to(torch.float32) - golden
    result = torch.mean(actual_error / golden_nmax) <= 2 ** (-7)
    return result

def ref_compare(golden: torch.Tensor, actual: torch.Tensor, thresh: float):
    golden = golden.to(torch.float32)
    golden_nmax = torch.clamp(torch.abs(golden), min=1)
    abs_error = torch.abs(actual.to(torch.float32) - golden)
    result = (abs_error <= thresh * golden_nmax).all()
    return result

class TestMoEDownQuantOperation(operation_test.OperationTest):
    def __gen_test_data(self, num_experts: int, num_tokens: int, hidden_size_in: int, hidden_size_out: int, top_k: int, dtype=torch.float16) -> None:
        activation = np.random.randint(-5, 5, (num_tokens * top_k, hidden_size_in)).astype(np.int8)
        weight = np.random.randint(-5, 5, (num_experts, hidden_size_in, hidden_size_out)).astype(np.int8)
        experts_map = np.array([np.random.permutation(num_experts)[:top_k] for _ in range(num_tokens)])
        experts_index = [np.where(experts_map == i)[0].tolist() for i in range(num_experts)]
        experts_count = [len(each) for each in experts_index]
        ref = np.zeros([num_tokens, hidden_size_out]).astype(np.float32)
        nscale = np.random.uniform(-0.1, 0.1, size=(num_experts, hidden_size_out)).astype(np.float32)
        mscale = np.random.uniform(-0.1, 0.1, size=(num_tokens * top_k, 1)).astype(np.float32)
        op_param = self.PARAM
        self.ksize = hidden_size_in

        e_start = 0
        for e in range(num_experts):
            if len(experts_index[e]) == 0:
                continue
            a = activation[e_start : e_start + experts_count[e]]
            b = weight[e]
            c = torch.mm(torch.tensor(a).int(), torch.tensor(b).int()).numpy()
            ref[experts_index[e]] += (c * mscale[e_start : e_start + experts_count[e]]) * nscale[e]
            e_start += experts_count[e]
        
        if op_param["transposeB"]:
            weight = np.transpose(weight, (0, 2, 1))
        self.activation = activation
        self.weight = weight
        self.experts_index = [item for sublist in experts_index for item in sublist]
        self.experts_count = experts_count
        self.mscale = mscale
        self.nscale = nscale
        self.ref = torch.tensor(ref).to(dtype)
        return
    
    def golden_calc(self, in_tensors):
        return [self.ref]

    def golden_compare(self, out_tensor, golden_out_tensor):
        eb = get_eb(golden_out_tensor.cpu(), out_tensor.cpu())
        cmp = ref_compare(golden_out_tensor.cpu(), out_tensor.cpu(), 2**-7 if self.ksize < 2048 else 2**-6)
        return eb and cmp

    def testcase_nd(self):
        if  operation_test.get_soc_version() == 'Ascend310B':
            print("this testcase don't supports Ascend310B")
            return True
        num_experts = 160
        num_tokens = 128
        hidden_size_in = 192 
        hidden_size_out = 5120
        top_k = 6
        dtype = torch.float16
        is_nz_weight = 0
        OP_NAME = "GroupedMatmulWithRoutingOperation"
        self.PARAM = {"topK": 6, "groupedMatmulType":1, "transposeB": False, "outDataType": 1}
        self.__gen_test_data(num_experts, num_tokens, hidden_size_in, hidden_size_out, top_k, dtype)
        weigth = torch.tensor(self.weight).char().npu()
        if is_nz_weight:
            weigth = torch_npu.npu_format_cast(weigth, 29)

        self.execute(OP_NAME, self.PARAM, [torch.tensor(self.activation).char().npu(), weigth, torch.tensor(self.experts_count).int().cumsum(-1).int().npu(), torch.tensor(self.experts_index).int().npu(), torch.tensor(self.nscale).float().npu() , torch.tensor(self.mscale).float().npu()])

    def testcase_nz(self):
        if  operation_test.get_soc_version() == 'Ascend310B':
            print("this testcase don't supports Ascend310B")
            return True
        num_experts = 160
        num_tokens = 128
        hidden_size_in = 192 
        hidden_size_out = 5120
        top_k = 6
        dtype = torch.float16
        is_nz_weight = 1
        OP_NAME = "GroupedMatmulWithRoutingOperation"
        self.PARAM = {"topK": 6, "groupedMatmulType":1, "transposeB": True, "outDataType": 1}
        self.__gen_test_data(num_experts, num_tokens, hidden_size_in, hidden_size_out, top_k, dtype)
        weigth = torch.tensor(self.weight).char().npu()
        if is_nz_weight:
            weigth = torch_npu.npu_format_cast(weigth, 29)

        self.execute(OP_NAME, self.PARAM, [torch.tensor(self.activation).char().npu(), weigth, torch.tensor(self.experts_count).int().cumsum(-1).int().npu(), torch.tensor(self.experts_index).int().npu(), torch.tensor(self.nscale).float().npu() , torch.tensor(self.mscale).float().npu()])

if __name__ == '__main__':
    unittest.main()
