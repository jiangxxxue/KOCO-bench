#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
import operation_test
epsilon = 1e-8
OP_NAME = "GatherPreRmsNormOperation"
 
class TestGatherPreRmsNormOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        data_type = in_tensors[0].dtype
        # in_tensors = in_tensors.cpu()
        # epsilon = self.op_desc["specificParam"]["epsilon"]
        #x, res_in, indices, g
        input0 = in_tensors[0].clone().detach().float().cpu()
        input1 = in_tensors[1].clone().detach().float().cpu()
        input2 = in_tensors[2].clone().detach().to(torch.int64).cpu()
        input3 = in_tensors[3].clone().detach().float().cpu()
        # use gather to obtain indexed input_x
        indices_brcb = input2.unsqueeze(1)
        index = indices_brcb.repeat(1, input1.shape[1])
        try:
            res_in_after_gather = input1.gather(0, index)
        except RuntimeError as e:
            print("some index values are out of range", e)
        # rmsnorm compute
        input_sum = input0 + res_in_after_gather
        square_sum = torch.sum(torch.square(input_sum), axis=-1, keepdims=True)
        factor = 1.0 / torch.sqrt(square_sum / input_sum.shape[-1] + epsilon)
        output = input_sum * factor * input3
        return [torch.tensor(output).to(float).npu(), torch.tensor(input_sum).to(data_type).npu()]
 
    def golden_compare(self, out_tensors, golden_out_tensors):
        out_tensors = out_tensors.cpu()
        golden_out_tensors = golden_out_tensors.cpu()
        diff0 = torch.abs(torch.subtract(out_tensors[0].float(), golden_out_tensors[0].float()))
        tensor_max0 = torch.maximum(torch.ones(golden_out_tensors[0].shape, dtype=golden_out_tensors[0].dtype, device=golden_out_tensors[0].device),
                                   torch.abs(golden_out_tensors[0])).float()
 
        factor0_1, factor0_2 = 2**(-11), 2**(-14) # precision standard for float dtype
        if torch.any(torch.greater(diff0, factor0_1 * tensor_max0)):
            print("[new standards] output0 accuracy failed")
            return False
        if torch.any(torch.greater(torch.abs(torch.mean(torch.div(diff0, tensor_max0))), factor0_2)):
            print("[new standards] output0 eb failed")
            return False
        # golden compare for output1
        diff1 = torch.abs(torch.subtract(out_tensors[1].float(), golden_out_tensors[1].float()))
        tensor_max1 = torch.maximum(torch.ones(golden_out_tensors[1].shape, dtype=golden_out_tensors[1].dtype, device=golden_out_tensors[0].device),
                                   torch.abs(golden_out_tensors[1])).float()
        factor1_1, factor1_2 = 2**(-8), 2**(-10)  # precision standard for float16 dtype
        if golden_out_tensors[1].dtype == torch.bfloat16: 
            factor1_1, factor1_2 = 2**(-7), 2**(-7)  # precision standard for bfloat16 dtype
        if torch.any(torch.greater(diff1, factor1_1 * tensor_max1)):
            print("[new standards] output1 accuracy failed")
            return False
        if torch.any(torch.greater(torch.abs(torch.mean(torch.div(diff1, tensor_max1))), factor1_2)):
            print("[new standards] output1 eb failed")
            return False
 
        return True
 
    def test1(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return True
        res_tokens = 16
        ind_tokens = 16
        hidden_size = 7168
        input0 = torch.empty((ind_tokens, hidden_size), dtype=torch.half).uniform_(0, 256)
        input1 = torch.empty((res_tokens, hidden_size), dtype=torch.half).uniform_(0, 256)
        input2 = torch.randint(0, res_tokens, (ind_tokens,), dtype=torch.int32)
        input3 = torch.empty((1, hidden_size), dtype=torch.half).uniform_(0, 256)
        in_tensors = [input0.npu(), input1.npu(), input2.npu(), input3.npu()]
 
        self.execute(OP_NAME, {"layerType": 4, "gatherPreRMSNormParam":{"epsilon": 1e-8}},
                     in_tensors)
 
if __name__ == '__main__':
    unittest.main()
