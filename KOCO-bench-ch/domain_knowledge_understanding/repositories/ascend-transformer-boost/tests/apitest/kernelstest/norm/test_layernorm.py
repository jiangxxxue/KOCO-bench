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
from tensor_file import read_tensor


OP_NAME = "NormOperation"
OP_PARAM0 = {"normType": 1, "beginNormAxis": 0, "beginParamsAxis": 0, "epsilon": 1e-6,
             "inGamma": True, "inBeta": True, "outMean": True, "outVarience": True}
OP_PARAM1 = {"normType": 1, "beginNormAxis": 2, "beginParamsAxis": 2, "epsilon": 1e-6}


class TestNorm(op_test.OpTest):
    def golden_calc(self, in_tensors):
        axis = self.op_desc["specificParam"]["beginNormAxis"]
        eps = self.op_desc["specificParam"]["epsilon"]
        layer_norm = torch.nn.LayerNorm(in_tensors[0].shape[axis:], eps)
        if (in_tensors[0].dtype == torch.bfloat16):
            return [layer_norm(in_tensors[0].float()).bfloat16()]
        else:
            return [layer_norm(in_tensors[0].float()).half()]

    def golden_compare(self, out_tensors, golden_out_tensors):
        return torch.allclose(out_tensors[0], golden_out_tensors[0], rtol=0.001, atol=0.001)

    def test_layenorm_1d_large(self):
        shape = (1000000)
        input0 = np.random.uniform(low=-10000, high=10000, size=shape).astype(np.float16)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.from_numpy(input0), torch.ones(shape).half(), torch.zeros(shape).half()],
                     [torch.zeros(shape).half(), torch.zeros((1)).half(), torch.zeros((1)).half()])

    @op_test.only_910b
    def test_layenorm_1d_large_bf16(self):
        shape = (1000000)
        input0 = np.random.uniform(low=-10000, high=10000, size=shape).astype(np.float32)
        self.set_param(OP_NAME, OP_PARAM0)
        self.execute([torch.from_numpy(input0).bfloat16(), torch.ones(shape).bfloat16(), torch.zeros(shape).bfloat16()],
                     [torch.zeros(shape).bfloat16(), torch.zeros((1)).bfloat16(), torch.zeros((1)).bfloat16()])

    ### example for read from tensor bin
    # def test_layernorm_from_file_example(self):
    #     tensor_dir = "/home/zcj/tensor_data/NormOpsRunner0714/0_LayernormF160004Kernel"
    #     intensor0 = read_tensor(os.path.join(tensor_dir, "intensor0.bin"))
    #     intensor1 = read_tensor(os.path.join(tensor_dir, "intensor1.bin"))
    #     intensor2 = read_tensor(os.path.join(tensor_dir, "intensor2.bin"))
    #     outtensor0 = torch.zeros_like(intensor0)
    #     outtensor1 = torch.zeros_like(intensor1)
    #     outtensor2 = torch.zeros_like(intensor2)
    #     self.set_param(OP_NAME, OP_PARAM1)
    #     self.execute([intensor0, intensor1, intensor2], [outtensor0, outtensor1, outtensor2])


if __name__ == '__main__':
    unittest.main()
