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

OP_NAME = "GenAttentionMaskOperation"
OP_PARAM = {"headNum": 0, "seqLen": None}

class TestElewiseSubOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        out = []
        for i, s in enumerate(OP_PARAM['seqLen']):
            for j in range(OP_PARAM["headNum"]):
                out.append(in_tensors[0][i, :, :s, :s].flatten())
        return [torch.hstack(out)]

    def test_2d_half(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        OP_PARAM["headNum"] = 2
        OP_PARAM["seqLen"] = [2, 1]
        batch = 2
        maxseqlen = 4
        input = torch.randint(1, 10, (batch, 1, maxseqlen, maxseqlen)).npu().half()
        input_tensor = [input]
        self.execute(OP_NAME, OP_PARAM, input_tensor)
        OP_PARAM["seqLen"] = [2, 2]
        self.execute_update_param(OP_NAME, OP_PARAM, input_tensor)

if __name__ == '__main__':
    unittest.main()