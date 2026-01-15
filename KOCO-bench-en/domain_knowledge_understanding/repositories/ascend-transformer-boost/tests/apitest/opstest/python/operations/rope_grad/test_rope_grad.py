#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import torch
import torch_npu
import torch.nn as nn
import sys
import os
import numpy as np
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test

OP_NAME = "RopeGradOperation"
OP_PARAM = {"qSeqLen":None}

class TestRopeGradOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        # x,128*32-->reshape x,32,128
        cos_list = [in_tensors[2][:x, :] for x in OP_PARAM['qSeqLen']]
        sin_list = [in_tensors[3][:x, :] for x in OP_PARAM['qSeqLen']]
        cos = torch.cat(cos_list, dim=0)
        sin = torch.cat(sin_list, dim=0)
        sin1 = sin[:,:64]
        sin2 = sin[:,64:]
        rohqgsin = torch.cat((sin2, -sin1), dim=-1)
        q_grad = torch.zeros_like(in_tensors[0])
        bs = int(in_tensors[0].shape[1] / 128)
        for i in range(bs):
            q_grad[:, i * 128:(i + 1) * 128] = in_tensors[0][:, i * 128:(i + 1) * 128] * (cos + rohqgsin)
    
        k_grad = torch.zeros_like(in_tensors[1])
        for i in range(bs):
            k_grad[:,i * 128:(i + 1) * 128] = in_tensors[1][:, i * 128:(i + 1) * 128] *(cos + rohqgsin)
        return [q_grad, k_grad]

    def get_test_data(self, maxSeqLen, batch, hiddenSize, headSize):
        OP_PARAM['qSeqLen'] = np.random.randint(maxSeqLen, size=batch).tolist()
        sumSeqLen = sum(OP_PARAM['qSeqLen'])
        q = torch.rand((sumSeqLen, hiddenSize)).npu().half()
        k = torch.rand((sumSeqLen, hiddenSize)).npu().half()
        cos = torch.rand((maxSeqLen, headSize)).npu().half()
        sin = torch.rand((maxSeqLen, headSize)).npu().half()
        return [q,k,cos,sin]

    def test(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        maxSeqLen = 2048
        batch = 4
        hiddenSize = 1024
        headSize = 128
        input_tensor = self.get_test_data(maxSeqLen, batch, hiddenSize, headSize)
        self.execute(OP_NAME, OP_PARAM, input_tensor)
        batch = 8
        input_tensor = self.get_test_data(maxSeqLen, batch, hiddenSize, headSize)
        self.execute_update_param(OP_NAME, OP_PARAM, input_tensor)

if __name__ == '__main__':
    unittest.main()
