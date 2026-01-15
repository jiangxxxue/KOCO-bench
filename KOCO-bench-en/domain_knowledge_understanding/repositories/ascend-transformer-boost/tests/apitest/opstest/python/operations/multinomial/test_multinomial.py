#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
import torch.nn as nn
import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

b = 32000
rand = [0.840188, 0.394383, 0.783099, 0.79844, 0.911647, 0.197551, 0.335223, 0.76823,
        0.277775, 0.55397, 0.477397, 0.628871, 0.364784, 0.513401, 0.95223, 0.916195,
        0.635712, 0.717297, 0.141603, 0.606969, 0.0163006, 0.242887, 0.137232,
        0.804177, 0.156679, 0.400944, 0.12979, 0.108809, 0.998924, 0.218257,
        0.512932, 0.839112, 0.61264, 0.296032, 0.637552, 0.524287, 0.493583,
        0.972775, 0.292517, 0.771358, 0.526745, 0.769914, 0.400229, 0.891529, 0.283315,
        0.352458, 0.807725, 0.919026, 0.0697553, 0.949327, 0.525995, 0.0860558, 0.192214,
        0.663227, 0.890233, 0.348893, 0.0641713, 0.020023, 0.457702,
        0.0630958, 0.23828, 0.970634, 0.902208, 0.85092]

OP_NAME = "MultinomialOperation"
OP_PARAM0 = {"numSamples": 2}


def multinomial_compute(input0, samples):
    rope_q = np.zeros(shape=(input0.shape[0], input0.shape[1])).astype(np.float16)
    rope_k = np.zeros(shape=(input0.shape[0], samples)).astype(np.int32)
    for i in range(0, input0.shape[0]):
        rope_q[i][0] = input0[i][0]

    for j in range(0, input0.shape[0]):
        for i in range(1, input0.shape[1]):
            rope_q[j][i] = rope_q[j][i - 1] + input0[j][i]

    for z in range(0, samples):
        for j in range(0, input0.shape[0]):
            for i in range(0, input0.shape[1]):
                if (rope_q[j][i] >= rand[z]):
                    rope_k[j][z] = i
                    break
    return torch.from_numpy(rope_k)


class TestMulti(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        numSamples = OP_PARAM0["numSamples"]
        result = multinomial_compute(np.array(in_tensors[0].cpu()), numSamples)
        return [result]

    def golden_compare(self, out_tensors, golden_out_tensors):
        return np.allclose(out_tensors[0].cpu(), golden_out_tensors[0].cpu(), rtol=0, atol=1)

    def test_layenorm_1d_large(self):
        if not operation_test.get_soc_version() == 'Ascend910B':
            print("this testcase only supports Ascend910B")
            return
        numSamples = 2
        tensor_dir = "."
        probs = torch.randn(8, b).float()
        sm = nn.Softmax(dim=-1)
        input_softmax = sm(probs).half().npu()
        self.execute_out(OP_NAME, OP_PARAM0, [input_softmax], [torch.zeros(8, numSamples).int().npu()])

if __name__ == '__main__':
    unittest.main()
