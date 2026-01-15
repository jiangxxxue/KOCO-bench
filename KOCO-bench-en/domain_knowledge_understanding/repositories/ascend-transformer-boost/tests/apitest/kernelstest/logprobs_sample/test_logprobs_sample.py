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

import op_test

OP_NAME = "LogprobsSampleOperation"
OP_PARAM = {"logprobsSize": 0}

class TestLogprobsSample(op_test.OpTest):
    def golden_calc(self, in_tensors):
        sorted_probs = in_tensors[0].to(torch.float32).numpy()
        cumsumed_probs = in_tensors[1].to(torch.float32).numpy()
        select_range = in_tensors[2].to(torch.int32).numpy().reshape(-1, 1)
        logprobs_size = OP_PARAM["logprobsSize"]

        select_index = (select_range - 1)
        select_index[select_index < 0] = 0
        select_prob = np.take_along_axis(cumsumed_probs, select_index, axis=-1).astype(np.float32)

        logprobs_output = sorted_probs[:, :logprobs_size]
        logprobs_output = np.log(np.divide(logprobs_output, select_prob))
        batch = sorted_probs.shape[0]
        increasing_index = np.tile(np.arange(logprobs_size), batch).reshape(batch, -1)
        mask = ~torch.tensor(increasing_index <= select_index)
        logprobs_output = torch.tensor(logprobs_output).masked_fill(mask=mask, value=-9999.0)
        return [logprobs_output.float()]


    def golden_compare(self, out_tensors, golden_out_tensors):
        return torch.allclose(out_tensors[0], golden_out_tensors[0], atol=0.001, rtol=0.001)

    def execute_case(self, batch, logprobs_size, input_size, input_data_type):
        OP_PARAM["logprobsSize"] = logprobs_size
        self.set_param(OP_NAME, OP_PARAM)

        output_shape = (batch, logprobs_size)
        probs = torch.randn(batch, input_size).float()
        sm = torch.nn.Softmax(dim=-1)
        probs = sm(probs)
        sorted_probs, _ = torch.sort(probs, descending=True)
        cumsumed_probs = torch.cumsum(sorted_probs, dim=-1).to(input_data_type)
        sorted_probs = sorted_probs.to(input_data_type)

        select_range = torch.from_numpy(
            np.random.uniform(0, min(logprobs_size * 2, input_size), [batch]).astype(np.int32))
        self.execute([sorted_probs, cumsumed_probs, select_range],
                     [torch.zeros(output_shape).float()])

    @op_test.skip_310b
    @op_test.skip_910a
    def test_logprobs_unpad_1_half(self):
        batch = 512
        logprobs_size = 7
        input_size = 16384
        self.execute_case(batch, logprobs_size, input_size, torch.half)

    @op_test.skip_310b
    @op_test.skip_910a
    def test_logprobs_unpad_2_half(self):
        batch = 10
        logprobs_size = 17
        input_size = 65536
        self.execute_case(batch, logprobs_size, input_size, torch.half)

    @op_test.skip_310b
    @op_test.skip_910a
    def test_logprobs_pad_half(self):
        batch = 3
        logprobs_size = 8
        input_size = 92367
        self.execute_case(batch, logprobs_size, input_size, torch.half)

    @op_test.skip_310b
    @op_test.skip_910a
    def test_logprobs_size_min_half(self):
        batch = 1
        logprobs_size = 1
        input_size = 16384
        self.execute_case(batch, logprobs_size, input_size, torch.half)

    @op_test.skip_310b
    @op_test.skip_910a
    def test_logprobs_size_max_half(self):
        batch = 512
        logprobs_size = 16384
        input_size = 16384
        self.execute_case(batch, logprobs_size, input_size, torch.half)

    @op_test.skip_310b
    @op_test.skip_910a
    def test_logprobs_size_0(self):
        batch = 512
        logprobs_size = 0
        input_size = 16384
        self.execute_case(batch, logprobs_size, input_size, torch.half)

    @op_test.only_910b
    def test_logprobs_unpad_1_bf16(self):
        batch = 511
        logprobs_size = 15
        input_size = 16384
        self.execute_case(batch, logprobs_size, input_size, torch.bfloat16)

    @op_test.only_910b
    def test_logprobs_unpad_2_bf16(self):
        batch = 10
        logprobs_size = 16383
        input_size = 65536
        self.execute_case(batch, logprobs_size, input_size, torch.bfloat16)

    @op_test.only_910b
    def test_logprobs_pad_bf16(self):
        batch = 5
        logprobs_size = 16
        input_size = 32
        self.execute_case(batch, logprobs_size, input_size, torch.bfloat16)

    @op_test.only_910b
    def test_logprobs_size_min_bf16(self):
        batch = 1
        logprobs_size = 1
        input_size = 100
        self.execute_case(batch, logprobs_size, input_size, torch.bfloat16)

    @op_test.only_910b
    def test_logprobs_size_max_bf16(self):
        batch = 512
        logprobs_size = 16384
        input_size = 32768
        self.execute_case(batch, logprobs_size, input_size, torch.bfloat16)



if __name__ == '__main__':
    unittest.main()
