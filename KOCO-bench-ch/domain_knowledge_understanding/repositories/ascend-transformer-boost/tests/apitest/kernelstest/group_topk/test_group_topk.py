#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import logging
import unittest
import torch
import numpy as np

import op_test

OP_NAME = "GroupTopkOperation"

class TestGroupTopk(op_test.OpTest):
    def golden_calc(self, in_tensors):
        group_num: int = self.op_desc["specificParam"]["groupNum"]
        k: int = self.op_desc["specificParam"]["k"]
        kInner: int = self.op_desc["specificParam"]["kInner"]
        input0 = in_tensors[0]
        token_num, expert_num = input0.shape
        input0 = torch.reshape(input0, (token_num, group_num, expert_num // group_num))
        output = input0.clone()
        input0 = input0.to(torch.float)
        group_tensor = torch.topk(input0, kInner).values
        group_tensor = torch.sum(group_tensor, dim=-1)
        # The torch version of the CI is too old. Not support the stable parameter in torch.argsort.
        sort_index = torch.from_numpy(np.argsort(-group_tensor.numpy(), kind='stable'))
        cols_to_use = torch.arange(k, group_num, dtype=torch.long)
        row_indices = torch.arange(sort_index.shape[0]).repeat_interleave(cols_to_use.shape[0])
        col_indices = sort_index.index_select(1, cols_to_use).view(-1)
        output[row_indices, col_indices] = 0
        return [torch.reshape(output, (token_num, expert_num))]

    def golden_compare(self, out_tensors, golden_out_tensors):
        res = torch.equal(out_tensors[0], golden_out_tensors[0])
        return res

    def generalize_param(self):
        token_nums = [1, 16, 33]
        expert_nums = [1, 4, 15, 16, 17, 32, 65, 128, 257, 1024]
        k_params = [1, 4, 15, 16, 17, 32, 65, 257, 1024]
        k_inner_params = [1, 2, 3, 4, 16, 32, 65, 1024]

        # find all factors of the expert_num
        expert_num_groups = []
        for expert_num in expert_nums:
            factors = set()
            for i in range(1, int(expert_num**0.5) + 1):
                if expert_num % i == 0:
                    factors.add(i)
                    factors.add(int(expert_num / i))
            expert_num_groups.append(list(factors))

        for token_num in token_nums:
            for expert_num, group_nums in zip(expert_nums, expert_num_groups):
                for group_num in group_nums:
                    for k in k_params:
                        for k_inner in k_inner_params:
                            if k > group_num or k_inner > expert_num // group_num:
                                continue
                            yield token_num, expert_num, group_num, k, k_inner

    @op_test.only_910b
    def test_generalize_fp16(self):
        for token_num, expert_num, group_num, k, k_inner in self.generalize_param():
            logging.debug(f'token_num={token_num} expert_num={expert_num} group={group_num} k={k} k_inner={k_inner}')
            self.do_execute(token_num, expert_num, group_num, k, k_inner, torch.float16)

    @op_test.only_910b
    def test_generalize_bf16(self):
        for token_num, expert_num, group_num, k, k_inner in self.generalize_param():
            logging.debug(f'token_num={token_num} expert_num={expert_num} group={group_num} k={k} k_inner={k_inner}')
            self.do_execute(token_num, expert_num, group_num, k, k_inner, torch.bfloat16)

    @op_test.only_910b
    def test_fp16_512_160_8_3_2(self):
        # typical case from DeepseekV3
        self.do_execute(512, 256, 8, 3, 2, torch.float16)

    @op_test.only_910b
    def test_bf16_512_160_8_3_2(self):
        # typical case from DeepseekV3
        self.do_execute(512, 256, 8, 3, 2, torch.bfloat16)

    def do_execute(self, token_num: int, expert_num: int, group_num: int, k: int, k_inner: int, dtype: torch.dtype):
        input0 = torch.empty((token_num, expert_num), dtype=dtype).uniform_(-2, 2)
        input1 = torch.arange(1024, dtype=torch.int32)
        self.set_param(OP_NAME, {"groupNum": group_num, "k": k, "kInner": k_inner})
        self.execute([input0, input1], [0])

if __name__ == '__main__':
    unittest.main()
