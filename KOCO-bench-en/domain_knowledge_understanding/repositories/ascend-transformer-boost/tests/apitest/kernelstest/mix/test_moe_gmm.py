# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import unittest
import numpy as np
import torch
import sys
import os

sys.path.append("../")
sys.path.append("../..")
import op_test
import logging


def get_eb(golden: torch.Tensor, actual: torch.Tensor):
    golden = golden.to(torch.float32)
    golden_nmax = torch.clamp(torch.abs(golden), min=1)
    actual_error = actual.to(torch.float32) - golden
    EB = torch.mean(actual_error / golden_nmax)
    result = EB <= 2 ** (-7)
    return result


def ref_compare(golden: torch.Tensor, actual: torch.Tensor, thresh: float):
    golden = golden.to(torch.float32)
    golden_nmax = torch.clamp(torch.abs(golden), min=1)
    abs_error = torch.abs(actual.to(torch.float32) - golden)
    result = (abs_error <= thresh * golden_nmax).all()
    return result


class TestMoeGmm(op_test.OpTest):
    def __moe_gmm_up_golden(
        self,
        num_experts: int,
        num_tokens: int,
        hidden_size_in: int,
        hidden_size_out: int,
        top_k: int,
        dtype: torch.dtype,
    ) -> None:
        activation = np.random.uniform(-5, 5, (num_tokens, hidden_size_in))
        weight = np.random.uniform(-5, 5, (num_experts, hidden_size_in, hidden_size_out))
        experts_map = np.array([np.random.permutation(num_experts)[:top_k] for _ in range(num_tokens)])
        experts_index = [np.where(experts_map == i)[0].tolist() for i in range(num_experts)]
        experts_count = [len(each) for each in experts_index]
        self.ksize = hidden_size_in
        attr = self.op_desc["specificParam"]
        ref = []
        for e in range(num_experts):
            if len(experts_index[e]) == 0:
                continue
            a = activation[experts_index[e]]  #
            b = weight[e]
            c = torch.mm(torch.tensor(a).to(dtype).float(), torch.tensor(b).to(dtype).float()).numpy()
            ref.append(c)
        ref = np.concatenate(ref)
        if attr["transposeB"]:
            weight = np.transpose(weight, (0, 2, 1))
        self.activation = activation
        self.weight = weight
        self.experts_index = [item for sublist in experts_index for item in sublist]
        self.experts_count = experts_count
        self.ref = torch.tensor(ref).to(dtype)
        return

    def __moe_gmm_down_golden(
        self,
        num_experts: int,
        num_tokens: int,
        hidden_size_in: int,
        hidden_size_out: int,
        top_k: int,
        dtype: torch.dtype,
    ) -> None:
        activation = np.random.uniform(-5, 5, (num_tokens * top_k, hidden_size_in))
        weight = np.random.uniform(-5, 5, (num_experts, hidden_size_in, hidden_size_out))
        experts_map = np.array([np.random.permutation(num_experts)[:top_k] for _ in range(num_tokens)])
        experts_index = [np.where(experts_map == i)[0].tolist() for i in range(num_experts)]
        experts_count = [len(each) for each in experts_index]
        ref = np.zeros([num_tokens, hidden_size_out]).astype(np.float32)
        self.ksize = hidden_size_in
        attr = self.op_desc["specificParam"]

        e_start = 0
        for e in range(num_experts):
            if len(experts_index[e]) == 0:
                continue
            a = activation[e_start : e_start + experts_count[e]]
            b = weight[e]
            c = torch.mm(torch.tensor(a).to(dtype).float(), torch.tensor(b).to(dtype).float()).numpy()
            ref[experts_index[e]] += c
            e_start += experts_count[e]

        if attr["transposeB"]:
            weight = np.transpose(weight, (0, 2, 1))
        self.activation = activation
        self.weight = weight
        self.experts_index = [item for sublist in experts_index for item in sublist]
        self.experts_count = experts_count
        self.ref = torch.tensor(ref).to(dtype)
        return

    def __moe_gmm_up_common(self, num_experts, num_tokens, hidden_size_in, hidden_size_out, top_k, dtype, trans_b):
        self.set_param(
            "MoeGmmOperation",
            {
                "transposeB": trans_b,
                "moeGmmMode": 0,
                "topK": top_k,
                "hiddenSize": [hidden_size_in, hidden_size_out],
            },
        )

        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])

        self.__moe_gmm_up_golden(num_experts, num_tokens, hidden_size_in, hidden_size_out, top_k, dtype)
        self.execute(
            [
                torch.tensor(self.activation).to(dtype),
                torch.tensor(self.weight).to(dtype),
                torch.tensor(self.experts_count).int().cumsum(-1).int(),
                torch.tensor(self.experts_index).int(),
            ],
            [torch.zeros(self.ref.shape).to(dtype)],
        )
        return

    def __moe_gmm_down_common(self, num_experts, num_tokens, hidden_size_in, hidden_size_out, top_k, dtype, trans_b):
        self.set_param(
            "MoeGmmOperation",
            {
                "transposeB": trans_b,
                "moeGmmMode": 1,
                "topK": top_k,
                "hiddenSize": [hidden_size_in, hidden_size_out],
            },
        )

        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])

        self.__moe_gmm_down_golden(num_experts, num_tokens, hidden_size_in, hidden_size_out, top_k, dtype)
        self.execute(
            [
                torch.tensor(self.activation).to(dtype),
                torch.tensor(self.weight).to(dtype),
                torch.tensor(self.experts_count).int().cumsum(-1).int(),
                torch.tensor(self.experts_index).int(),
            ],
            [torch.zeros(self.ref.shape).to(dtype)],
        )
        return

    def golden_calc(self, in_tensors):
        return [self.ref]

    def golden_compare(self, out_tensors, golden_out_tensors):
        if "specificParam" in self.op_desc.keys():
            logging.debug(str(self.op_desc["specificParam"]))
        eb = get_eb(golden_out_tensors[0], out_tensors[0])
        cmp = ref_compare(golden_out_tensors[0], out_tensors[0], 2**-7 if self.ksize < 2048 else 2**-6)
        return eb and cmp

    @op_test.only_910b
    def testcase_moe_gmm_up_fp16(self):
        self.__moe_gmm_up_common(
            num_experts=160,
            num_tokens=1024,
            hidden_size_in=5120,
            hidden_size_out=192,
            top_k=6,
            dtype=torch.float16,
            trans_b=False,
        )
        return

    @op_test.only_910b
    def testcase_moe_gmm_up_fp16_trans_weight(self):
        self.__moe_gmm_up_common(
            num_experts=160,
            num_tokens=1024,
            hidden_size_in=5120,
            hidden_size_out=192,
            top_k=6,
            dtype=torch.float16,
            trans_b=True,
        )
        return

    @op_test.only_910b
    def testcase_moe_gmm_up_bf16(self):
        self.__moe_gmm_up_common(
            num_experts=160,
            num_tokens=1024,
            hidden_size_in=5120,
            hidden_size_out=192,
            top_k=6,
            dtype=torch.bfloat16,
            trans_b=False,
        )
        return

    @op_test.only_910b
    def testcase_moe_gmm_up_bf16_trans_weight(self):
        self.__moe_gmm_up_common(
            num_experts=160,
            num_tokens=1024,
            hidden_size_in=5120,
            hidden_size_out=192,
            top_k=6,
            dtype=torch.bfloat16,
            trans_b=True,
        )
        return

    @op_test.only_910b
    def testcase_moe_gmm_down_fp16(self):
        self.__moe_gmm_down_common(
            num_experts=160,
            num_tokens=1024,
            hidden_size_in=96,
            hidden_size_out=5120,
            top_k=6,
            dtype=torch.float16,
            trans_b=False,
        )
        return

    @op_test.only_910b
    def testcase_moe_gmm_down_fp16_trans_weight(self):
        self.__moe_gmm_down_common(
            num_experts=160,
            num_tokens=1024,
            hidden_size_in=96,
            hidden_size_out=5120,
            top_k=6,
            dtype=torch.float16,
            trans_b=True,
        )
        return

    @op_test.only_910b
    def testcase_moe_gmm_down_bf16(self):
        self.__moe_gmm_down_common(
            num_experts=160,
            num_tokens=1024,
            hidden_size_in=96,
            hidden_size_out=5120,
            top_k=6,
            dtype=torch.bfloat16,
            trans_b=False,
        )
        return

    @op_test.only_910b
    def testcase_moe_gmm_down_bf16_trans_weight(self):
        self.__moe_gmm_down_common(
            num_experts=160,
            num_tokens=1024,
            hidden_size_in=96,
            hidden_size_out=5120,
            top_k=6,
            dtype=torch.bfloat16,
            trans_b=True,
        )
        return


if __name__ == "__main__":
    unittest.main()
