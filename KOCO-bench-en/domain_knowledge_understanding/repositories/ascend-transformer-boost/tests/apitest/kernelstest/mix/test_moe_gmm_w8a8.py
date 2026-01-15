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

BLOCK_SIZE_16 = 16
BLOCK_SIZE_32 = 32


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


def round_up(val: int, align: int) -> int:
    if align == 0:
        return 0
    return -(val // -align) * align


def cvt_nd_to_nz(nd_mat: np.ndarray, block_size: tuple = (BLOCK_SIZE_16, BLOCK_SIZE_32)) -> np.ndarray:
    b = nd_mat.shape[0]
    r = round_up(nd_mat.shape[1], block_size[0])
    c = round_up(nd_mat.shape[2], block_size[1])
    r_pad = r - nd_mat.shape[1]
    c_pad = c - nd_mat.shape[2]
    nd_mat = np.pad(nd_mat, ((0, 0), (0, r_pad), (0, c_pad)))
    nz_mat = np.transpose(
        np.reshape(nd_mat, (b, r // block_size[0], block_size[0], c // block_size[1], block_size[1])), (0, 3, 1, 2, 4)
    )
    nz_mat = np.reshape(nz_mat, (nz_mat.shape[0], nz_mat.shape[1], nz_mat.shape[2] * nz_mat.shape[3], nz_mat.shape[4]))
    return nz_mat


class TestMoeGmmW8a8(op_test.OpTest):
    def __moe_gmm_up_golden(
        self,
        num_experts: int,
        num_tokens: int,
        hidden_size_in: int,
        hidden_size_out: int,
        top_k: int,
        dtype: torch.dtype,
    ) -> None:
        weight_nz = self.op_desc["input_formats"][1] == self.format_nz
        activation = np.random.randint(-5, 5, (num_tokens, hidden_size_in)).astype(np.int8)
        weight = np.random.randint(-5, 5, (num_experts, hidden_size_in, hidden_size_out)).astype(np.int8)
        experts_map = np.array([np.random.permutation(num_experts)[:top_k] for _ in range(num_tokens)])
        experts_index = [np.where(experts_map == i)[0].tolist() for i in range(num_experts)]
        experts_count = [len(each) for each in experts_index]
        nscale = np.random.uniform(-0.1, 0.1, size=(num_experts, hidden_size_out)).astype(np.float32)
        mscale = np.random.uniform(-0.1, 0.1, size=(num_tokens, 1)).astype(np.float32)

        attr = self.op_desc["specificParam"]
        self.ksize = hidden_size_in
        ref = []

        for e in range(num_experts):
            if len(experts_index[e]) == 0:
                continue
            a = activation[experts_index[e]]
            b = weight[e]
            c = torch.mm(torch.tensor(a).int(), torch.tensor(b).int()).numpy()
            ref.append(c * mscale[experts_index[e]] * nscale[e])

        ref = np.concatenate(ref)
        if attr["transposeB"]:
            weight = np.transpose(weight, (0, 2, 1))
        if weight_nz:
            weight = cvt_nd_to_nz(weight)
        self.activation = activation
        self.weight = weight
        self.experts_index = [item for sublist in experts_index for item in sublist]
        self.experts_count = experts_count
        self.ref = torch.tensor(ref).to(dtype)
        self.mscale = mscale
        self.nscale = nscale
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
        weight_nz = self.op_desc["input_formats"][1] == self.format_nz
        activation = np.random.randint(-5, 5, (num_tokens * top_k, hidden_size_in)).astype(np.int8)
        weight = np.random.randint(-5, 5, (num_experts, hidden_size_in, hidden_size_out)).astype(np.int8)
        experts_map = np.array([np.random.permutation(num_experts)[:top_k] for _ in range(num_tokens)])
        experts_index = [np.where(experts_map == i)[0].tolist() for i in range(num_experts)]
        experts_count = [len(each) for each in experts_index]
        ref = np.zeros([num_tokens, hidden_size_out]).astype(np.float32)
        nscale = np.random.uniform(-0.1, 0.1, size=(num_experts, hidden_size_out)).astype(np.float32)
        mscale = np.random.uniform(-0.1, 0.1, size=(num_tokens * top_k, 1)).astype(np.float32)
        op_param = self.op_desc["specificParam"]
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
        if weight_nz:
            weight = cvt_nd_to_nz(weight)
        self.activation = activation
        self.weight = weight
        self.experts_index = [item for sublist in experts_index for item in sublist]
        self.experts_count = experts_count
        self.mscale = mscale
        self.nscale = nscale
        self.ref = torch.tensor(ref).to(dtype)
        return

    def __moe_gmm_up_common(
        self, num_experts: int, num_tokens: int, hidden_size_in, hidden_size_out, top_k, dtype, trans_b, format_b
    ):
        self.set_param(
            "MoeGmmOperation",
            {
                "transposeB": trans_b,
                "topK": top_k,
                "moeGmmMode": 0,
                "moeGmmDequantType": int(dtype == torch.bfloat16),
                "hiddenSize": [hidden_size_in, hidden_size_out],
            },
        )
        self.set_input_formats(
            [self.format_nd, format_b, self.format_nd, self.format_nd, self.format_nd, self.format_nd]
        )
        self.set_output_formats([self.format_nd])
        self.__moe_gmm_up_golden(num_experts, num_tokens, hidden_size_in, hidden_size_out, top_k, dtype)
        self.execute(
            [
                torch.tensor(self.activation).char(),
                torch.tensor(self.weight).char(),
                torch.tensor(self.experts_count).int().cumsum(-1).int(),
                torch.tensor(self.experts_index).int(),
                torch.tensor(self.nscale).float(),
                torch.tensor(self.mscale).float(),
            ],
            [torch.zeros(self.ref.shape).to(dtype)],
        )
        return

    def __moe_gmm_down_common(
        self, num_experts, num_tokens, hidden_size_in, hidden_size_out, top_k, dtype, trans_b, format_b
    ):
        self.set_param(
            "MoeGmmOperation",
            {
                "transposeB": trans_b,
                "topK": top_k,
                "moeGmmMode": 1,
                "moeGmmDequantType": int(dtype == torch.bfloat16),
                "hiddenSize": [hidden_size_in, hidden_size_out],
            },
        )
        self.set_input_formats(
            [self.format_nd, format_b, self.format_nd, self.format_nd, self.format_nd, self.format_nd]
        )
        self.set_output_formats([self.format_nd])
        self.__moe_gmm_down_golden(num_experts, num_tokens, hidden_size_in, hidden_size_out, top_k, dtype)

        self.execute(
            [
                torch.tensor(self.activation).char(),
                torch.tensor(self.weight).char(),
                torch.tensor(self.experts_count).int().cumsum(-1).int(),
                torch.tensor(self.experts_index).int(),
                torch.tensor(self.nscale).float(),
                torch.tensor(self.mscale).float(),
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
    def testcase_moe_gmm_up_nd_weight(self):
        self.__moe_gmm_up_common(
            num_experts=160,
            num_tokens=128,
            hidden_size_in=5120,
            hidden_size_out=384,
            top_k=6,
            dtype=torch.float16,
            trans_b=True,
            format_b=self.format_nz,
        )
        return

    @op_test.only_910b
    def testcase_moe_gmm_down_nd_weight(self):
        self.__moe_gmm_down_common(
            num_experts=160,
            num_tokens=1024,
            hidden_size_in=192,
            hidden_size_out=5120,
            top_k=6,
            dtype=torch.float16,
            trans_b=True,
            format_b=self.format_nz,
        )
        return

    @op_test.only_910b
    def testcase_moe_gmm_up_nz_weight(self):
        self.__moe_gmm_up_common(
            num_experts=160,
            num_tokens=128,
            hidden_size_in=5120,
            hidden_size_out=384,
            top_k=6,
            dtype=torch.float16,
            trans_b=True,
            format_b=self.format_nz,
        )
        return

    @op_test.only_910b
    def testcase_moe_gmm_down_nz_weight(self):
        self.__moe_gmm_down_common(
            num_experts=160,
            num_tokens=1024,
            hidden_size_in=192,
            hidden_size_out=5120,
            top_k=6,
            dtype=torch.float16,
            trans_b=True,
            format_b=self.format_nz,
        )
        return

    @op_test.only_910b
    def testcase_moe_gmm_up_nd_weight_bf16(self):
        self.__moe_gmm_up_common(
            num_experts=160,
            num_tokens=128,
            hidden_size_in=5120,
            hidden_size_out=384,
            top_k=6,
            dtype=torch.bfloat16,
            trans_b=True,
            format_b=self.format_nz,
        )
        return

    @op_test.only_910b
    def testcase_moe_gmm_down_nd_weight_bf16(self):
        self.__moe_gmm_down_common(
            num_experts=160,
            num_tokens=1024,
            hidden_size_in=192,
            hidden_size_out=5120,
            top_k=6,
            dtype=torch.bfloat16,
            trans_b=True,
            format_b=self.format_nz,
        )
        return

    @op_test.only_910b
    def testcase_moe_gmm_up_nz_weight_bf16(self):
        self.__moe_gmm_up_common(
            num_experts=160,
            num_tokens=128,
            hidden_size_in=5120,
            hidden_size_out=384,
            top_k=6,
            dtype=torch.bfloat16,
            trans_b=True,
            format_b=self.format_nz,
        )
        return

    @op_test.only_910b
    def testcase_moe_gmm_down_nz_weight_bf16(self):
        self.__moe_gmm_down_common(
            num_experts=160,
            num_tokens=1024,
            hidden_size_in=192,
            hidden_size_out=5120,
            top_k=6,
            dtype=torch.bfloat16,
            trans_b=True,
            format_b=self.format_nz,
        )
        return


if __name__ == "__main__":
    unittest.main()
