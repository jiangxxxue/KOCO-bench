#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# AscendOpCommonLib is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#
import unittest
import numpy as np
import torch
import sys
import op_test
import logging
import torch_npu
import torch.nn.functional as F
import os

DRANGE = (-5, 5)


def round_up(val: int, align: int) -> int:
    if align == 0:
        return 0
    return -(val // -align) * align


def transdata(nd_mat, block_size: tuple = (16, 16)):
    r = round_up(nd_mat.shape[0], block_size[0])
    c = round_up(nd_mat.shape[1], block_size[1])
    r_pad = r - nd_mat.shape[0]
    c_pad = c - nd_mat.shape[1]
    nd_mat = F.pad(nd_mat, ((0, c_pad, 0, r_pad)))
    nz_mat = torch.permute(
        torch.reshape(nd_mat, (r // block_size[0], block_size[0], c // block_size[1], block_size[1])), [2, 0, 1, 3]
    )
    nz_mat = torch.reshape(nz_mat, (nz_mat.shape[0], nz_mat.shape[1] * nz_mat.shape[2], nz_mat.shape[3]))
    return nz_mat


def get_eb(golden: torch.Tensor, actual: torch.Tensor):
    golden_nmax = torch.clamp(torch.abs(golden), min=1)
    actual_error = actual.to(torch.float32) - golden
    EB = torch.mean(actual_error / golden_nmax)
    result = EB <= 2 ** (-7)
    return result


def ref_compare(golden: torch.Tensor, actual: torch.Tensor, thresh: float):
    golden_nmax = torch.clamp(torch.abs(golden), min=1)
    abs_error = torch.abs(actual.to(torch.float32) - golden)
    result = (abs_error <= thresh * golden_nmax).all()
    return result


class TestPpMatmulEinSumFp16(op_test.OpTest):
    def __gen_test_data(self, shape: tuple, dtype: torch.dtype = torch.float16) -> None:
        bsize, msize, ksize, nsize = shape
        bat_A, bat_B, bat_C = [], [], []
        op_param = self.op_desc["specificParam"]
        input_formats = self.op_desc["input_formats"]
        for _ in range(bsize):
            a = torch.rand(size=(msize, ksize), dtype=dtype) * (DRANGE[1] - DRANGE[0]) + DRANGE[0]
            b = torch.rand(size=(ksize, nsize), dtype=dtype) * (DRANGE[1] - DRANGE[0]) + DRANGE[0]
            c = torch.mm(a.float(), b.float())
            if op_param["transposeB"]:
                b.transpose_(1, 0)
            if input_formats[1] == self.format_nz:
                b = transdata(b)
            bat_A.append(a)
            bat_B.append(b)
            bat_C.append(c)
        self.bat_A = torch.stack(bat_A).permute((1, 0, 2))
        self.bat_B = torch.stack(bat_B)
        self.bat_C = torch.stack(bat_C).permute((1, 0, 2))
        return

    def golden_calc(self, in_tensors):
        return [self.bat_C]

    def golden_compare(self, out_tensors, golden_out_tensors):
        if "specificParam" in self.op_desc.keys():
            logging.debug(str(self.op_desc["specificParam"]))
        ksize = self.op_desc["specificParam"]["oriShape"][1]
        eb = get_eb(golden_out_tensors[0], out_tensors[0])
        err_thod = 2**-8 if ksize < 2048 else 2**-7
        cmp = ref_compare(golden_out_tensors[0], out_tensors[0], err_thod)
        return eb and cmp

    @op_test.only_910b
    def testcase0(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 5, 16, 345, 657
        dtype = torch.float16
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "matmulType": 4,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), dtype)
        self.execute(
            [self.bat_A, self.bat_B],
            [torch.zeros(self.bat_C.shape).to(dtype)],
        )

    @op_test.only_910b
    def testcase1(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 32, 64, 128, 512
        dtype = torch.float16
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "matmulType": 4,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), dtype)
        self.execute(
            [self.bat_A, self.bat_B],
            [torch.zeros(self.bat_C.shape).to(dtype)],
        )

    @op_test.only_910b
    def testcase2(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 32, 64, 512, 128
        dtype = torch.float16
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "matmulType": 4,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), dtype)
        self.execute(
            [self.bat_A, self.bat_B],
            [torch.zeros(self.bat_C.shape).to(dtype)],
        )

    @op_test.only_910b
    def testcase3(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 5, 16, 345, 657
        dtype = torch.bfloat16
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "matmulType": 4,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), dtype)
        self.execute(
            [self.bat_A, self.bat_B],
            [torch.zeros(self.bat_C.shape).to(dtype)],
        )

    @op_test.only_910b
    def testcase4(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 32, 64, 128, 512
        dtype = torch.bfloat16
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "matmulType": 4,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), dtype)
        self.execute(
            [self.bat_A, self.bat_B],
            [torch.zeros(self.bat_C.shape).to(dtype)],
        )

    @op_test.only_910b
    def testcase5(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 32, 64, 512, 128
        dtype = torch.bfloat16
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "matmulType": 4,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), dtype)
        self.execute(
            [self.bat_A, self.bat_B],
            [torch.zeros(self.bat_C.shape).to(dtype)],
        )

    @op_test.only_910b
    def testcase6(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 32, 256, 456, 789
        dtype = torch.float16
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "matmulType": 4,
                "enShuffleK": True,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), dtype)
        self.execute(
            [self.bat_A, self.bat_B],
            [torch.zeros(self.bat_C.shape).to(dtype)],
        )

    @op_test.only_910b
    def testcase7(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 32, 256, 512, 128
        dtype = torch.float16
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "matmulType": 4,
                "enShuffleK": True,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), dtype)
        self.execute(
            [self.bat_A, self.bat_B],
            [torch.zeros(self.bat_C.shape).to(dtype)],
        )

    @op_test.only_910b
    def testcase8(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 32, 256, 456, 789
        dtype = torch.float16
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "matmulType": 4,
                "enShuffleK": True,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), dtype)
        self.execute(
            [self.bat_A, self.bat_B],
            [torch.zeros(self.bat_C.shape).to(dtype)],
        )

    @op_test.only_910b
    def testcase_einsum_gemv(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 32, 1, 456, 789
        dtype = torch.float16
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "matmulType": 4,
                "enShuffleK": True,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), dtype)
        self.execute(
            [self.bat_A, self.bat_B],
            [torch.zeros(self.bat_C.shape).to(dtype)],
        )


if __name__ == "__main__":
    unittest.main()
