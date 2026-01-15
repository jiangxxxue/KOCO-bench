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
import torch
import logging
import numpy as np
import op_test
from typing import List, Tuple

BLOCK_SIZE = 16
DRANGE = (-2, 2)


def ceil_div(dividend: int, divisor: int) -> int:
    if divisor == 0:
        return int("inf")
    return -(dividend // -divisor)


def round_up(val: int, align: int) -> int:
    if align == 0:
        return 0
    return -(val // -align) * align


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


def cvt_nd_to_nz(nd_mat: torch.Tensor, block_size: tuple = (BLOCK_SIZE, BLOCK_SIZE)) -> torch.Tensor:
    nd_mat = nd_mat.float().numpy()
    r = round_up(nd_mat.shape[0], block_size[0])
    c = round_up(nd_mat.shape[1], block_size[1])
    r_pad = r - nd_mat.shape[0]
    c_pad = c - nd_mat.shape[1]
    nd_mat = np.pad(nd_mat, ((0, r_pad), (0, c_pad)))
    nz_mat = np.transpose(
        np.reshape(nd_mat, (r // block_size[0], block_size[0], c // block_size[1], block_size[1])), (2, 0, 1, 3)
    )
    nz_mat = np.reshape(nz_mat, (nz_mat.shape[0], nz_mat.shape[1] * nz_mat.shape[2], nz_mat.shape[3]))
    return torch.tensor(nz_mat).bfloat16()


class TestPpMatmulBf16(op_test.OpTest):
    def __gen_test_data(self, shpae: Tuple[int], trans_A: bool, trans_B: bool) -> None:
        bsize, msize, ksize, nsize = shpae
        bat_A, bat_B, bat_C = [], [], []
        input_formats = self.op_desc["input_formats"]
        for _ in range(bsize):
            a = torch.rand(size=(msize, ksize)).bfloat16() * (DRANGE[1] - DRANGE[0]) + DRANGE[0]
            b = torch.rand(size=(ksize, nsize)).bfloat16() * (DRANGE[1] - DRANGE[0]) + DRANGE[0]
            c = torch.mm(a.float(), b.float()).bfloat16()
            if trans_A:
                a.transpose_(1, 0)
            if trans_B:
                b.transpose_(1, 0)
            if input_formats[0] == self.format_nz:
                a = cvt_nd_to_nz(a)
            if input_formats[1] == self.format_nz:
                b = cvt_nd_to_nz(b)
            bat_A.append(a)
            bat_B.append(b)
            bat_C.append(c)
        self.bat_A = torch.stack(bat_A)
        self.bat_B = torch.stack(bat_B)
        self.bat_C = torch.stack(bat_C)
        return

    def golden_calc(self, in_tensors: List[torch.Tensor]):
        return [self.bat_C]

    def golden_compare(self, out_tensors: List[torch.Tensor], golden_out_tensors: List[torch.Tensor]):
        if "specificParam" in self.op_desc.keys():
            logging.debug(str(self.op_desc["specificParam"]))
        ksize = self.op_desc["specificParam"]["oriShape"][1]
        eb = get_eb(golden_out_tensors[0], out_tensors[0])
        cmp = ref_compare(golden_out_tensors[0], out_tensors[0], 2**-7 if ksize < 2048 else 2**-6)
        return eb and cmp

    @op_test.only_910b
    def testcase0(self):
        bsize, msize, ksize, nsize = 1, 28, 8192, 3072
        for trans_A in [False, True]:
            for trans_B in [False, True]:
                self.set_param(
                    "MatMulOperation",
                    {"transposeA": trans_A, "transposeB": trans_B, "oriShape": [msize, ksize, nsize]},
                )
                self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
                self.set_output_formats([self.format_nd])
                self.__gen_test_data((bsize, msize, ksize, nsize), trans_A, trans_B)
                self.execute(
                    [self.bat_A, self.bat_B],
                    [torch.zeros(self.bat_C.shape).bfloat16()],
                )

    @op_test.only_910b
    def testcase1(self):
        bsize, msize, ksize, nsize = 1, 129, 1024, 8192
        for trans_A in [False, True]:
            for trans_B in [False, True]:
                self.set_param(
                    "MatMulOperation",
                    {"transposeA": trans_A, "transposeB": trans_B, "oriShape": [msize, ksize, nsize]},
                )
                self.set_input_formats([self.format_nd, self.format_nd])
                self.set_output_formats([self.format_nd])
                self.__gen_test_data((bsize, msize, ksize, nsize), trans_A, trans_B)
                self.execute(
                    [self.bat_A, self.bat_B],
                    [torch.zeros(self.bat_C.shape).bfloat16()],
                )

    @op_test.only_910b
    def testcase2(self):
        bsize, msize, ksize, nsize = 1, 1, 1024, 8192
        for trans_A in [False, True]:
            for trans_B in [False, True]:
                self.set_param(
                    "MatMulOperation",
                    {"transposeA": trans_A, "transposeB": trans_B, "oriShape": [msize, ksize, nsize]},
                )
                self.set_input_formats([self.format_nd, self.format_nd])
                self.set_output_formats([self.format_nd])
                self.__gen_test_data((bsize, msize, ksize, nsize), trans_A, trans_B)
                self.execute(
                    [self.bat_A, self.bat_B],
                    [torch.zeros(self.bat_C.shape).bfloat16()],
                )

    @op_test.only_910b
    def testcase3(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 60, 12288, 1792
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), self.trans_A, self.trans_B)
        self.execute(
            [self.bat_A, self.bat_B],
            [torch.zeros(self.bat_C.shape).bfloat16()],
        )

    @op_test.only_910b
    def testcase4(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 60, 1536, 12288
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), self.trans_A, self.trans_B)
        self.execute(
            [self.bat_A, self.bat_B],
            [torch.zeros(self.bat_C.shape).bfloat16()],
        )

    @op_test.only_910b
    def testcase5(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 60, 12288, 7808
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), self.trans_A, self.trans_B)
        self.execute(
            [self.bat_A, self.bat_B],
            [torch.zeros(self.bat_C.shape).bfloat16()],
        )

    @op_test.only_910b
    def testcase6(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 60, 3904, 12288
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), self.trans_A, self.trans_B)
        self.execute(
            [self.bat_A, self.bat_B],
            [torch.zeros(self.bat_C.shape).bfloat16()],
        )

    @op_test.only_910b
    def testcase7(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 80, 1536, 12288
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), self.trans_A, self.trans_B)
        self.execute(
            [self.bat_A, self.bat_B],
            [torch.zeros(self.bat_C.shape).bfloat16()],
        )

    @op_test.only_910b
    def testcase8(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 80, 1536, 12288
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), self.trans_A, self.trans_B)
        self.execute(
            [self.bat_A, self.bat_B],
            [torch.zeros(self.bat_C.shape).bfloat16()],
        )

    @op_test.only_910b
    def testcase9(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 80, 12288, 7808
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), self.trans_A, self.trans_B)
        self.execute(
            [self.bat_A, self.bat_B],
            [torch.zeros(self.bat_C.shape).bfloat16()],
        )

    @op_test.only_910b
    def testcase10(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 80, 3904, 12288
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), self.trans_A, self.trans_B)
        self.execute(
            [self.bat_A, self.bat_B],
            [torch.zeros(self.bat_C.shape).bfloat16()],
        )

    @op_test.only_910b
    def testcase11(self):
        bsize, msize, ksize, nsize = 1, 28, 8192, 3072
        for trans_A in [False, True]:
            for trans_B in [False, True]:
                self.set_param(
                    "MatMulOperation",
                    {"transposeA": trans_A, "transposeB": trans_B, "oriShape": [msize, ksize, nsize]},
                )
                self.set_input_formats([self.format_nd, self.format_nz])
                self.set_output_formats([self.format_nd])
                self.__gen_test_data((bsize, msize, ksize, nsize), trans_A, trans_B)
                self.execute(
                    [self.bat_A, self.bat_B],
                    [torch.zeros(self.bat_C.shape).bfloat16()],
                )

    @op_test.only_910b
    def testcase12(self):
        bsize, msize, ksize, nsize = 1, 129, 1024, 8192
        for trans_A in [False, True]:
            for trans_B in [False, True]:
                self.set_param(
                    "MatMulOperation",
                    {"transposeA": trans_A, "transposeB": trans_B, "oriShape": [msize, ksize, nsize]},
                )
                self.set_input_formats([self.format_nd, self.format_nz])
                self.set_output_formats([self.format_nd])
                self.__gen_test_data((bsize, msize, ksize, nsize), trans_A, trans_B)
                self.execute(
                    [self.bat_A, self.bat_B],
                    [torch.zeros(self.bat_C.shape).bfloat16()],
                )

    @op_test.only_910b
    def testcase13(self):
        bsize, msize, ksize, nsize = 1, 1, 1024, 8192
        for trans_A in [False, True]:
            for trans_B in [False, True]:
                self.set_param(
                    "MatMulOperation",
                    {"transposeA": trans_A, "transposeB": trans_B, "oriShape": [msize, ksize, nsize]},
                )
                self.set_input_formats([self.format_nd, self.format_nz])
                self.set_output_formats([self.format_nd])
                self.__gen_test_data((bsize, msize, ksize, nsize), trans_A, trans_B)
                self.execute(
                    [self.bat_A, self.bat_B],
                    [torch.zeros(self.bat_C.shape).bfloat16()],
                )


if __name__ == "__main__":
    unittest.main()
