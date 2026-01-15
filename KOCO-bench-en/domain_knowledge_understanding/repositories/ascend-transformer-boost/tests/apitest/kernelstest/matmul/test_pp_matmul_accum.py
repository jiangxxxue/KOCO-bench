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
DRANGE = (-5, 5)
MATMUL_ACCUM_ATOMIC = 2


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


class TestPpMatmulAccum(op_test.OpTest):
    def __gen_test_data(self, shape, trans_A: bool, trans_B: bool, in_dtype=torch.float16) -> None:
        bsize, msize, ksize, nsize = shape
        bat_A, bat_B, bat_C, bat_ref = [], [], [], []
        for _ in range(bsize):
            a = torch.rand(size=(msize, ksize)).to(in_dtype) * (DRANGE[1] - DRANGE[0]) + DRANGE[0]
            b = torch.rand(size=(ksize, nsize)).to(in_dtype) * (DRANGE[1] - DRANGE[0]) + DRANGE[0]
            c = torch.rand(size=(msize, nsize)).float() * (DRANGE[1] - DRANGE[0]) + DRANGE[0]
            ref = torch.mm(a.float(), b.float()).float() + c
            if trans_A:
                a.transpose_(1, 0)
            if trans_B:
                b.transpose_(1, 0)
            bat_A.append(a)
            bat_B.append(b)
            bat_C.append(c)
            bat_ref.append(ref)
        self.bat_A = torch.stack(bat_A)
        self.bat_B = torch.stack(bat_B)
        self.bat_C = torch.stack(bat_C)
        self.bat_ref = torch.stack(bat_ref)
        return

    def golden_calc(self, in_tensors: List[torch.Tensor]):
        return [self.bat_ref]

    def golden_compare(self, out_tensors: List[torch.Tensor], golden_out_tensors: List[torch.Tensor]):
        if "specificParam" in self.op_desc.keys():
            logging.debug(str(self.op_desc["specificParam"]))
        ksize = self.op_desc["specificParam"]["oriShape"][1]
        eb = get_eb(golden_out_tensors[0], out_tensors[0])
        err_thod = 2**-8 if ksize < 2048 else 2**-7
        cmp = ref_compare(golden_out_tensors[0], out_tensors[0], err_thod)
        return eb and cmp

    @op_test.only_910b
    def testcase_matmul_accum_atomic_bf16_nn(self):
        bsize, msize, ksize, nsize = 1, 28, 8192, 3072
        ta, tb = False, False
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": ta,
                "transposeB": tb,
                "oriShape": [msize, ksize, nsize],
                "matmulType": MATMUL_ACCUM_ATOMIC,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), ta, tb, torch.bfloat16)
        self.execute(
            [self.bat_A.bfloat16(), self.bat_B.bfloat16(), self.bat_C.float()],
            [2],
        )

    @op_test.only_910b
    def testcase_matmul_accum_atomic_bf16_nt(self):
        bsize, msize, ksize, nsize = 1, 28, 8192, 3072
        ta, tb = False, True
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": ta,
                "transposeB": tb,
                "oriShape": [msize, ksize, nsize],
                "matmulType": MATMUL_ACCUM_ATOMIC,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), ta, tb, torch.bfloat16)
        self.execute(
            [self.bat_A.bfloat16(), self.bat_B.bfloat16(), self.bat_C.float()],
            [2],
        )

    @op_test.only_910b
    def testcase_matmul_accum_atomic_bf16_tn(self):
        bsize, msize, ksize, nsize = 1, 28, 8192, 3072
        ta, tb = True, False
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": ta,
                "transposeB": tb,
                "oriShape": [msize, ksize, nsize],
                "matmulType": MATMUL_ACCUM_ATOMIC,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), ta, tb, torch.bfloat16)
        self.execute(
            [self.bat_A.bfloat16(), self.bat_B.bfloat16(), self.bat_C.float()],
            [2],
        )

    @op_test.only_910b
    def testcase_matmul_accum_atomic_bf16_tt(self):
        bsize, msize, ksize, nsize = 1, 28, 8192, 3072
        ta, tb = True, True
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": ta,
                "transposeB": tb,
                "oriShape": [msize, ksize, nsize],
                "matmulType": MATMUL_ACCUM_ATOMIC,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), ta, tb, torch.bfloat16)
        self.execute(
            [self.bat_A.bfloat16(), self.bat_B.bfloat16(), self.bat_C.float()],
            [2],
        )

    @op_test.only_910b
    def testcase_matmul_accum_atomic_fp16_nn(self):
        bsize, msize, ksize, nsize = 1, 28, 8192, 3072
        ta, tb = False, False
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": ta,
                "transposeB": tb,
                "oriShape": [msize, ksize, nsize],
                "matmulType": MATMUL_ACCUM_ATOMIC,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), ta, tb, torch.float16)
        self.execute(
            [self.bat_A.half(), self.bat_B.half(), self.bat_C.float()],
            [2],
        )

    @op_test.only_910b
    def testcase_matmul_accum_atomic_fp16_nt(self):
        bsize, msize, ksize, nsize = 1, 28, 8192, 3072
        ta, tb = False, True
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": ta,
                "transposeB": tb,
                "oriShape": [msize, ksize, nsize],
                "matmulType": MATMUL_ACCUM_ATOMIC,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), ta, tb, torch.float16)
        self.execute(
            [self.bat_A.half(), self.bat_B.half(), self.bat_C.float()],
            [2],
        )

    @op_test.only_910b
    def testcase_matmul_accum_atomic_fp16_tn(self):
        bsize, msize, ksize, nsize = 1, 28, 8192, 3072
        ta, tb = True, False
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": ta,
                "transposeB": tb,
                "oriShape": [msize, ksize, nsize],
                "matmulType": MATMUL_ACCUM_ATOMIC,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), ta, tb, torch.float16)
        self.execute(
            [self.bat_A.half(), self.bat_B.half(), self.bat_C.float()],
            [2],
        )

    @op_test.only_910b
    def testcase_matmul_accum_atomic_fp16_tt(self):
        bsize, msize, ksize, nsize = 1, 28, 8192, 3072
        ta, tb = True, True
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": ta,
                "transposeB": tb,
                "oriShape": [msize, ksize, nsize],
                "matmulType": MATMUL_ACCUM_ATOMIC,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), ta, tb, torch.float16)
        self.execute(
            [self.bat_A.half(), self.bat_B.half(), self.bat_C.float()],
            [2],
        )

    @op_test.only_910b
    def testcase_matmul_accum_atomic_fp16_gemv(self):
        bsize, msize, ksize, nsize = 1, 1, 8192, 3072
        ta, tb = False, True
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": ta,
                "transposeB": tb,
                "oriShape": [msize, ksize, nsize],
                "matmulType": MATMUL_ACCUM_ATOMIC,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), ta, tb, torch.float16)
        self.execute(
            [self.bat_A.half(), self.bat_B.half(), self.bat_C.float()],
            [2],
        )


if __name__ == "__main__":
    unittest.main()
