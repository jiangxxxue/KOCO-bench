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
import numpy as np
import torch
import op_test
import logging

BLOCK_SIZE = 16


def ceil_div(dividend: int, divisor: int) -> int:
    if divisor == 0:
        return int("inf")
    return -(dividend // -divisor)


def round_up(val: int, align: int) -> int:
    if align == 0:
        return 0
    return -(val // -align) * align


def cvt_nd_to_nz(nd_mat: np.ndarray, block_size: tuple = (BLOCK_SIZE, BLOCK_SIZE)) -> np.ndarray:
    r = round_up(nd_mat.shape[0], block_size[0])
    c = round_up(nd_mat.shape[1], block_size[1])
    r_pad = r - nd_mat.shape[0]
    c_pad = c - nd_mat.shape[1]
    nd_mat = np.pad(nd_mat, ((0, r_pad), (0, c_pad)))
    nz_mat = np.transpose(
        np.reshape(nd_mat, (r // block_size[0], block_size[0], c // block_size[1], block_size[1])), (2, 0, 1, 3)
    )
    nz_mat = np.reshape(nz_mat, (nz_mat.shape[0], nz_mat.shape[1] * nz_mat.shape[2], nz_mat.shape[3]))
    return nz_mat


class TestPpMatmulF16(op_test.OpTest):
    def __gen_test_data(self, shpae: tuple) -> None:
        bsize, msize, ksize, nsize = shpae
        bat_A, bat_B, bat_C = [], [], []
        op_param = self.op_desc["specificParam"]
        input_formats = self.op_desc["input_formats"]
        for _ in range(bsize):
            a = np.random.uniform(-2, 2, size=(msize, ksize)).astype(np.float16)
            b = np.random.uniform(-2, 2, size=(ksize, nsize)).astype(np.float16)
            c = np.dot(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)
            if op_param["transposeA"]:
                a = np.transpose(a, (1, 0))
            if op_param["transposeB"]:
                b = np.transpose(b, (1, 0))
            if input_formats[0] == self.format_nz:
                a = cvt_nd_to_nz(a)
            if input_formats[1] == self.format_nz:
                b = cvt_nd_to_nz(b)
            bat_A.append(a)
            bat_B.append(b)
            bat_C.append(c)
        self.bat_A = np.stack(bat_A)
        self.bat_B = np.stack(bat_B)
        self.bat_C = np.stack(bat_C)
        return

    def golden_calc(self, in_tensors):
        return [torch.tensor(self.bat_C).half()]

    def golden_compare(self, out_tensors, golden_out_tensors):
        if "specificParam" in self.op_desc.keys():
            logging.debug(str(self.op_desc["specificParam"]))
        return torch.allclose(out_tensors[0].half(), golden_out_tensors[0].half(), rtol=0.001, atol=0.001)

    @op_test.only_910b
    def testcase0(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 1, 8192, 1024
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase1(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 1, 456, 789
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase2(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 1, 2752, 1024
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase3(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 5, 1, 2752, 1024
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )
    @op_test.only_910b
    def testcase4(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 1, 1, 8192, 1024
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase5(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 1, 1, 456, 789
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase6(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 1, 1, 2752, 1024
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase7(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 5, 1, 2752, 1024
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase8(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 1, 4, 2752, 1024
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase9(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 5, 16, 2752, 1024
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase10(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 5, 16, 345, 657
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase11(self):
        self.trans_A, self.trans_B = True, True
        bsize, msize, ksize, nsize = 5, 16, 345, 657
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase12(self):
        self.trans_A, self.trans_B = True, False
        bsize, msize, ksize, nsize = 5, 16, 345, 657
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase13(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 2, 14336, 250880
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase14(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 129, 1024, 8192
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase15(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 1, 204800, 80, 8192
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase16(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 401692, 96, 80
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase17(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 143962, 64, 64
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase18(self):
        self.trans_A, self.trans_B = True, False
        bsize, msize, ksize, nsize = 3, 96, 160, 96
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nz])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase_llama65b_1(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 167, 8192, 3072
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase_llama65b_2(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 167, 8192, 1024
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase_llama65b_3(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 167, 8192, 5504
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase_llama65b_4(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 167, 8192, 2752
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase_llama65b_5(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 167, 2752, 8192
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase_llama65b_6(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 167, 8192, 32000
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )

if __name__ == "__main__":
    unittest.main()
