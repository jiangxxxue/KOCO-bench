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
import logging
import op_test

BLOCK_SIZE_16 = 16
BLOCK_SIZE_32 = 32


def ceil_div(dividend: int, divisor: int) -> int:
    if divisor == 0:
        return int("inf")
    return -(dividend // -divisor)


def round_up(val: int, align: int) -> int:
    if align == 0:
        return 0
    return -(val // -align) * align


def cvt_nd_to_nz(nd_mat: np.ndarray, block_size: tuple = (BLOCK_SIZE_16, BLOCK_SIZE_32)) -> np.ndarray:
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


def golden_func(mat_a: np.ndarray, mat_b: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    ret = np.matmul(mat_a.astype(np.float32), mat_b.astype(np.float32)).astype(np.int32)
    ret = ret + bias
    ret = ret.astype(np.float32) * scale.astype(np.float32)
    return ret.astype(np.float16)


class TestPpMatmul910aDequant(op_test.OpTest):
    def __gen_test_data(self, shpae: tuple) -> None:
        np.random.seed(0)
        bsize, msize, ksize, nsize = shpae
        bat_A, bat_B, bat_C, scale_n, bias_n, bat_pertoken_descale = [], [], [], [], [], []
        for _ in range(bsize):
            a = np.random.randint(-2, 2, size=(msize, ksize)).astype(np.int8)
            b = np.random.randint(-2, 2, size=(ksize, nsize)).astype(np.int8)
            scale = np.random.rand(nsize).astype(np.float32) / 1000
            bias = np.random.randint(-1, 1, size=(1, nsize)).astype(np.int32)
            ret = golden_func(a, b, scale, bias)
            if self.trans_A:
                a = np.transpose(a, (1, 0))
            if self.trans_B:
                b = np.transpose(b, (1, 0))
            a = cvt_nd_to_nz(a, (BLOCK_SIZE_16, BLOCK_SIZE_32))
            b = cvt_nd_to_nz(b, (BLOCK_SIZE_16, BLOCK_SIZE_32))
            ret = cvt_nd_to_nz(ret, (BLOCK_SIZE_16, BLOCK_SIZE_16))
            scale_n.append(scale)

            bias_n.append(bias)
            bat_A.append(a)
            bat_B.append(b)
            bat_C.append(ret)

        self.scale = np.stack(scale_n)
        self.bias = np.stack(bias_n)
        self.bat_A = np.stack(bat_A)
        self.bat_B = np.stack(bat_B)
        self.bat_C = np.stack(bat_C)
        pertoken_descale = np.empty(msize)
        bat_pertoken_descale.append(pertoken_descale)
        self.bat_pertoken_descale = np.stack(bat_pertoken_descale)
        return

    def golden_calc(self, in_tensors):
        return [torch.tensor(self.bat_C)]

    def golden_compare(self, out_tensors, golden_out_tensors):
        if "specificParam" in self.op_desc.keys():
            logging.debug(str(self.op_desc["specificParam"]))
        logging.debug(f"out_tensors.half():{out_tensors[0].to(torch.half)}")
        logging.debug(f"golden_out_tensors.half():{golden_out_tensors[0].to(torch.half)}")
        return torch.allclose(
            out_tensors[0].to(torch.half), golden_out_tensors[0].to(torch.half), rtol=0.002, atol=0.002
        )

    @op_test.only_910a
    def testcase0(self):
        self.trans_A, self.trans_B, self.enDequant = False, True, True
        case_list = [[1, 32, 8192, 2560]]
        for bsize, msize, ksize, nsize in case_list:
            logging.debug(f"ppmatmul int8 nz bsize:{bsize}, msize:{msize}, ksize:{ksize}, nsize:{nsize}")
            self.set_param(
                "MatMulOperation",
                {
                    "transposeA": self.trans_A,
                    "transposeB": self.trans_B,
                    "oriShape": [msize, ksize, nsize],
                    "enDequant": self.enDequant,
                },
            )
            self.set_input_formats([self.format_nz, self.format_nz, self.format_nd, self.format_nd, self.format_nd])
            self.set_output_formats([self.format_nz])
            self.__gen_test_data((bsize, msize, ksize, nsize))
            logging.debug(self.bat_A.shape)
            self.execute(
                [
                    torch.tensor(self.bat_A).to(torch.int8),
                    torch.tensor(self.bat_B).to(torch.int8),
                    torch.tensor(self.bias).to(torch.int32),
                    torch.tensor(self.scale).to(torch.float32),
                    torch.tensor(self.bat_pertoken_descale, dtype=torch.float)
                ],
                [torch.zeros(self.bat_C.shape).to(torch.half)],
            )

    @op_test.only_910a
    def testcase1(self):
        self.trans_A, self.trans_B, self.enDequant = False, True, True
        bsize, msize, ksize, nsize = 1, 32, 1024, 2048
        logging.debug(f"ppmatmul int8 nz bsize:{bsize}, msize:{msize}, ksize:{ksize}, nsize:{nsize}")
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "enDequant": self.enDequant,
            },
        )
        self.set_input_formats([self.format_nz, self.format_nz, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        logging.debug(self.bat_A.shape)
        self.execute(
            [
                torch.tensor(self.bat_A).to(torch.int8),
                torch.tensor(self.bat_B).to(torch.int8),
                torch.tensor(self.bias).to(torch.int32),
                torch.tensor(self.scale).float(),
                torch.tensor(self.bat_pertoken_descale, dtype=torch.float)
            ],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910a
    def testcase2(self):
        self.trans_A, self.trans_B, self.enDequant = False, True, True
        bsize, msize, ksize, nsize = 1, 892, 128, 128
        logging.debug(f"ppmatmul int8 nz bsize:{bsize}, msize:{msize}, ksize:{ksize}, nsize:{nsize}")
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "enDequant": self.enDequant,
            },
        )
        self.set_input_formats([self.format_nz, self.format_nz, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        logging.debug(self.bat_A.shape)
        self.execute(
            [
                torch.tensor(self.bat_A).to(torch.int8),
                torch.tensor(self.bat_B).to(torch.int8),
                torch.tensor(self.bias).to(torch.int32),
                torch.tensor(self.scale).float(),
                torch.tensor(self.bat_pertoken_descale, dtype=torch.float)
            ],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910a
    def testcase3(self):
        self.trans_A, self.trans_B, self.enDequant = False, True, True
        bsize, msize, ksize, nsize = 1, 12, 16384, 128
        logging.debug(f"ppmatmul int8 nz bsize:{bsize}, msize:{msize}, ksize:{ksize}, nsize:{nsize}")
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "enDequant": self.enDequant,
            },
        )
        self.set_input_formats([self.format_nz, self.format_nz, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        logging.debug(self.bat_A.shape)
        self.execute(
            [
                torch.tensor(self.bat_A).to(torch.int8),
                torch.tensor(self.bat_B).to(torch.int8),
                torch.tensor(self.bias).to(torch.int32),
                torch.tensor(self.scale).float(),
                torch.tensor(self.bat_pertoken_descale, dtype=torch.float)
            ],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910a
    def testcase4(self):
        self.trans_A, self.trans_B, self.enDequant = False, True, True
        bsize, msize, ksize, nsize = 1, 16, 5120, 640
        logging.debug(f"ppmatmul int8 nz bsize:{bsize}, msize:{msize}, ksize:{ksize}, nsize:{nsize}")
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "enDequant": self.enDequant,
            },
        )
        self.set_input_formats([self.format_nz, self.format_nz, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        logging.debug(self.bat_A.shape)
        self.execute(
            [
                torch.tensor(self.bat_A).to(torch.int8),
                torch.tensor(self.bat_B).to(torch.int8),
                torch.tensor(self.bias).to(torch.int32),
                torch.tensor(self.scale).float(),
                torch.tensor(self.bat_pertoken_descale, dtype=torch.float)
            ],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910a
    def testcase5(self):
        self.trans_A, self.trans_B, self.enDequant = False, True, True
        bsize, msize, ksize, nsize = 2, 2, 64, 32
        logging.debug(f"ppmatmul int8 nz bsize:{bsize}, msize:{msize}, ksize:{ksize}, nsize:{nsize}")
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "enDequant": self.enDequant,
            },
        )
        self.set_input_formats([self.format_nz, self.format_nz, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        logging.debug(self.bat_A.shape)
        self.execute(
            [
                torch.tensor(self.bat_A).to(torch.int8),
                torch.tensor(self.bat_B).to(torch.int8),
                torch.tensor(self.bias).to(torch.int32),
                torch.tensor(self.scale).float(),
                torch.tensor(self.bat_pertoken_descale, dtype=torch.float)
            ],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910a
    def testcase6(self):
        self.trans_A, self.trans_B, self.enDequant = False, True, True
        bsize, msize, ksize, nsize = 2, 1, 64, 32
        logging.debug(f"ppmatmul int8 nz bsize:{bsize}, msize:{msize}, ksize:{ksize}, nsize:{nsize}")
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "enDequant": self.enDequant,
            },
        )
        self.set_input_formats([self.format_nz, self.format_nz, self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nz])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        logging.debug(self.bat_A.shape)
        self.execute(
            [
                torch.tensor(self.bat_A).to(torch.int8),
                torch.tensor(self.bat_B).to(torch.int8),
                torch.tensor(self.bias).to(torch.int32),
                torch.tensor(self.scale).float(),
                torch.tensor(self.bat_pertoken_descale, dtype=torch.float)
            ],
            [torch.zeros(self.bat_C.shape).half()],
        )


if __name__ == "__main__":
    unittest.main()
