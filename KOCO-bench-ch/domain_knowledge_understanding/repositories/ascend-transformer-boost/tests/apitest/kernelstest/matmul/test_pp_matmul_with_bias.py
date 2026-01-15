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

MATMUL_WITH_BIAS = 3
DRANGE = (-5, 5)


def get_eb(golden: torch.Tensor, actual: torch.Tensor, thresh: float = 2 ** (-7)):
    golden = golden.to(torch.float32)
    golden_nmax = torch.clamp(torch.abs(golden), min=1)
    actual_error = actual.to(torch.float32) - golden
    EB = torch.mean(actual_error / golden_nmax)
    result = EB <= thresh
    return result


def ref_compare(golden: torch.Tensor, actual: torch.Tensor, thresh: float):
    golden = golden.to(torch.float32)
    golden_nmax = torch.clamp(torch.abs(golden), min=1)
    abs_error = torch.abs(actual.to(torch.float32) - golden)
    result = (abs_error <= thresh * golden_nmax).all()
    return result


class TestPpMatmulF16(op_test.OpTest):
    def __gen_test_data(self, shpae: tuple, dtype=torch.float16) -> None:
        bsize, msize, ksize, nsize = shpae
        bat_A, bat_B, bat_bias, bat_C = [], [], [], []
        op_param = self.op_desc["specificParam"]
        for _ in range(bsize):
            a = torch.rand(size=(msize, ksize)).to(dtype) * (DRANGE[1] - DRANGE[0]) + DRANGE[0]
            b = torch.rand(size=(ksize, nsize)).to(dtype) * (DRANGE[1] - DRANGE[0]) + DRANGE[0]
            bias = torch.rand(size=(1, nsize)).float() * (DRANGE[1] - DRANGE[0]) + DRANGE[0]
            c = (torch.mm(a.float(), b.float()) + bias).to(dtype)
            if op_param["transposeA"]:
                a.transpose_(1, 0)
            if op_param["transposeB"]:
                b.transpose_(1, 0)
            bat_A.append(a)
            bat_B.append(b)
            bat_bias.append(bias)
            bat_C.append(c)
        self.bat_A = torch.stack(bat_A)
        self.bat_B = torch.stack(bat_B)
        self.bat_bias = torch.stack(bat_bias)
        self.bat_C = torch.stack(bat_C)
        return

    def golden_calc(self, in_tensors):
        return [self.bat_C]

    def golden_compare(self, out_tensors, golden_out_tensors):
        if "specificParam" in self.op_desc.keys():
            logging.debug(str(self.op_desc["specificParam"]))
        ksize = self.op_desc["specificParam"]["oriShape"][1]
        err_thod = 2**-8 if ksize < 2048 else 2**-7
        eb_thod = 2**-10
        if out_tensors[0].dtype == torch.bfloat16:
            err_thod = 2**-7 if ksize < 2048 else 2**-6
            eb_thod = 2**-7
        eb = get_eb(golden_out_tensors[0], out_tensors[0], eb_thod)
        cmp = ref_compare(golden_out_tensors[0], out_tensors[0], err_thod)
        return eb and cmp

    @op_test.only_910b
    def testcase_matmul_with_bias_fp16_nn(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 1, 167, 3072, 1024
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "matmulType": MATMUL_WITH_BIAS,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [self.bat_A.half(), self.bat_B.half(), self.bat_bias.float()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase_matmul_with_bias_fp16_nt(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 167, 3072, 1024
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "matmulType": MATMUL_WITH_BIAS,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [self.bat_A.half(), self.bat_B.half(), self.bat_bias.float()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase_matmul_with_bias_fp16_tn(self):
        self.trans_A, self.trans_B = True, False
        bsize, msize, ksize, nsize = 1, 167, 3072, 1024
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "matmulType": MATMUL_WITH_BIAS,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [self.bat_A.half(), self.bat_B.half(), self.bat_bias.float()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase_matmul_with_bias_fp16_tt(self):
        self.trans_A, self.trans_B = True, True
        bsize, msize, ksize, nsize = 1, 167, 3072, 1024
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "matmulType": MATMUL_WITH_BIAS,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [self.bat_A.half(), self.bat_B.half(), self.bat_bias.float()],
            [torch.zeros(self.bat_C.shape).half()],
        )

    @op_test.only_910b
    def testcase_matmul_with_bias_bf16_nn(self):
        self.trans_A, self.trans_B = False, False
        bsize, msize, ksize, nsize = 1, 167, 3072, 1024
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "matmulType": MATMUL_WITH_BIAS,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), dtype=torch.bfloat16)
        self.execute(
            [self.bat_A.bfloat16(), self.bat_B.bfloat16(), self.bat_bias.float()],
            [torch.zeros(self.bat_C.shape).bfloat16()],
        )

    @op_test.only_910b
    def testcase_matmul_with_bias_bf16_nt(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 167, 3072, 1024
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "matmulType": MATMUL_WITH_BIAS,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), dtype=torch.bfloat16)
        self.execute(
            [self.bat_A.bfloat16(), self.bat_B.bfloat16(), self.bat_bias.float()],
            [torch.zeros(self.bat_C.shape).bfloat16()],
        )

    @op_test.only_910b
    def testcase_matmul_with_bias_bf16_tn(self):
        self.trans_A, self.trans_B = True, False
        bsize, msize, ksize, nsize = 1, 167, 3072, 1024
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "matmulType": MATMUL_WITH_BIAS,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), dtype=torch.bfloat16)
        self.execute(
            [self.bat_A.bfloat16(), self.bat_B.bfloat16(), self.bat_bias.float()],
            [torch.zeros(self.bat_C.shape).bfloat16()],
        )

    @op_test.only_910b
    def testcase_matmul_with_bias_bf16_tt(self):
        self.trans_A, self.trans_B = True, True
        bsize, msize, ksize, nsize = 1, 167, 3072, 1024
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "matmulType": MATMUL_WITH_BIAS,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), dtype=torch.bfloat16)
        self.execute(
            [self.bat_A.bfloat16(), self.bat_B.bfloat16(), self.bat_bias.float()],
            [torch.zeros(self.bat_C.shape).bfloat16()],
        )

    @op_test.only_910b
    def testcase_matmul_with_bias_bf16_gemv(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 1, 3072, 1024
        self.set_param(
            "MatMulOperation",
            {
                "transposeA": self.trans_A,
                "transposeB": self.trans_B,
                "oriShape": [msize, ksize, nsize],
                "matmulType": MATMUL_WITH_BIAS,
            },
        )
        self.set_input_formats([self.format_nd, self.format_nd, self.format_nd])
        self.set_output_formats([self.format_nd])
        self.__gen_test_data((bsize, msize, ksize, nsize), dtype=torch.bfloat16)
        self.execute(
            [self.bat_A.bfloat16(), self.bat_B.bfloat16(), self.bat_bias.float()],
            [torch.zeros(self.bat_C.shape).bfloat16()],
        )


if __name__ == "__main__":
    unittest.main()
