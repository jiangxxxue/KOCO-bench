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
import os
import unittest
import numpy as np
import torch
import logging
import op_test
import math

BLOCK_SIZE = 16


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


class TestPpMatmul310p(op_test.OpTest):
    def __gen_test_data(self, shpae: tuple) -> None:
        bsize, msize, ksize, nsize = shpae
        bat_A, bat_B, bat_C = [], [], []
        op_param = self.op_desc["specificParam"]
        for _ in range(bsize):
            a = np.random.uniform(-2, 2, size=(msize, ksize)).astype(np.float16)
            b = np.random.uniform(-2, 2, size=(ksize, nsize)).astype(np.float16)
            c = np.dot(a.astype(np.float32), b.astype(np.float32)).astype(np.float16)
            if op_param["transposeA"]:
                a = np.transpose(a, (1, 0))
            if op_param["transposeB"]:
                b = np.transpose(b, (1, 0))
            bat_A.append(cvt_nd_to_nz(a))
            bat_B.append(cvt_nd_to_nz(b))
            bat_C.append(cvt_nd_to_nz(c))
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

    @op_test.only_310p
    def testcase0(self):
        self.trans_A, self.trans_B = False, True
        bsize, msize, ksize, nsize = 1, 512, 512, 512
        self.set_param(
            "MatMulOperation",
            {"transposeA": self.trans_A, "transposeB": self.trans_B, "oriShape": [msize, ksize, nsize]},
        )
        self.set_input_formats([self.format_nz, self.format_nz])
        self.set_output_formats([self.format_nz])
        self.__gen_test_data((bsize, msize, ksize, nsize))
        self.execute(
            [torch.tensor(self.bat_A).half(), torch.tensor(self.bat_B).half()],
            [torch.zeros(self.bat_C.shape).half()],
        )


if __name__ == "__main__":
    unittest.main()
