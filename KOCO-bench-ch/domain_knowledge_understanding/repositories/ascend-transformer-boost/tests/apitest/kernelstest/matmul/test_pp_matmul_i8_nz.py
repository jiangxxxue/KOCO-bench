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
import logging
import op_test

def ceil_div(dividend: int, divisor: int) -> int:
    if divisor == 0:
        return int("inf")
    return -(dividend // -divisor)


def round_up(val: int, align: int) -> int:
    if align == 0:
        return 0
    return -(val // -align) * align

def process_deq_scale(deq_scale) -> np.ndarray:
    new_deq_scale = np.frombuffer(deq_scale.tobytes(), dtype=np.uint32)
    return new_deq_scale.astype(np.int64)

def cvt_nd_to_nz(nd_mat: np.ndarray, block_size: tuple = (16, 32)) -> np.ndarray:
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

def process_deq_scale(deq_scale) -> np.ndarray:
    new_deq_scale = np.frombuffer(deq_scale.tobytes(), dtype=np.uint32)
    return new_deq_scale.astype(np.int64)

class TestPpMatmulI8(op_test.OpTest):
    def __gen_test_data(self, shpae: tuple) -> None:
        bsize, msize, ksize, nsize = shpae
        bat_A, bat_B, bat_C, bat_scale, bat_bias, bat_pertoken_descale = [], [], [], [], [], []
        for _ in range(bsize):
            a = np.random.uniform(-2, 2, size=(msize, ksize)).astype(np.int8)
            b = np.random.uniform(-2, 2, size=(ksize, nsize)).astype(np.int8)
            bias = np.random.randint(-10, 10, (1, nsize)).astype(np.int32)
            # tmp = np.random.randint(1, 5, (1, nsize)).astype(np.float32)
            tmp = np.random.rand(nsize).astype(np.float32) / 1000
            c = np.dot(a.astype(np.float32), b.astype(np.float32)).astype(np.int32)
            if self.bias_flag:
                c = c + bias
            c = c.astype(np.float32) * tmp
            c = c.astype(np.float16)
            if self.trans_A:
                a = np.transpose(a, (1, 0))
            if self.trans_B:
                b = np.transpose(b, (1, 0))
            if self.NZ:
                b = cvt_nd_to_nz(b, (16, 32))
            scale = process_deq_scale(tmp)
            bat_A.append(a)
            bat_B.append(b)
            bat_bias.append(bias)
            bat_scale.append(scale)
            bat_C.append(c)
        self.bat_A = np.stack(bat_A)
        self.bat_B = np.stack(bat_B)
        self.bat_C = np.stack(bat_C)
        self.bat_scale = np.stack(bat_scale)
        self.bat_bias = np.stack(bat_bias)
        pertoken_descale = np.empty(msize)
        bat_pertoken_descale.append(pertoken_descale)
        self.bat_pertoken_descale = np.stack(bat_pertoken_descale)
        return

    def golden_calc(self, in_tensors):
        return [torch.tensor(self.bat_C).half()]

    def golden_compare(self, out_tensors, golden_out_tensors):
        if "specificParam" in self.op_desc.keys():
            logging.debug(str(self.op_desc["specificParam"]))
        curesult = torch.allclose(out_tensors[0], golden_out_tensors[0], rtol=0.002, atol=0.002)
        logging.debug("SUCCESS" if curesult else "FAIL!!")
        return curesult

    @op_test.only_310b
    def testcase_nobiascase1(self):
        transFlag = [[False, False], [False, True], [True, False], [True, True]]
        for trans in transFlag:
            self.trans_A, self.trans_B = trans[0], trans[1]
            self.bias_flag = False
            self.NZ = True
            bsize, msize, ksize, nsize = 1, 128, 2560, 5120
            self.set_param("MatMulOperation",
                           {"transposeA": self.trans_A,
                            "transposeB": self.trans_B,
                            "withBias": self.bias_flag,
                            "oriShape": [msize, ksize, nsize],
                            "enDequant": True})
            self.set_input_formats([self.format_nd,
                                    self.format_nz,
                                    self.format_nd,
                                    self.format_nd,
                                    self.format_nd])
            self.set_output_formats([self.format_nd])
            self.__gen_test_data((bsize, msize, ksize, nsize))
            self.execute([torch.tensor(self.bat_A, dtype=torch.int8),
                          torch.tensor(self.bat_B, dtype=torch.int8),
                          torch.tensor(self.bat_bias, dtype=torch.int32),
                          torch.tensor(self.bat_scale, dtype=torch.int64),
                          torch.tensor(self.bat_pertoken_descale, dtype=torch.float)],
                         [torch.zeros(self.bat_C.shape, dtype=torch.float16)])


if __name__ == '__main__':
    unittest.main()
