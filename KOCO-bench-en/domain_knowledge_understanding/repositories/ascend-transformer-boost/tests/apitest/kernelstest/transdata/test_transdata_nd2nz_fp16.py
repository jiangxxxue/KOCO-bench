#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import os
import unittest
import math
import numpy as np
import torch
import sys
sys.path.append('../')
import op_test
from tensor_file import read_tensor
import subprocess
import logging

OP_NAME = "TransdataOperation"
OP_PARAM0 = {"transdataType": 2}

BLOCK_SIZE_16 = 16

def RoundUp(val: int, align: int) -> int:
    if (align == 0):
        return 0
    return -(val // -align) * align

def TransformNdToNz(nd_mat: np.ndarray, block_size: tuple = (BLOCK_SIZE_16, BLOCK_SIZE_16)) -> np.ndarray:
    r = RoundUp(nd_mat.shape[1], block_size[0])
    c = RoundUp(nd_mat.shape[2], block_size[1])
    r_pad = r - nd_mat.shape[1]
    c_pad = c - nd_mat.shape[2]
    nd_mat = np.pad(nd_mat, ((0, 0), (0, r_pad), (0, c_pad)))
    ndz_mat = np.transpose(np.reshape(nd_mat, (nd_mat.shape[0], r // block_size[0], block_size[0], c // block_size[1], block_size[1])), (0, 3, 1, 2 ,4))
    nz_mat = np.reshape(ndz_mat, (ndz_mat.shape[0], c // block_size[1], -1, block_size[1]))
    return nz_mat

class TestTransdataOperation(op_test.OpTest):
    def golden_calc(self, in_tensors):
        input0 = in_tensors[0].cpu().numpy()
        output0 = TransformNdToNz(input0, (BLOCK_SIZE_16, BLOCK_SIZE_16))
        return [torch.tensor(output0)]

    def golden_compare(self, out_tensors, golden_out_tensors):
        logging.debug(out_tensors[0].shape)
        logging.debug(golden_out_tensors[0].shape)
        return torch.allclose(out_tensors[0], golden_out_tensors[0], rtol=0.001, atol=0.001)

    def test_transdata_fp16_nd2nz_case1(self):
        shape1 = (3, 4096, 1024)
        inputNd = np.random.uniform(low=0, high=1, size=shape1).astype(np.float16)
        outputNz = TransformNdToNz(inputNd, (BLOCK_SIZE_16, BLOCK_SIZE_16))
        self.set_param(OP_NAME, OP_PARAM0)
        self.set_input_formats([self.format_nd])
        self.set_output_formats([self.format_nz])
        self.execute([torch.tensor(inputNd).half()],
                     [torch.zeros(outputNz.shape).half()])

    def test_transdata_fp16_nd2nz_case2(self):
        shape1 = (3, 4000, 1000)
        inputNd = np.random.uniform(low=0, high=1, size=shape1).astype(np.float16)
        outputNz = TransformNdToNz(inputNd, (BLOCK_SIZE_16, BLOCK_SIZE_16))
        self.set_param(OP_NAME, OP_PARAM0)
        self.set_input_formats([self.format_nd])
        self.set_output_formats([self.format_nz])
        self.execute([torch.tensor(inputNd).half()],
                     [torch.zeros(outputNz.shape).half()])

if __name__ == '__main__':
    unittest.main()
