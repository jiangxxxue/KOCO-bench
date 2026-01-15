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

def TransformNzToNd(nz_mat: np.ndarray, out_crops: tuple = (0, 0)) -> np.ndarray:
    nzd_mat = np.transpose(nz_mat, (0, 2, 1, 3))
    nd_mat_pad = np.reshape(nzd_mat, (nzd_mat.shape[0], nzd_mat.shape[1], -1))
    if out_crops[0] > 0 and out_crops[1] > 0:
        nd_mat = nd_mat_pad[:, 0:out_crops[0], 0:out_crops[1]]
    else:
        nd_mat = nd_mat_pad
    return nd_mat

class TestTransdataOperation(op_test.OpTest):
    def set_outCrops_param(self, out_crops: tuple = (0, 0)):
        self.out_crops = out_crops

    def golden_calc(self, in_tensors):
        input0 = in_tensors[0].cpu().numpy()
        output0 = TransformNzToNd(input0, self.out_crops)
        return [torch.tensor(output0)]

    def golden_compare(self, out_tensors, golden_out_tensors):
        logging.debug(out_tensors[0].shape)
        logging.debug(golden_out_tensors[0].shape)
        return torch.allclose(out_tensors[0], golden_out_tensors[0], rtol=0.001, atol=0.001)

    def test_transdata_fp16_nz2nd_noUnpad_case1(self):
        OP_PARAM = {"transdataType": 1, "outCrops": [0, 0]}
        shape = (3, 64, 4096, 16)
        out_crops = (0, 0)
        inputNZ = np.random.uniform(low=0, high=1, size=shape).astype(np.float16)
        self.set_outCrops_param(out_crops)
        outputND = TransformNzToNd(inputNZ, out_crops)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nz])
        self.set_output_formats([self.format_nd])
        self.execute([torch.tensor(inputNZ).half()],
                     [torch.zeros(outputND.shape).half()])

    def test_transdata_fp16_nz2nd_noUnpad_case2(self):
        OP_PARAM = {"transdataType": 1, "outCrops": [0, 0]}
        shape = (1, 9504, 16, 16)
        out_crops = (0, 0)
        inputNZ = np.random.uniform(low=0, high=1, size=shape).astype(np.float16)
        self.set_outCrops_param(out_crops)
        outputND = TransformNzToNd(inputNZ, out_crops)
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_nz])
        self.set_output_formats([self.format_nd])
        self.execute([torch.tensor(inputNZ).half()],
                     [torch.zeros(outputND.shape).half()])

    def test_transdata_fp16_nz2nd_Unpad_case1(self):
        OP_PARAM_unpad = {"transdataType": 1, "outCrops": [32, 32]}
        shape = (1, 2, 32, 16)
        out_crops_unpad = (32, 32)
        inputNZ = np.random.uniform(low=0, high=1, size=shape).astype(np.float16)
        self.set_outCrops_param(out_crops_unpad)
        outputND = TransformNzToNd(inputNZ, out_crops_unpad)
        self.set_param(OP_NAME, OP_PARAM_unpad)
        self.set_input_formats([self.format_nz])
        self.set_output_formats([self.format_nd])
        self.execute([torch.tensor(inputNZ).half()],
                     [torch.zeros(outputND.shape).half()])

    def test_transdata_fp16_nz2nd_Unpad_case2(self):
        OP_PARAM_unpad = {"transdataType": 1, "outCrops": [10, 20]}
        shape = (1, 2, 16, 16)
        out_crops_unpad = (10, 20)
        inputNZ = np.random.uniform(low=0, high=1, size=shape).astype(np.float16)
        self.set_outCrops_param(out_crops_unpad)
        outputND = TransformNzToNd(inputNZ, out_crops_unpad)
        self.set_param(OP_NAME, OP_PARAM_unpad)
        self.set_input_formats([self.format_nz])
        self.set_output_formats([self.format_nd])
        self.execute([torch.tensor(inputNZ).half()],
                     [torch.zeros(outputND.shape).half()])

    def test_transdata_fp16_nz2nd_Unpad_case3(self):
        OP_PARAM_unpad = {"transdataType": 1, "outCrops": [32, 32]}
        shape = (2, 2, 32, 16)
        out_crops_unpad = (32, 32)
        inputNZ = np.random.uniform(low=0, high=1, size=shape).astype(np.float16)
        self.set_outCrops_param(out_crops_unpad)
        outputND = TransformNzToNd(inputNZ, out_crops_unpad)
        self.set_param(OP_NAME, OP_PARAM_unpad)
        self.set_input_formats([self.format_nz])
        self.set_output_formats([self.format_nd])
        self.execute([torch.tensor(inputNZ).half()],
                     [torch.zeros(outputND.shape).half()])

    def test_transdata_fp16_nz2nd_Unpad_case4(self):
        OP_PARAM_unpad = {"transdataType": 1, "outCrops": [10, 20]}
        shape = (2, 2, 16, 16)
        out_crops_unpad = (10, 20)
        inputNZ = np.random.uniform(low=0, high=1, size=shape).astype(np.float16)
        self.set_outCrops_param(out_crops_unpad)
        outputND = TransformNzToNd(inputNZ, out_crops_unpad)
        self.set_param(OP_NAME, OP_PARAM_unpad)
        self.set_input_formats([self.format_nz])
        self.set_output_formats([self.format_nd])
        self.execute([torch.tensor(inputNZ).half()],
                     [torch.zeros(outputND.shape).half()])

    def test_transdata_fp16_nz2nd_Unpad_case5(self):
        OP_PARAM_unpad = {"transdataType": 1, "outCrops": [2573, 3792]}
        shape = (1, 237, 2576, 16)
        out_crops_unpad = (2573, 3792)
        inputNZ = np.random.uniform(low=0, high=1, size=shape).astype(np.float16)
        self.set_outCrops_param(out_crops_unpad)
        outputND = TransformNzToNd(inputNZ, out_crops_unpad)
        self.set_param(OP_NAME, OP_PARAM_unpad)
        self.set_input_formats([self.format_nz])
        self.set_output_formats([self.format_nd])
        self.execute([torch.tensor(inputNZ).half()],
                     [torch.zeros(outputND.shape).half()])

if __name__ == '__main__':
    unittest.main()