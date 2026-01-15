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
import logging
import numpy as np
import torch
import itertools
import op_test

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
OP_NAME = "ActivationOperation"
OP_PARAM = {"activationType": 9}  # faster_gelu_forward

test_list = [1, 4, 15, 16, 17, 32, 64, 65, 128, 256, 257, 13107]
generalize_shapes = list(itertools.product(test_list, repeat=2))
generalize_shapes = [list(pair) for pair in generalize_shapes]


class TestFasterGeluActivation(op_test.OpTest):

    @staticmethod
    def fast_gelu_ori():
        def forward(in_tensor) -> torch.Tensor:
            abs_value = torch.abs(in_tensor)
            return in_tensor * torch.sigmoid(1.702 * abs_value) * torch.exp(0.851 * (in_tensor - abs_value))

        return forward

    def golden_calc(self, in_tensors):
        x = in_tensors[0].float()
        y = self.fast_gelu_ori()(x).to(in_tensors[0].dtype)
        return [y]

    def golden_compare(self, out_tensors, golden_out_tensors):
        logging.debug(f"f==golden result==\n {golden_out_tensors[0]}")
        logging.debug(f"==output result==\n {out_tensors[0]}")
        return self._golden_compare(out_tensors, golden_out_tensors)

    @staticmethod
    def _golden_compare(out_tensors, golden_out_tensors):
        err = 2 ** (-8)
        if out_tensors[0].dtype == torch.bfloat16:
            err = 2 ** (-7)
        elif out_tensors[0].dtype == torch.float32:
            err = 2 ** (-14)
        out_tensors[0] = out_tensors[0].to(torch.float32)
        golden_out_tensors[0] = golden_out_tensors[0].to(torch.float32)
        diff = torch.abs(torch.subtract(out_tensors[0], golden_out_tensors[0]))
        tensor_max = torch.maximum(torch.ones(golden_out_tensors[0].shape, dtype=golden_out_tensors[0].dtype),
                                   torch.abs(golden_out_tensors[0]))
        if torch.any(torch.greater(diff, err * tensor_max)):
            logging.debug("new golden result: failed")
            return False
        logging.debug("new golden result: true")
        return True

    @op_test.only_910b
    def test_faster_gelu_bf16_base(self):
        for shape in [[40, 24], [1, 1024], [8, 5504], [8, 64], [8192, 5504], [8192, 64]]:
            input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float32).flatten()
            self.set_param(OP_NAME, OP_PARAM)
            self.set_input_formats([self.format_nd] * 1)
            self.set_output_formats([self.format_nd] * 1)
            self.execute([torch.from_numpy(input0).bfloat16()],
                         [torch.zeros(input0.shape).bfloat16()])

    @op_test.skip_310b
    def test_faster_gelu_fp32_base(self):
        for shape in [[40, 24], [1, 1024], [8, 5504], [8, 64], [8192, 5504], [8192, 64]]:
            input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float32).flatten()
            self.set_param(OP_NAME, OP_PARAM)
            self.set_input_formats([self.format_nd] * 1)
            self.set_output_formats([self.format_nd] * 1)
            self.execute([torch.from_numpy(input0)],
                         [torch.zeros(input0.shape)])

    @op_test.skip_310b
    def test_faster_gelu_fp16_base(self):
        for shape in [[40, 24], [1, 1024], [8, 5504], [8, 64], [8192, 5504], [8192, 64]]:
            input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float32).flatten()
            self.set_param(OP_NAME, OP_PARAM)
            self.set_input_formats([self.format_nd] * 1)
            self.set_output_formats([self.format_nd] * 1)
            self.execute([torch.from_numpy(input0).half()],
                         [torch.zeros(input0.shape).half()])

    @op_test.skip_310b
    def test_faster_gelu_fp16_nz_cases(self):
        for shape in [[40, 24], [1, 1024], [8, 5504], [8, 64], [8192, 5504], [8192, 64]]:
            input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float32).flatten()
            self.set_param(OP_NAME, OP_PARAM)
            self.set_input_formats([self.format_nz] * 1)
            self.set_output_formats([self.format_nz] * 1)
            self.execute([torch.from_numpy(input0).half()],
                         [torch.zeros(input0.shape).half()])

    @op_test.skip_310b
    def test_faster_gelu_generalize_fp32(self):
        for shape in generalize_shapes:
            input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float32).flatten()
            self.set_param(OP_NAME, OP_PARAM)
            self.set_input_formats([self.format_nd] * 1)
            self.set_output_formats([self.format_nd] * 1)
            self.execute([torch.from_numpy(input0)],
                         [torch.zeros(input0.shape)])

    @op_test.skip_310b
    def test_generalize_16Align_fp16(self):
        base_shape = [16, 1024]
        shapes = []
        for i in range(1, 31):
            shapes.append([base_shape[0], base_shape[1] + 16 * i])
        for shape in shapes:
            input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float32).flatten()
            self.set_param(OP_NAME, OP_PARAM)
            self.set_input_formats([self.format_nd] * 1)
            self.set_output_formats([self.format_nd] * 1)
            self.execute([torch.from_numpy(input0).half()],
                         [torch.zeros(input0.shape).half()])

    @op_test.skip_310b
    def test_generalize_16Unalign_fp16(self):
        base_shape = [15, 1024]
        shapes = []
        for i in range(1, 31):
            shapes.append([base_shape[0], base_shape[1] + 15 * i])
        for shape in shapes:
            input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float32).flatten()
            self.set_param(OP_NAME, OP_PARAM)
            self.set_input_formats([self.format_nd] * 1)
            self.set_output_formats([self.format_nd] * 1)
            self.execute([torch.from_numpy(input0).half()],
                         [torch.zeros(input0.shape).half()])

    @op_test.skip_310b
    def test_generalize_16Align_fp32(self):
        base_shape = [16, 1024]
        shapes = []
        for i in range(1, 31):
            shapes.append([base_shape[0], base_shape[1] - 16 * i])
        for shape in shapes:
            input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float32).flatten()
            self.set_param(OP_NAME, OP_PARAM)
            self.set_input_formats([self.format_nd] * 1)
            self.set_output_formats([self.format_nd] * 1)
            self.execute([torch.from_numpy(input0).half()],
                         [torch.zeros(input0.shape).half()])

    @op_test.skip_310b
    def test_generalize_16Unalign_fp32(self):
        base_shape = [15, 1024]
        shapes = []
        for i in range(1, 31):
            shapes.append([base_shape[0], base_shape[1] - 15 * i])
        for shape in shapes:
            input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float32).flatten()
            self.set_param(OP_NAME, OP_PARAM)
            self.set_input_formats([self.format_nd] * 1)
            self.set_output_formats([self.format_nd] * 1)
            self.execute([torch.from_numpy(input0).half()],
                         [torch.zeros(input0.shape).half()])

    @op_test.only_910b
    def test_generalize_16Align_bf16(self):
        base_shape = [16, 1024]
        shapes = []
        for i in range(1, 31):
            shapes.append([base_shape[0], base_shape[1] + 16 * i])
        for shape in shapes:
            input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float32).flatten()
            self.set_param(OP_NAME, OP_PARAM)
            self.set_input_formats([self.format_nd] * 1)
            self.set_output_formats([self.format_nd] * 1)
            self.execute([torch.from_numpy(input0).bfloat16()],
                         [torch.zeros(input0.shape).bfloat16()])

    @op_test.only_910b
    def test_generalize_16Unalign_bf16(self):
        base_shape = [15, 1024]
        shapes = []
        for i in range(1, 31):
            shapes.append([base_shape[0], base_shape[1] - 15 * i])
        for shape in shapes:
            input0 = np.random.uniform(low=-5, high=5, size=shape).astype(np.float32).flatten()
            self.set_param(OP_NAME, OP_PARAM)
            self.set_input_formats([self.format_nd] * 1)
            self.set_output_formats([self.format_nd] * 1)
            self.execute([torch.from_numpy(input0).bfloat16()],
                         [torch.zeros(input0.shape).bfloat16()])


if __name__ == '__main__':
    unittest.main()
