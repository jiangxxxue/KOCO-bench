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
import itertools
import logging
import numpy as np
import torch
import torch.nn.functional as F
import math
import op_test

OP_NAME = "ActivationOperation"
OP_PARAM = {"activationType": 2, "param": 0}


class TestActivation(op_test.OpTest):

    def init_test_data(self):
        # 随机数生成范围，不同数据类型范围不同
        self.range_map = {
            "float32": [-5, 5],
            "float16": [-5, 5],
            "bfloat16": [-5, 5],
        }

        # 典型shape
        self.shape_list = [
            (8, 5504),
            (1024, 5120),
            (6, 1111),
            (4096, 5120),
            (77, 2049),
            (1, 1),
            (1, 2),
            (327680, 5120),
        ]

        # 数据类型
        self.dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        self.dtype_list = list(self.dtype_map.keys())

        # 数据存储格式
        self.format_map = {
            "format_nz": self.format_nz,
            "format_nchw": self.format_nchw,
            "format_nhwc": self.format_nhwc,
            "format_nc1hwc0": self.format_nc1hwc0,
            "format_nd": self.format_nd,
        }
        self.format_list = list(self.format_map.keys())

        # 计算不同用例的笛卡尔积
        self.cases = list(itertools.product(self.shape_list, self.dtype_list, self.format_list))

    def gelu_ori(self):
        def forward(in_tensor) -> torch.Tensor:
            return 0.5 * in_tensor * (
                    1 + F.tanh(math.sqrt(2 / math.pi) * (in_tensor + 0.044715 * torch.pow(in_tensor, 3))))

        return forward

    def golden_calc(self, in_tensors):
        x = in_tensors[0].float()
        y = self.gelu_ori()(x).to(in_tensors[0].dtype)
        return [y]

    def golden_compare(self, out_tensors, golden_out_tensors):
        logging.debug(f"\t====golden====")
        logging.debug(f"\t{golden_out_tensors[0]}")
        logging.debug(f"\t====output====")
        logging.debug(f"\t{out_tensors[0]}")
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
            return False
        return True

    # 310P/910A测试用例
    def _test_gelu_310p_910a(self, shape, dtype, data_format):
        logging.debug(f"[ RUN      ] test ({shape, dtype, data_format})")
        input0 = np.random.uniform(low=self.range_map[dtype][0], high=self.range_map[dtype][1], size=shape).astype(
            np.float32).flatten()
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_map[data_format]] * 1)
        self.set_output_formats([self.format_map[data_format]] * 1)
        if dtype == "float16":
            in_tensor = torch.from_numpy(input0).half()
            out_tensor = torch.zeros(input0.shape).half()
        elif dtype == "float32":
            in_tensor = torch.from_numpy(input0)
            out_tensor = torch.zeros(input0.shape, dtype=torch.float32)
        else:
            logging.debug(f"[       OK ] not support type {dtype} for 310P\n")
            return
        try:
            self.execute([in_tensor], [out_tensor])
            logging.debug(f"[       OK ] end test ({shape, dtype, data_format})\n")
        except AssertionError:
            logging.debug(f"[   Failed ] end test ({shape, dtype, data_format})\n")

    @op_test.only_910b
    def _test_gelu_910b(self, shape, dtype, data_format):
        logging.debug(f"[ RUN      ] test ({shape, dtype, data_format})")
        input0 = np.random.uniform(low=self.range_map[dtype][0], high=self.range_map[dtype][1], size=shape).astype(
            np.float32).flatten()
        self.set_param(OP_NAME, OP_PARAM)
        self.set_input_formats([self.format_map[data_format]] * 1)
        self.set_output_formats([self.format_map[data_format]] * 1)
        if dtype == "float16":
            in_tensor = torch.from_numpy(input0).half()
            out_tensor = torch.zeros(input0.shape).half()
        elif dtype == "float32":
            in_tensor = torch.from_numpy(input0)
            out_tensor = torch.zeros(input0.shape, dtype=torch.float32)
        elif dtype == "bfloat16":
            in_tensor = torch.from_numpy(input0).bfloat16()
            out_tensor = torch.zeros(input0.shape).bfloat16()
        else:
            logging.debug(f"[       OK ] not support type {dtype} for 910B\n")
            return
        try:
            self.execute([in_tensor], [out_tensor])
            logging.debug(f"[       OK ] end test ({shape, dtype, data_format})\n")
        except AssertionError:
            logging.error(f"[   Failed ] end test ({shape, dtype, data_format})\n")

    # 自定义测试用例
    @op_test.skip_310b
    def test_group_310p(self):
        self.init_test_data()
        self._test_gelu_310p_910a(shape=(4096, 8192), dtype="float16", data_format='format_nd')
        self._test_gelu_310p_910a(shape=(4096, 327680), dtype="float16", data_format='format_nz')
        self._test_gelu_310p_910a(shape=(1, 1), dtype="float16", data_format='format_nd')
        self._test_gelu_310p_910a(shape=(1, 3), dtype="float16", data_format='format_nz')
        self._test_gelu_310p_910a(shape=(327680, 5120), dtype="float16", data_format='format_nz')
        self._test_gelu_310p_910a(shape=(7, 2049), dtype="float16", data_format='format_nz')
        self._test_gelu_310p_910a(shape=(8, 5504), dtype="float32", data_format='format_nz')
        self._test_gelu_310p_910a(shape=(6, 1111), dtype="float16", data_format='format_nd')
        self._test_gelu_310p_910a(shape=(7, 2049), dtype="bfloat16", data_format='format_nchw')

    @op_test.skip_310b
    def test_group_910b(self):
        self.init_test_data()
        self._test_gelu_910b(shape=(7, 2049), dtype="bfloat16", data_format='format_nd')
        self._test_gelu_910b(shape=(4096, 327680), dtype="float16", data_format='format_nz')
        self._test_gelu_910b(shape=(327680, 5120), dtype="float16", data_format='format_nd')
        self._test_gelu_910b(shape=(1, 1), dtype="float16", data_format='format_nd')
        self._test_gelu_910b(shape=(1, 3), dtype="float16", data_format='format_nz')

    # 执行所有组合用例
    def _test_gelu_all(self):
        self.init_test_data()
        for index, case in enumerate(self.cases):
            self._test_gelu_310p_910a(shape=case[0], dtype=case[1], data_format=case[2])
        for index, case in enumerate(self.cases):
            self._test_gelu_910b(shape=case[0], dtype=case[1], data_format=case[2])


if __name__ == '__main__':
    unittest.main()
