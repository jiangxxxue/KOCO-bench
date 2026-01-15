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
import sys
import unittest
import json
import numpy
import torch
import torch_npu
import re

ATB_HOME_PATH = os.environ.get("ATB_HOME_PATH")
if ATB_HOME_PATH is None:
    raise RuntimeError(
        "env ATB_HOME_PATH not exist, source set_env.sh")

sys.path.append(os.path.join(os.path.dirname(__file__), "./pythontools"))

import tensor_file  # NOQA:E402

LIB_PATH = os.path.join(ATB_HOME_PATH, "lib/libatb_test_framework.so")
torch.classes.load_library(LIB_PATH)

DEVICE_ID = os.environ.get("SET_NPU_DEVICE")
if DEVICE_ID is not None:
    torch.npu.set_device(torch.device(f"npu:{DEVICE_ID}"))


def get_soc_version():
    device_name = torch.npu.get_device_name()
    if (re.search("Ascend910B", device_name, re.I) and len(device_name) > 10) or re.search("Ascend910_93", device_name,
                                                                                           re.I):
        soc_version = "Ascend910B"
    elif re.search("Ascend310P", device_name, re.I):
        soc_version = "Ascend310P"
    elif re.search("Ascend910ProB", device_name, re.I):
        soc_version = "Ascend910A"
    elif re.search("Ascend910B", device_name, re.I):
        soc_version = "Ascend910A"
    elif re.search("Ascend910PremiumA", device_name, re.I):
        soc_version = "Ascend910A"
    elif re.search("Ascend910ProA", device_name, re.I):
        soc_version = "Ascend910A"
    elif re.search("Ascend910A", device_name, re.I):
        soc_version = "Ascend910A"
    else:
        print("device_name {} is not supported".format(device_name))
        quit(1)
    device_count = torch.npu.device_count()
    current_device = torch.npu.current_device()
    print(
        "Device Properties: device_name: {}, soc_version: {}, device_count: {}, current_device: {}".format(device_name,
                                                                                                           soc_version,
                                                                                                           device_count,
                                                                                                           current_device))
    return soc_version


class OperationTest(unittest.TestCase):
    operation = None

    def execute(self, op_name, op_param, in_tensors):
        print(f"———————— {op_name} test start ————————")
        self.op_param = op_param
        self.operation = torch.classes.OperationTorch.OperationTorch(
            op_name)
        if isinstance(op_param, dict):
            self.operation.set_param(json.dumps(op_param))
        elif isinstance(op_param, str):
            self.operation.set_param(op_param)
        out_tensors = self.operation.execute(in_tensors)
        print("out_tensor", out_tensors[0].size())
        golden_out_tensors = self.golden_calc(in_tensors)
        print("golden_calc", golden_out_tensors[0].size())
        self.__golden_compare_all(out_tensors, golden_out_tensors)

    def execute_out(self, op_name, op_param, in_tensors, out_tensors):
        print(f"———————— {op_name} test start ————————")
        self.op_param = op_param
        self.operation = torch.classes.OperationTorch.OperationTorch(
            op_name)
        if isinstance(op_param, dict):
            self.operation.set_param(json.dumps(op_param))
        elif isinstance(op_param, str):
            self.operation.set_param(op_param)
        self.operation.execute_out(in_tensors, out_tensors)
        print("out_tensor", out_tensors[0].size())
        golden_out_tensors = self.golden_calc(in_tensors)
        print("golden_calc", golden_out_tensors[0].size())
        self.__golden_compare_all(out_tensors, golden_out_tensors)

    def execute_with_param(self, op_name, op_param, run_param, in_tensors):
        print(f"———————— {op_name} test start ————————")
        self.operation = torch.classes.OperationTorch.OperationTorch(
            op_name)
        if isinstance(op_param, dict):
            self.operation.set_param(json.dumps(op_param))
        elif isinstance(op_param, str):
            self.operation.set_param(op_param)
        self.operation.set_varaintpack_param(run_param)
        out_tensors = self.operation.execute(in_tensors)
        print("out_tensor", out_tensors[0].size())
        golden_out_tensors = self.golden_calc(in_tensors)
        print("golden_calc", golden_out_tensors[0].size())
        self.__golden_compare_all(out_tensors, golden_out_tensors)

    def execute_inplace(self, op_name, op_param, in_tensors, indexes):
        print(f"———————— {op_name} test start ————————")
        operation = torch.classes.OperationTorch.OperationTorch(
            op_name)
        if isinstance(op_param, dict):
            operation.set_param(json.dumps(op_param))
        elif isinstance(op_param, str):
            operation.set_param(op_param)
        operation.execute(in_tensors)
        out_tensors = []
        for index in indexes:
            out_tensors.append(in_tensors[index])
        golden_out_tensors = self.golden_calc(in_tensors)
        self.__golden_compare_all(out_tensors, golden_out_tensors)

    def execute_update_param(self, op_name, op_param, in_tensors):
        print(f"———————— {op_name} test start ————————")
        self.assertIsNotNone(self.operation, "self.operation should not be None")
        self.op_param = op_param
        if isinstance(op_param, dict):
            self.operation.update_param(json.dumps(op_param))
        elif isinstance(op_param, str):
            self.operation.update_param(op_param)
        out_tensors = self.operation.execute(in_tensors)
        print("out_tensor", out_tensors[0].size())
        golden_out_tensors = self.golden_calc(in_tensors)
        print("golden_calc", golden_out_tensors[0].size())
        self.__golden_compare_all(out_tensors, golden_out_tensors)

    def golden_compare(self, out_tensor, golden_out_tensor, rtol=0.02, atol=0.02):
        result = torch.allclose(out_tensor.cpu(), golden_out_tensor.cpu(), rtol=rtol, atol=atol)
        if not result:
            print("out_tensor.shape", out_tensor.shape,
                  "\ngolden_out_tensor.shape:", golden_out_tensor.shape)
            print("out_tensor:", out_tensor,
                  ", \ngolden_oute_tensor:", golden_out_tensor)
        return result

    def get_tensor(self, file_path):
        if not os.path.exists(file_path):
            raise RuntimeError(f"{file_path} not exist")
        return tensor_file.read_tensor(file_path)

    def execute_with_param_and_tensor_list(self, op_name, op_param, run_param, in_tensors, tensor_list, list_name):
        operation = torch.classes.OperationTorch.OperationTorch(
            op_name)
        if isinstance(op_param, dict):
            operation.set_param(json.dumps(op_param))
        elif isinstance(op_param, str):
            operation.set_param(op_param)
        for i in range(len(tensor_list)):
            operation.set_tensor_list(tensor_list[i], list_name[i])
        operation.set_varaintpack_param(run_param)
        out_tensors = operation.execute(in_tensors)
        print("out_tensor", out_tensors[0].size())
        golden_out_tensors = self.golden_calc(in_tensors)
        print("golden_calc", golden_out_tensors[0].size())
        self.__golden_compare_customize(out_tensors, golden_out_tensors)

    def __golden_compare_all(self, out_tensors, golden_out_tensors):
        self.assertEqual(len(out_tensors), len(golden_out_tensors))
        tensor_count = len(out_tensors)
        for i in range(tensor_count):
            self.assertTrue(self.golden_compare(
                out_tensors[i], golden_out_tensors[i]))

    def __golden_compare_customize(self, out_tensors, golden_out_tensors):
        self.assertTrue(self.golden_compare(out_tensors, golden_out_tensors))

    def __get_npu_device(self):
        npu_device = os.environ.get("MKI_NPU_DEVICE")
        if npu_device is None:
            npu_device = "npu:0"
        else:
            npu_device = f"npu:{npu_device}"
        return npu_device