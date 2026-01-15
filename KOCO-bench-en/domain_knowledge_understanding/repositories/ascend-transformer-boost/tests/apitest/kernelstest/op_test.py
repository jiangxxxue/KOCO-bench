# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import os
import unittest
import logging
import json
import re
import numpy
import torch
import torch_npu
import warnings


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


ATB_HOME_PATH = os.environ.get("ATB_HOME_PATH")
if ATB_HOME_PATH is None:
    raise RuntimeError(
        "env ATB_HOME_PATH not exist, source set_env.sh")
LIB_PATH = os.path.join(ATB_HOME_PATH, "lib/libmki_torch.so")
torch.classes.load_library(LIB_PATH)


class OpTest(unittest.TestCase):
    def setUp(self):
        logging.info("running testcase "
                     f"{self.__class__.__name__}.{self._testMethodName}")
        self.format_default = -1
        self.format_nz = 29
        self.format_nchw = 0
        self.format_nhwc = 1
        self.format_nc1hwc0 = 3
        self.format_nd = 2
        self.multiplex = False
        self.out_flag = False
        self.support_soc = []
        self.nct = False

    def set_param(self, op_name, op_param):
        self.op_desc = {
            "opName": op_name,
            "specificParam": op_param}
        self.mki = torch.classes.MkiTorch.MkiTorch(json.dumps(
            self.op_desc))

    def set_param_perf(self, op_name, run_times, op_param):
        self.op_desc = {
            "opName": op_name,
            "runTimes": run_times,
            "specificParam": op_param}
        self.mki = torch.classes.MkiTorch.MkiTorch(json.dumps(
            self.op_desc))

    def set_support_910b(self):
        warnings.warn(
            "It is useless and will be removed recently, please use soc decorator instead", DeprecationWarning)
        self.support_soc.append("Ascend910B")

    def set_support_310p(self):
        warnings.warn(
            "It is useless and will be removed recently, please use soc decorator instead", DeprecationWarning)
        self.support_soc.append("Ascend310P")

    def set_support_910b_only(self):
        warnings.warn(
            "It is useless and will be removed recently, please use soc decorator instead", DeprecationWarning)
        self.support_soc = ["Ascend910B"]

    def set_support_310p_only(self):
        warnings.warn(
            "It is useless and will be removed recently, please use soc decorator instead", DeprecationWarning)
        self.support_soc = ["Ascend310P"]

    def set_input_formats(self, formats):
        self.op_desc["input_formats"] = formats
        self.mki = torch.classes.MkiTorch.MkiTorch(json.dumps(
            self.op_desc))

    def set_output_formats(self, formats):
        self.op_desc["output_formats"] = formats
        self.mki = torch.classes.MkiTorch.MkiTorch(json.dumps(
            self.op_desc))

    def execute(self, in_tensors, out_tensors, envs=None):
        npu_device = self.__get_npu_device()
        torch_npu.npu.set_device(npu_device)

        if self.support_soc:
            soc_version = get_soc_name()
            if (soc_version == None):
                quit(1)

            if soc_version not in self.support_soc:
                logging.info("Current soc is %s is not supported for this case: %s",
                             soc_version, str(self.support_soc))
                return

        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        out_tensors_npu = [in_tensors_npu[i] if isinstance(i, int) else i.npu()
                           for i in out_tensors]

        self.__set_envs(envs)
        if self.nct:
            self.mki.execute_nct(in_tensors_npu, out_tensors_npu)
        else:
            self.mki.execute(in_tensors_npu, out_tensors_npu)
        self.__unset_envs(envs)

        if out_tensors_npu:
            out_tensors = [tensor.cpu() for tensor in out_tensors_npu]
        else:
            logging.info("No output tensor, use input tensors as output")
            out_tensors = [tensor.cpu() for tensor in in_tensors_npu]

        golden_out_tensors = self.golden_calc(in_tensors)
        for idx, tensor in enumerate(in_tensors):
            logging.debug("PythonTest Input Tensor[%s]:", idx)
            logging.debug(tensor)
        for idx, tensor in enumerate(out_tensors):
            logging.debug("PythonTest Output Tensor[%s]:", idx)
            logging.debug(tensor)
        for idx, tensor in enumerate(golden_out_tensors):
            logging.debug("PythonTest Golden Tensor[%s]:", idx)
            logging.debug(tensor)
        self.assertTrue(self.golden_compare(out_tensors, golden_out_tensors))

    def execute_perf(self, in_tensors, out_tensors, envs=None):
        npu_device = self.__get_npu_device()
        torch_npu.npu.set_device(npu_device)
        if self.support_soc:
            soc_version = get_soc_name()
            if (soc_version == None):
                quit(1)

            if soc_version not in self.support_soc:
                logging.info("Current soc is %s is not supported for this case: %s",
                             soc_version, str(self.support_soc))
                return

        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        out_tensors_npu = [in_tensors_npu[i] if isinstance(i, int) else i.npu()
                           for i in out_tensors]
        self.__set_envs(envs)
        if self.nct:
            self.run_result = self.mki.execute_nct(in_tensors_npu, out_tensors_npu)
        else:
            self.run_result = self.mki.execute(in_tensors_npu, out_tensors_npu)
        self.__unset_envs(envs)

        if out_tensors_npu:
            out_tensors = [tensor.cpu() for tensor in out_tensors_npu]
        else:
            logging.info("No output tensor, use input tensors as output")
            out_tensors = [tensor.cpu() for tensor in in_tensors_npu]

        return out_tensors

    def __set_envs(self, env: dict):
        if env:
            for key, value in env.items():
                os.environ[key] = value

    def __unset_envs(self, env: dict):
        if env:
            for key, _ in env.items():
                os.environ[key] = ""

    def __get_npu_device(self):
        npu_device = os.environ.get("MKI_NPU_DEVICE")
        if npu_device is None:
            npu_device = "npu:0"
        else:
            npu_device = f"npu:{npu_device}"
        return npu_device

    def __create_tensor(self, dtype, format, shape, minValue, maxValue, device=None):
        if device is None:
            device = self.__get_npu_device()
        input = numpy.random.uniform(minValue, maxValue, shape).astype(dtype)
        cpu_input = torch.from_numpy(input)
        npu_input = torch.from_numpy(input).to(device)
        if format != -1:
            npu_input = torch_npu.npu_format_cast(npu_input, format)
        return cpu_input, npu_input


def get_soc_name():
    device_name = torch_npu.npu.get_device_name()
    if (re.search("Ascend910B", device_name, re.I) and len(device_name) > 10) or re.search("Ascend910_93", device_name, re.I):
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
    elif re.search("Ascend310B", device_name, re.I) and len(device_name) > 10:
        soc_version = "Ascend310B"
    else:
        logging.error("device_name %s is not supported", device_name)
        soc_version = None
    return soc_version


def only_soc(soc_name):
    return unittest.skipIf(soc_name != get_soc_name(), f"This case only runs on {soc_name}")


only_910b = only_soc("Ascend910B")
only_910b4 = only_soc("Ascend910B4")
only_310p = only_soc("Ascend310P")
only_910a = only_soc("Ascend910A")
skip_910a = unittest.skipIf(get_soc_name() == "Ascend910A","don't support 910a")
only_310b = only_soc("Ascend310B")
skip_310b = unittest.skipIf(get_soc_name() == "Ascend310B", "don't support 310b")
only_310p_and_910a = unittest.skipIf(get_soc_name() != "Ascend310P" and get_soc_name() != "Ascend910A","only support 310p and 910a")
