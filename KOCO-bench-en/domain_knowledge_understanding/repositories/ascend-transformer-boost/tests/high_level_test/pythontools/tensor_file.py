#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import numpy
import torch

ATTR_VERSION = "$Version"
ATTR_END = "$End"
ATTR_OBJECT_LENGTH = "$Object.Length"
ATTR_OBJECT_COUNT = "$Object.Count"
ATTR_OBJECT_PREFIX = "$Object."


class TensorBinFile:
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.dtype = 0
        self.format = 0
        self.dims = []

        self.__parse_bin_file()

    def get_tensor(self):
        if self.dtype == 0:
            dtype = numpy.float32
        elif self.dtype == 1:
            dtype = numpy.float16
        elif self.dtype == 2:  # int8
            dtype = numpy.int8
        elif self.dtype == 3:  # int32
            dtype = numpy.int32
        elif self.dtype == 9:  # int64
            dtype = numpy.int64
        elif self.dtype == 12:
            dtype = numpy.bool8
        else:
            print("error, unsupport dtype:", self.dtype)
            pass
        tensor = torch.tensor(numpy.frombuffer(self.obj_buffer, dtype=dtype))
        tensor = tensor.view(self.dims)
        return tensor

    def __parse_bin_file(self):
        end_str = f"{ATTR_END}=1"
        with open(self.file_path, "rb") as fd:
            file_data = fd.read()
            file_data_len = len(file_data)
            begin_offset = 0
            for i in range(file_data_len):
                if file_data[i] == ord("\n"):
                    line = file_data[begin_offset: i].decode("utf-8")
                    begin_offset = i + 1
                    fields = line.split("=")
                    attr_name = fields[0]
                    attr_value = fields[1]
                    if attr_name == ATTR_END:
                        self.obj_buffer = file_data[i + 1:]
                        break
                    elif attr_name.startswith("$"):
                        self.__parse_system_atrr(attr_name, attr_value)
                    else:
                        self.__parse_user_attr(attr_name, attr_value)
                        pass

    def __parse_system_atrr(self, attr_name, attr_value):
        if attr_name == ATTR_OBJECT_LENGTH:
            self.obj_len = int(attr_value)
        elif attr_name == ATTR_OBJECT_PREFIX:

            pass

    def __parse_user_attr(self, attr_name, attr_value):
        if attr_name == "dtype":
            self.dtype = int(attr_value)
        elif attr_name == "format":
            self.format = int(attr_value)
        elif attr_name == "dims":
            self.dims = attr_value.split(",")
            dims_len = len(self.dims)
            for i in range(dims_len):
                self.dims[i] = int(self.dims[i])


def read_tensor(file_path):
    if file_path.endswith(".bin"):
        binfile = TensorBinFile(file_path)
        return binfile.get_tensor()
    else:
        try:
            return list(torch.load(file_path).state_dict().values())[0]
        except Exception as e:
            return torch.load(file_path)
