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
from pathlib import Path
import ctypes
import torch
import torch_npu

package_dir = Path(__file__).resolve().parent


def _load_atb_libs():
    atb_lib_path = package_dir / 'lib'
    atb_lib_files = ['libmki.so', 'libasdops.so', 'liblcal.so', 'libatb_mixops.so', 'libatb.so', 'libatb_train.so']
    for lib_file in atb_lib_files:
        try:
            ctypes.CDLL(str(atb_lib_path / lib_file), mode=ctypes.RTLD_GLOBAL)
        except OSError as err:
            raise RuntimeError(f"Failed to load {lib_file}: {err}") from err


def _init_env_params():
    os.environ["ATB_HOME_PATH"] = str(package_dir)

_load_atb_libs()
_init_env_params()

from torch_atb._C import *


def get_atb_home_path():
    return os.environ.get("ATB_HOME_PATH")
