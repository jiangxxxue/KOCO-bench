#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import torch
import torch_npu
import json
import os

ASCEND_HOME_PATH = os.environ.get("ASCEND_HOME_PATH")
if ASCEND_HOME_PATH is None:
    raise RuntimeError("env ASCEND_HOME_PATH not exist, source set_env.sh")

ATB_HOME_PATH = os.environ.get("ATB_HOME_PATH")
if ATB_HOME_PATH is None:
    raise RuntimeError("env ATB_HOME_PATH not exist, source set_env.sh")

torch.classes.load_library(os.path.join(ATB_HOME_PATH, "lib/libatb_test_framework.so"))

DEVICE_ID = os.environ.get("SET_NPU_DEVICE")
if DEVICE_ID is not None:
    torch.npu.set_device(torch.device(f"npu:{DEVICE_ID}"))
else:
    torch.npu.set_device(torch.device(f"npu:0"))

def add_operation_execution():
    operation = torch.classes.OperationTorch.OperationTorch("ElewiseOperation")
    param = {"elewiseType": 8}
    tensors = [torch.randn(1, 4).npu(), torch.randn(1, 4).npu()]
    operation.set_param(json.dumps(param))
    operation.execute(tensors)

def delete_tmp():
    current_dir = os.getcwd()
    if os.path.exists(f"{current_dir}/tmp"):
        os.system(f"rm -rf {current_dir}/tmp")