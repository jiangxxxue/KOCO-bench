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
import os
import unittest
import json
import logging
import datetime
 
 
ATB_HOME_PATH = os.environ.get("ATB_HOME_PATH")
if ATB_HOME_PATH is None:
    raise RuntimeError(
        "env ATB_HOME_PATH not exist, source set_env.sh")
LIBTORCH_PATH = os.path.join(ATB_HOME_PATH, "lib/libatb_test_framework.so")
LIB_PATH = os.path.join(ATB_HOME_PATH, "lib/libatb.so")
torch.classes.load_library(LIBTORCH_PATH)
 
class TestViewGraph(unittest.TestCase):
    def test_view_graph(self):
        param = {}
        operation = torch.classes.OperationTorch.OperationTorch("CommonGraphOperation")
        intensor0 = torch.rand(8, 16, dtype=torch.float16).npu().half()
        intensor1 = torch.rand(32, 16, dtype=torch.float16).npu().half()
        intensors = [intensor0, intensor1]
        outtensor0 = torch.rand(8, 32, dtype=torch.float16).npu().half()
        outtensors = [outtensor0]
        operation.set_param(json.dumps(param))
        setup_result = operation.setup(intensors, outtensors)
        execute_result = operation.execute(intensors)
        json_data = json.loads(setup_result)
        return 0
 
if __name__ == '__main__':
    unittest.main()