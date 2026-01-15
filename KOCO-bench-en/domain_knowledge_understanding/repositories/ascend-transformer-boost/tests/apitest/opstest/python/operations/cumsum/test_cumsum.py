#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import sys
import os
import unittest
import torch
import torch_npu
import logging
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test  # NOQA: E402

OP_NAME = "CumsumOperation"
PARAM = {"axes":[1]}


class TestCumsumOperation(operation_test.OperationTest):
    def golden_calc(self, in_tensors):
        in_tensors = [tensor.cpu() for tensor in in_tensors]
        golden_result = np.cumsum(in_tensors[0], axis=1)
        # golden_result = torch.cumsum(in_tensors[0], dim=1)
        return [golden_result.npu()]
    
    def test(self):
        if  operation_test.get_soc_version() == 'Ascend910A' or \
            operation_test.get_soc_version() == 'Ascend310B':
            logging.info("this testcase don't supports Ascend910A\Ascend310B")
            return True
        self.execute(OP_NAME, PARAM, [torch.randn(28, 32, 4096, dtype=torch.float16).npu()])

if __name__ == '__main__':
    unittest.main()
