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
import json
import unittest
import torch
import torch_npu
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import operation_test

OP_NAME = "SortOperation"


class TestSortOperation(operation_test.OperationTest):    
    def golden_calc(self, in_tensors):
        num = self.op_param['num']
        values, indices = torch.topk(in_tensors[0].npu(), k=num[0], largest=True)
        print(values.size())
        print(indices.int().size())
        return [values,indices.int()]

    def test_3d_float(self):
        if  operation_test.get_soc_version() == 'Ascend910A' or \
            operation_test.get_soc_version() == 'Ascend310B':
            logging.info("this testcase don't supports Ascend910A\Ascend310B")
            return True
        PARAM = {"num": [3000]}
        self.execute(OP_NAME, PARAM, [torch.randint(-65504, 65504, (10, 22, 4096)).float().npu().half()])
        PARAM = {"num": [1500]}
        self.execute_update_param(OP_NAME,PARAM,[torch.randint(-65504, 65504, (10, 22, 4096)).float().npu().half()])

if __name__ == '__main__':
    unittest.main()
