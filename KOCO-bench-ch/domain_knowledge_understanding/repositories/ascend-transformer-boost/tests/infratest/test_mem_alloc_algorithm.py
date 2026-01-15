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
import task_runner
import torch
import infratest_utils

class TestWorkSpaceAlloc(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        intensor0 = torch.rand(28, 8, 8192, dtype=torch.float16).npu().half()
        intensor1 = torch.rand(8192, dtype=torch.float16).npu().half()
        intensor2 = torch.rand(3072, 8192, dtype=torch.float16).npu().half()
        intensor3 = torch.rand(8192, 1024, dtype=torch.float16).npu().half()
        intensor4 = torch.rand(8192, dtype=torch.float16).npu().half()
        intensor5 = torch.rand(5504, 8192, dtype=torch.float16).npu().half()
        intensor6 = torch.rand(1, 172, 8192, 16, dtype=torch.float16).npu().half()
        intensor6_nz = torch_npu.npu_format_cast(intensor6, 29) # format NZ
        intensor7 = torch.randint(1, 10, size=(28, 8), dtype=torch.int64).npu()
        intensor8 = torch.rand(2, 28, 1, 8, 128, dtype=torch.float32).npu()
        intensor8_nd = torch_npu.npu_format_cast(intensor8, 2) # format NZ
        intensor9 = torch.rand(28, 1, 4096, 4096, dtype=torch.float16).npu().half()
        intensor10 = torch.rand(28, 8, 4096, 128, dtype=torch.float16).npu().half()
        intensor11 = torch.rand(28, 8, 4096, 128, dtype=torch.float16).npu().half()
        intensor12 = torch.randint(1, 10, size=(28, 1), dtype=torch.int32).npu()
        intensor13 = torch.randint(1, 10, size=(28, 1), dtype=torch.int32).npu()
        intensor14 = torch.randint(1, 10, size=(1,), dtype=torch.int32).npu()
        intensor15 = torch.randint(1, 10, size=(28,), dtype=torch.int32).npu()
        TestWorkSpaceAlloc.intensors = [intensor0, intensor1, intensor2, intensor3, intensor4, intensor5, intensor6_nz, intensor7,
                   intensor8_nd, intensor9, intensor10, intensor11, intensor12, intensor13, intensor14, intensor15]
        outtensor0 = torch.rand(28, 8, 8192, dtype=torch.float16).npu().half()
        TestWorkSpaceAlloc.outtensors = [outtensor0]

    def test_workspacesize_switch(self):
        os.putenv("ATB_WORKSPACE_MEM_ALLOC_GLOBAL", "0")
        workspace_size_off = self.diff_mem_alloc_global_type(TestWorkSpaceAlloc.intensors, TestWorkSpaceAlloc.outtensors)
        os.putenv("ATB_WORKSPACE_MEM_ALLOC_GLOBAL", "1")
        workspace_size_on = self.diff_mem_alloc_global_type(TestWorkSpaceAlloc.intensors, TestWorkSpaceAlloc.outtensors)
        assert workspace_size_off > workspace_size_on
    
    def test_workspacesize_alg_type(self):
        os.putenv("ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE", "0")
        workspace_size_0 = self.diff_mem_alloc_global_type(TestWorkSpaceAlloc.intensors, TestWorkSpaceAlloc.outtensors)
        baseline_alg_0 = 46088192
        assert workspace_size_0 <= baseline_alg_0
        os.putenv("ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE", "1")
        workspace_size_1 = self.diff_mem_alloc_global_type(TestWorkSpaceAlloc.intensors, TestWorkSpaceAlloc.outtensors)
        baseline_alg_1 = 30863360
        assert workspace_size_1 <= baseline_alg_1
        os.putenv("ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE", "3")
        workspace_size_2 = self.diff_mem_alloc_global_type(TestWorkSpaceAlloc.intensors, TestWorkSpaceAlloc.outtensors)
        baseline_alg_2 = 31236096
        assert workspace_size_2 <= baseline_alg_2
        os.putenv("ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE", "4")
        workspace_size_3 = self.diff_mem_alloc_global_type(TestWorkSpaceAlloc.intensors, TestWorkSpaceAlloc.outtensors)
        baseline_alg_3 = 31236096
        assert workspace_size_3 <= baseline_alg_3

    def diff_mem_alloc_global_type(self, intensors, outtensors):
        operation = torch.classes.OperationTorch.OperationTorch("Llama65BLayerOperation")
        param = {"rmsNormEps": 1e-5}
        operation.set_param(json.dumps(param))
        setup_result = operation.setup(intensors, outtensors)
        json_data = json.loads(setup_result)
        workspace_size = json_data['workspace_size']
        return workspace_size

if __name__ == '__main__':
    unittest.main()
