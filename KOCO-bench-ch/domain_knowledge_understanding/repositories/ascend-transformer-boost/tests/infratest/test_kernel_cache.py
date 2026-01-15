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
import unittest
import os
import json
import logging
import torch
import torch_npu
import task_runner
import infratest_utils


class TestKernelCache(unittest.TestCase):
    """
    kernelCache缓存性能测试:
    开启本地+全局缓存功能
    """

    @classmethod
    def setUpClass(cls):
        TestKernelCache.in_tensors = [torch.randn(16, 32, dtype=torch.float16).npu(), torch.randn(16, 32, dtype=torch.float16).npu()]
        TestKernelCache.out_tensors = [torch.randn(32, 32, dtype=torch.float16).npu()]

    def test_performance(self):
        setup_result, setup_time_with_both_cache, precision_result = self._get_operation_result()
        assert setup_result == 0
        assert setup_time_with_both_cache
        assert precision_result

        logging.info(f"setup_time_with_both_cache:{setup_time_with_both_cache}")


    def _get_operation_result(self):
        logging.info(f"get operation result start.")

        atb_operation = torch.classes.OperationTorch.OperationTorch("ConcatOperation")
        atb_operation.set_param(json.dumps({"concatDim":1}))

        logging.info("atb operation setup1 start")
        setup_rep_json = atb_operation.setup(TestKernelCache.in_tensors, TestKernelCache.out_tensors)
        logging.info("atb operation setup1 end, setup_rep_json:" + setup_rep_json)

        logging.info("atb operation setup2 start")
        setup_rep_json = atb_operation.setup(TestKernelCache.in_tensors, TestKernelCache.out_tensors)
        logging.info("atb operation setup2 end, setup_rep_json:" + setup_rep_json)

        setup_rep_obj =  json.loads(setup_rep_json)

        golden_out_tensor = torch.cat((TestKernelCache.in_tensors[0], TestKernelCache.in_tensors[1]), dim=1)
        logging.info("atb operation execute start")
        out_tensors = atb_operation.execute(TestKernelCache.in_tensors)
        precision_result = torch.allclose(out_tensors[0].cpu(), golden_out_tensor.cpu(), rtol=0.001, atol=0.001)
        logging.info(f"atb operation execute end, precision_result:{precision_result}")

        return (setup_rep_obj["result"], setup_rep_obj["setup_time"], precision_result)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("main start")
    unittest.main()
    logging.info("main end")