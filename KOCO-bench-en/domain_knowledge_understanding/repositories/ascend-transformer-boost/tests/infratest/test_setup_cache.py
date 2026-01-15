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
import multiprocessing
import task_runner
import infratest_utils


class TestSetupCache(unittest.TestCase):
    """
    kernelCache缓存性能测试:
    0、不开启缓存
    1、开启本地缓存功能时，同op，setup入参一样，setup运行时间相交不开启缓存功能，时间缩短
    2、开启全局缓存功能时，同op，setup入参一样，setup运行时间相交不开启缓存功能，时间缩短
    3、开启本地+全局缓存功能时，同op，setup入参一样，setup运行时间相交不开启缓存功能，时间缩短
    """
    @classmethod
    def setUpClass(cls):
        TestSetupCache.in_tensors = [torch.randn(16, 32, dtype=torch.float16).npu(), torch.randn(16, 32, dtype=torch.float16).npu()]
        TestSetupCache.out_tensors = [torch.randn(32, 32, dtype=torch.float16).npu()]

    def test_performance(self):
        os.putenv("ATB_OPSRUNNER_SETUP_CACHE_ENABLE", "0")
        setup_result, setup_time_without_cache, precision_result = self._get_operation_result()
        logging.info(f"setup_result:{setup_result}, setup_time_without_cache:{setup_time_without_cache}, precision_result:{precision_result}")
        assert setup_result == 0
        assert precision_result

        os.putenv("ATB_OPSRUNNER_SETUP_CACHE_ENABLE", "1")
        setup_result, setup_time_with_cache, precision_result = self._get_operation_result()
        logging.info(f"setup_result:{setup_result}, setup_time_with_cache:{setup_time_with_cache}, precision_result:{precision_result}")
        assert setup_result == 0
        assert precision_result
    
    def _get_operation_result(self):
        logging.info(f"get operation result start.")

        atb_operation = torch.classes.OperationTorch.OperationTorch("ConcatOperation")
        atb_operation.set_param(json.dumps({"concatDim":1}))

        logging.info("atb operation setup1 start")
        setup_rep_json = atb_operation.setup(TestSetupCache.in_tensors, TestSetupCache.out_tensors)
        logging.info("atb operation setup1 end, setup_rep_json:" + setup_rep_json)

        logging.info("atb operation setup2 start")
        setup_rep_json = atb_operation.setup(TestSetupCache.in_tensors, TestSetupCache.out_tensors)
        logging.info("atb operation setup2 end, setup_rep_json:" + setup_rep_json)

        setup_rep_obj =  json.loads(setup_rep_json)

        golden_out_tensor = torch.cat((TestSetupCache.in_tensors[0], TestSetupCache.in_tensors[1]), dim=1)
        logging.info("atb operation execute start")
        out_tensors = atb_operation.execute(TestSetupCache.in_tensors)
        precision_result = torch.allclose(out_tensors[0].cpu(), golden_out_tensor.cpu(), rtol=0.001, atol=0.001)
        logging.info(f"atb operation execute end, precision_result:{precision_result}")

        return (setup_rep_obj["result"], setup_rep_obj["setup_time"], precision_result)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("main start")
    unittest.main()
    logging.info("main end")