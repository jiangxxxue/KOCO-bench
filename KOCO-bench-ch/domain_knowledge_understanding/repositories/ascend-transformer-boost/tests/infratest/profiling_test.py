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
import infratest_utils
import logging

class TestProfiling(unittest.TestCase):
    def _profiling_test_impl(self, search_name, experimental_config):
        current_dir = os.getcwd()
        if os.path.exists(f"{current_dir}/tmp"):
            os.system(f"rm -rf {current_dir}/tmp")

        prof = torch_npu.profiler.profile(
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f"{current_dir}/tmp"),
        experimental_config=experimental_config)
        prof.start()
        infratest_utils.add_operation_execution()
        prof.stop()
        msprof_path = f"{infratest_utils.ASCEND_HOME_PATH}/tools/profiler/profiler_tool/analysis/msprof/msprof.py"
        os.system(f"python3 {msprof_path} export timeline -dir {current_dir}/tmp > /dev/null")
        ret = os.system(
            f"grep_result=`grep -r '{search_name}' {current_dir}/tmp`\n\
            if [ -z \"$grep_result\" ];\n\
            then\n\
                exit 999\n\
            else\n\
                exit 0\n\
            fi")
        
        infratest_utils.delete_tmp()
        if ret != 0:
            return -1
        else:
            return 0

    def test_profiling(self):
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        )
        ret = self._profiling_test_impl("AddF32Tactic", experimental_config)
        if ret == 0:
            print("profiling_test1 success!")
        else:
            print("profiling_test1 error!")
        self.assertTrue(ret == 0)

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        )
        ret = self._profiling_test_impl("1,4", experimental_config)
        if ret == 0:
            print("profiling_test2 success!")
        else:
            print("profiling_test2 error!")
        self.assertTrue(ret == 0)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("main start")
    unittest.main()
    logging.info("main end")