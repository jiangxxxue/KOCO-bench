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
import unittest
import infratest_utils
import logging
import re

class TestMultiStream(unittest.TestCase):
    def _multi_stream_test_impl(self):
        os.putenv("ATB_PROFILING_ENABLE", "1")
        current_dir = os.getcwd()
        if os.path.exists(f"{current_dir}/tmp"):
            os.system(f"rm -rf {current_dir}/tmp")
        prof = torch.npu.profile(f"{current_dir}/tmp")
        prof.__enter__()

        infratest_utils.add_operation_execution()

        stream = torch.npu.current_stream()
        stream.synchronize()
        prof.__exit__(None, None, None)
        msprof_path = f"{infratest_utils.ASCEND_HOME_PATH}/tools/profiler/profiler_tool/analysis/msprof/msprof.py"
        os.system(f"python3 {msprof_path} export timeline -dir {current_dir}/tmp > /dev/null")

        copy_stream = 0
        execute_stream = 0
        for fpath, dirs, fs in os.walk(f"{current_dir}/tmp"):
            for f in fs:
                path = os.path.join(fpath, f)
                if "msprof_" in path:
                    with open(path, 'r') as load_file:
                        load_dict = json.load(load_file)
                        for block in load_dict:
                            if block["name"] == "MemcopyAsync":
                                copy_stream = block["args"]["Stream Id"]
                            if block["name"] == "AddF16Tactic":
                                execute_stream = block["args"]["Stream Id"]

        infratest_utils.delete_tmp()
        if copy_stream == execute_stream:
            return -1
        else:
            return 0

    def test_multi_stream(self):
        os.putenv("ATB_USE_TILING_COPY_STREAM", "1")
        device_name = torch.npu.get_device_name()
        if re.search("Ascend310P", device_name, re.I):
            ret = self._multi_stream_test_impl()
        else:
            ret = 0
        if ret != 0:
            print("CopyStream Error!")
        else:
            print("CopyStream Success!")
        self.assertTrue(ret == 0)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("main start")
    unittest.main()
    logging.info("main end")