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
import unittest
import json
import sys
import torch
import torch_npu
import datetime
import glob
import psutil
import logging
import infratest_utils

def get_file_name_by_path(dir_path):
    if not os.path.exists(dir_path):
        return 0
    file_list = os.listdir(dir_path)
    file_num = len(file_list)
    return file_num

def check_file_exit(file_path):
    return os.path.exists(file_path)

def create_test_operation():
    torch.classes.OperationTorch.OperationTorch("FillOperation")

def check_not_available_space(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    disk_usage = psutil.disk_usage(dir_path)
    available_space = disk_usage.free / (1024 ** 3)  # 将字节转换为GB
    return available_space < 10

class TestLog(unittest.TestCase):
    def _init(self):
        os.environ['ASCEND_MODULE_LOG_LEVEL'] = 'ATB=1:' + os.environ.get('ASCEND_MODULE_LOG_LEVEL', '')

        self._log_prefix = 'atb'

        dir_path = "~"
        env_var = os.environ.get('HOME') + "/ascend/log"
        if env_var:
            dir_path = env_var
        env_var = os.environ.get('ASCEND_WORK_PATH')
        if env_var:
            dir_path = env_var
        env_var = os.environ.get('ASCEND_PROCESS_LOG_PATH')
        if env_var:
            dir_path = env_var

        self._dir_path = os.path.expanduser(dir_path + "/" + self._log_prefix + "/")
        self._max_file_num = 50

    def _reset(self):
        os.environ['ASCEND_MODULE_LOG_LEVEL'] = 'ATB=4:' + os.environ.get('ASCEND_MODULE_LOG_LEVEL', '')
        
    def _get_current_log_num(self):
        return get_file_name_by_path(self._dir_path)

    def _create_aging_24h_log_file(self, tid):
        now = datetime.datetime.now()  # 获取当前时间
        delta = datetime.timedelta(hours=24, seconds=1)  # 创建一个时间差对象，表示前24小时
        before_24h = now - delta  # 前24小时的
        before_24h_str = before_24h.strftime('%Y%m%d%H%M%S')
        aging_file_path = os.path.join(self._dir_path, self._log_prefix + "_" + tid + "_" + before_24h_str + ".log")
        print("aging_file_path: ", aging_file_path)
        with open(aging_file_path, "w") as f:
            f.write("This is a log file created before 24 hours.")
        return aging_file_path

    def _create_num_log_file(self, num):
        time = datetime.datetime.now() - datetime.timedelta(seconds=5)
        create_file_time_str = time.strftime('%Y%m%d%H%M%S')
        for i in range(1, num):
            new_file_path = os.path.join(self._dir_path, self._log_prefix + "_0" + str(i) + "_" + create_file_time_str + ".log")
            with open(new_file_path, "w") as f:
                f.write("This is a log file.")

    def _remove_create_num_log_file(self):
        for file_name in os.listdir(self._dir_path):
            if file_name.startswith(self._log_prefix + "_0") and file_name.endswith(".log"):
                os.remove(os.path.join(self._dir_path, file_name))

    def test_log_file(self):
        """
        日志文件测试
        1、程序运行正常创建的日志文件
        2、老化24小时前创建的日志文件
        3、最多保存50个日志文件
        """
        self._init()
        if check_not_available_space(self._dir_path):
            return 0
        start_file_num = self._get_current_log_num()
        print("初始文件数为：",start_file_num)
        now = datetime.datetime.now() - datetime.timedelta()
        # 创建24小时前的日志文件
        aging_file_1 = self._create_aging_24h_log_file("888888")
        aging_file_2 = self._create_aging_24h_log_file("888889")
        # 创建日志文件超过50个
        self._create_num_log_file(self._max_file_num - start_file_num + 3)
        self.assertTrue(self._get_current_log_num() > self._max_file_num)

        create_test_operation()

        self.assertFalse(check_file_exit(aging_file_1))
        self.assertFalse(check_file_exit(aging_file_2))

        self.assertTrue(self._get_current_log_num() <= self._max_file_num)
        # 删除多创建的文件
        self._remove_create_num_log_file()
        end_file_num = self._get_current_log_num()
        print("结束文件数为：", end_file_num)

        new_file_name_pattern = '*' + now.strftime('%Y%m%d%H%M%S') + '.log'
        self.assertTrue(len(glob.glob(self._dir_path + new_file_name_pattern)) > 0)

        self._reset()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("main start")
    unittest.main()
    logging.info("main end")