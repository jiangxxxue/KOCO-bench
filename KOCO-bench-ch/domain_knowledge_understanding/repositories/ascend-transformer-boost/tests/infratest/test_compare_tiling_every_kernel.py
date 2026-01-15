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
import datetime
import infratest_utils

os.environ['ASCEND_MODULE_LOG_LEVEL'] = 'ATB=1:' + os.environ.get('ASCEND_MODULE_LOG_LEVEL', '')

os.environ['ATB_COMPARE_TILING_EVERY_KERNEL'] = '1'

class TestCmpKernel(unittest.TestCase):
    def test(self):
        operation = torch.classes.OperationTorch.OperationTorch("RopeOperation")
        param = {"rotaryCoeff": 4}
        ntoken = 4
        seqlen = 4
        hidden_size = 4096
        head_size = 128
        intensor0 = torch.rand(ntoken, hidden_size).npu().half()
        intensor1 = torch.rand(ntoken, hidden_size).npu().half()
        intensor2 = torch.rand(ntoken, head_size).npu().half()
        intensor3 = torch.rand(ntoken, head_size).npu().half()
        intensor4 = torch.tensor([seqlen], dtype=torch.int32).npu()
        tensors = [intensor0, intensor1, intensor2, intensor3, intensor4]
        operation.set_param(json.dumps(param))
        operation.execute(tensors)
    
    def test_kernel_res(self):
        dir_path = "~"
        env_var = os.environ.get('HOME')
        if env_var:
            dir_path = env_var + "/ascend/log"
        env_var = os.environ.get('ASCEND_WORK_PATH')
        if env_var:
            dir_path = env_var
        env_var = os.environ.get('ASCEND_PROCESS_LOG_PATH')
        if env_var:
            dir_path = env_var
        log_path = os.path.expanduser(dir_path + "/atb/")
        files = os.listdir(log_path)
        latest_time = datetime.datetime.now()
        offset = datetime.timedelta(seconds=0.5)
        latest_file = ''
        for file in files:
            if file.endswith('.log'):
                modify_time = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(log_path, file)))
                if modify_time + offset > latest_time:
                    latest_time = modify_time + offset
                    latest_file = file
        
        if latest_file:
            with open(os.path.join(log_path, latest_file), 'r') as f:
                for line in f:
                    if 'change global device tiling' in line:
                        print('False')
                        break
                else:
                    print('True')
                    os.remove(log_path + latest_file)
                    os.environ['ASCEND_MODULE_LOG_LEVEL'] = 'ATB=4:' + os.environ.get('ASCEND_MODULE_LOG_LEVEL', '')
                    os.environ['ATB_COMPARE_TILING_EVERY_KERNEL'] = '0'
        else:
            print('No log file found.')


if __name__ == '__main__':
    unittest.main()
