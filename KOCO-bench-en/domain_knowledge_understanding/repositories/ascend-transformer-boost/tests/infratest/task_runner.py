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
import logging
import torch

def _process_func(env_args, queue, func, func_args):
    logging.basicConfig(level=logging.INFO,format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    for env_name in env_args:
        logging.info(f"set env {env_name}={env_args[env_name]}")
        os.putenv(env_name, str(env_args[env_name]))

    queue.put(func(func_args))

def run_task_in_other_process(env_args, func, func_args):
    import multiprocessing
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_process_func, args=(env_args, queue, func, func_args,))
    p.start()
    p.join()
    p.terminate()
    return queue.get()
