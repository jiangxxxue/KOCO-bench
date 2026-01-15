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
import random


parameters = {
    "ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE": [0, 1],
    "ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE": [0, 1],
    "ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE": [0, 1],
    "ATB_OPSRUNNER_SETUP_CACHE_ENABLE": [0, 1],
    "ATB_WORKSPACE_MEM_ALLOC_GLOBAL": [0, 1],
    "ATB_COMPARE_TILING_EVERY_KERNEL": [0, 1],
    "ATB_MATMUL_SHUFFLE_K_ENABLE": [0, 1],
    "ACLNN_CACHE_LIMIT": [1, 10000, 10000000],
    "LCCL_DETERMINISTIC": [0, 1]
}


if __name__ == "__main__":
    workspace_path = "/home/atb_csv_ops_test/ascend-transformer-boost-master/tests/high_level_test"
    with open(os.path.join(workspace_path, "EnvironmentCombinationSet/log_atb_envs.sh"), "w") as file:
        for key, val in parameters.items():
            line = "export " + key + "=" + str(random.choice(val))
            file.write(line + "\n")