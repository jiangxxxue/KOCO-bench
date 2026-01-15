#!/bin/bash
#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

function get_cxx_abi_option()
{
    cxx_abi=""
    until [[ -z "$1" ]]
    do {
        case "$1" in
        "--cxx_abi=1")
            cxx_abi=1
            ;;
        "--cxx_abi=0")
            cxx_abi=0
            ;;
        esac
        shift
    }
    done
    if [[ "$cxx_abi" == "" ]]; then
        res=$(python3 -c "import torch" &> /dev/null || echo "torch_not_exist")
        if [[ "$res" == "torch_not_exist" ]]; then
            cxx_abi=1
        else
            if [[ $(python3 -c 'import torch; print(torch.compiled_with_cxx11_abi())') == "True" ]]; then
                cxx_abi=1
            else
                cxx_abi=0
            fi
        fi
    fi
}

set_env_path="${BASH_SOURCE[0]}"
if [[ -n "$ZSH_VERSION" ]]; then
    set_env_path="$0"
fi

if [[ -f "$set_env_path" ]] && [[ "$set_env_path" =~ 'set_env.sh' ]];then
    atb_path=$(cd $(dirname $set_env_path); pwd)
    get_cxx_abi_option "$@"
    export ATB_HOME_PATH="${atb_path}/cxx_abi_${cxx_abi}"
    export LD_LIBRARY_PATH=$ATB_HOME_PATH/lib:$ATB_HOME_PATH/examples:$ATB_HOME_PATH/tests/atbopstest:$LD_LIBRARY_PATH
    export PATH=$ATB_HOME_PATH/bin:$PATH

    #加速库环境变量
    export ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=0 #每个Kernel的Execute时就做同步
    export ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE=0 #每个Runner的Execute时就做同步
    export ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE=0 #每个Operation的Execute时就做同步
    export ATB_OPSRUNNER_SETUP_CACHE_ENABLE=1 #是否开启SetupCache，当检查到输入和输出没有变化时，不做setup
    export ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT=1 #本地缓存个数，支持范围1~1024
    export ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT=5 #全局缓存个数，支持范围1~1024
    export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=1 #0:暴力算法 1:block分配算法 2:有序heap算法 3:引入block合并(SOMAS算法退化版)
    export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1 #0:不开启 1:开启全局中间tensor内存分配
    export ATB_COMPARE_TILING_EVERY_KERNEL=0 #每个Kernel运行后，比较运行前和后的NPU上tiling内容是否变化
    export ATB_HOST_TILING_BUFFER_BLOCK_NUM=128 #Context内部HostTilingBuffer块数，通常使用默认值即可，可配置范围：最小128，最大1024
    export ATB_DEVICE_TILING_BUFFER_BLOCK_NUM=32 #Context内部DeviceTilingBuffer块数，通常使用默认值即可，可配置范围：最小32，最大1024
    export ATB_SHARE_MEMORY_NAME_SUFFIX="" #共享内存命名后缀，多用户同时使用通信算子时，需通过设置该值进行共享内存的区分
    export ATB_MATMUL_SHUFFLE_K_ENABLE=1 #Shuffle-K使能，默认开

    #算子库环境变量
    export LCCL_DETERMINISTIC=0 #LCCL确定性AllReduce(保序加)是否开启，0关闭，1开启。
    export LCCL_PARALLEL=0 #LCCL多通信域并行，0关闭，1开启。
    
else
    echo "There is no 'set_env.sh' to import"
fi
