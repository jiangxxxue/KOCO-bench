#!/bin/bash
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

handle_error(){
    rm -rf linear_parallel_generation
    rm -rf *.bin

    cd $current_dir
}

trap handle_error ERR

set -e

current_dir=$(pwd)

cd "$(dirname "$0")"

cxx_abi=$(python3 -c '
try:
    import torch
    print("1" if torch.compiled_with_cxx11_abi() else "0")
except ImportError:
    print("1")
')

echo "Using cxx_abi=$cxx_abi"

DEV_NUM=2

M=2
K=256
N=2

DATA_TYPE="FLOAT16"

DATA_TYPE_CPP=""
DATA_TYPE_PY=""

case ${DATA_TYPE} in
    FLOAT16)
        DATA_TYPE_CPP="aclDataType::ACL_FLOAT16"
        DATA_TYPE_PY="torch.float16"
        ;;
    BF16)
        DATA_TYPE_CPP="aclDataType::ACL_BF16"
        DATA_TYPE_PY="torch.bfloat16"
        ;;
    *)
        DATA_TYPE_CPP=""
        DATA_TYPE_PY=""
        ;;
esac

# 修改 DEV_NUM 的值
sed -i "s/^const int32_t DEV_NUM = .*;/const int32_t DEV_NUM = ${DEV_NUM};/" linear_parallel_generation.cpp
# 修改 M 的值
sed -i "s/^const int32_t M = .*;/const int32_t M = ${M};/" linear_parallel_generation.cpp
# 修改 K 的值
sed -i "s/^const int32_t K = .*;/const int32_t K = ${K};/" linear_parallel_generation.cpp
# 修改 N 的值
sed -i "s/^const int32_t N = .*;/const int32_t N = ${N};/" linear_parallel_generation.cpp
# 修改 DATA_TYPE 的值
sed -i "s/^const aclDataType DATA_TYPE = .*;/const aclDataType DATA_TYPE = ${DATA_TYPE_CPP};/" linear_parallel_generation.cpp

# 修改 DEV_NUM 的值
sed -i "s/^DEV_NUM = .*/DEV_NUM = ${DEV_NUM}/" linear_parallel_mc2_linear_reduce_scatter.py
# 修改 M 的值
sed -i "s/^M = .*/M = ${M}/" linear_parallel_mc2_linear_reduce_scatter.py
# 修改 K 的值
sed -i "s/^K = .*/K = ${K}/" linear_parallel_mc2_linear_reduce_scatter.py
# 修改 N 的值
sed -i "s/^N = .*/N = ${N}/" linear_parallel_mc2_linear_reduce_scatter.py
# 修改 DATA_TYPE 的值
sed -i "s/^DATA_TYPE = .*/DATA_TYPE = ${DATA_TYPE_PY}/" linear_parallel_mc2_linear_reduce_scatter.py

g++ -D_GLIBCXX_USE_CXX11_ABI=$cxx_abi -I "${ATB_HOME_PATH}/include" -I "${ASCEND_HOME_PATH}/include" -L "${ATB_HOME_PATH}/lib" -L "${ASCEND_HOME_PATH}/lib64" \
linear_parallel_generation.cpp -l atb -l ascendcl -l hccl -l nnopbase -l opapi -o linear_parallel_generation
./linear_parallel_generation

python linear_parallel_mc2_linear_reduce_scatter.py

rm -rf linear_parallel_generation
rm -rf *.bin

cd $current_dir