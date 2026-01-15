# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set(toppsample_rand_srcs
${CMAKE_CURRENT_LIST_DIR}/toppsample_rand_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/toppsample_rand_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/toppsample_rand_tiling.cpp
)

add_operation(ToppsampleRandOperation "${toppsample_rand_srcs}")

add_kernel(toppsample_rand ascend910b vector
        op_kernel/toppsample_rand.cpp
        ToppsampleRandKernel)

add_kernel(toppsample_rand ascend310p vector
        op_kernel/toppsample_rand.cpp
        ToppsampleRandKernel)