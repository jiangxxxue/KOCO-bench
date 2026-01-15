# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set(rope_q_concat_srcs
    ${CMAKE_CURRENT_LIST_DIR}/rope_q_concat_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rope_q_concat_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/rope_q_concat_tiling.cpp
)

add_operation(RopeQConcatOperation "${rope_q_concat_srcs}")

add_kernel(rope_q_concat ascend910b vector
    op_kernel/rope_q_concat.cpp
    RopeQConcatKernel)
