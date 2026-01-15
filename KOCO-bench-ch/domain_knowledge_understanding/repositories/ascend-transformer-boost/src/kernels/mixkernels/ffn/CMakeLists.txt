# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set(ffn_srcs
    ${CMAKE_CURRENT_LIST_DIR}/ffn_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ffn_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/ffn_tiling.cpp
)

add_operation(FFNOperation "${ffn_srcs}")

add_kernel(ffn ascend310p cube
        op_kernel/ffn.cpp
        FFNKernel)