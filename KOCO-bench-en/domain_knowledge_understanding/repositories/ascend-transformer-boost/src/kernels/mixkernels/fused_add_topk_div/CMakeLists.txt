# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
 
set(fused_add_topk_div_srcs
    ${CMAKE_CURRENT_LIST_DIR}/fused_add_topk_div_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fused_add_topk_div_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/fused_add_topk_div_tiling.cpp
)
 
add_operation(FusedAddTopkDivOperation "${fused_add_topk_div_srcs}")
 
add_kernel(fused_add_topk_div ascend910b vector
    op_kernel/fused_add_topk_div.cpp
    FusedAddTopkDivKernel)