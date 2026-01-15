# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set(moe_gmm_src
    ${CMAKE_CURRENT_LIST_DIR}/moe_gmm_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/moe_gmm_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/moe_gmm_tiling.cpp
)

add_operation(MoeGmmOperation "${moe_gmm_src}")

add_kernel(moe_gmm ascend910b mix op_kernel/moe_gmm.cce MoeGmmKernel)
add_kernel(moe_gmm_w8a8 ascend910b mix op_kernel/moe_gmm_w8a8.cce MoeGmmW8a8Kernel)