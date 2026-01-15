# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set(paged_cache_load_srcs
    ${CMAKE_CURRENT_LIST_DIR}/paged_cache_load_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/paged_cache_load_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/paged_cache_load_tiling.cpp
)

add_operation(PagedCacheLoadOperation "${paged_cache_load_srcs}")

add_kernel(paged_cache_load_nz ascend910b vector
    op_kernel/paged_cache_load_nz.cpp
    PagedCacheLoadNzKernel)

add_kernel(paged_cache_load_nd ascend910b vector
    op_kernel/paged_cache_load_nd.cpp
    PagedCacheLoadNdKernel)