# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set(reshape_and_cache_srcs
    ${CMAKE_CURRENT_LIST_DIR}/reshape_and_cache_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reshape_and_cache_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/reshape_and_cache_tiling.cpp
)

add_operation(ReshapeAndCacheOperation "${reshape_and_cache_srcs}")

add_kernel(reshape_and_cache ascend910b vector
    op_kernel/reshape_and_cache_nd.cpp
    ReshapeAndCacheNdKernel)

add_kernel(reshape_and_cache_wins ascend910b vector
    op_kernel/reshape_and_cache_compress.cpp
    ReshapeAndCacheCompressKernel)

add_kernel(reshape_and_cache_wins_rope ascend910b vector
    op_kernel/reshape_and_cache_compress_rope.cpp
    ReshapeAndCacheCompressRopeKernel)

add_kernel(reshape_and_cache_nz ascend910b vector
    op_kernel/reshape_and_cache_nz.cpp
    ReshapeAndCacheNzKernel)

add_kernel(reshape_and_cache_nz ascend310p vector
    op_kernel/reshape_and_cache_nz.cpp
    ReshapeAndCacheNzKernel)

add_kernel(reshape_and_cache_nz ascend910 vector
    op_kernel/reshape_and_cache_nz.cpp
    ReshapeAndCacheNzKernel)

add_kernel(reshape_and_cache_siso ascend910b vector
    op_kernel/reshape_and_cache_nd_siso.cpp
    ReshapeAndCacheNdSisoKernel)

add_kernel(reshape_and_cache_nz ascend310b vector
    op_kernel/reshape_and_cache_nz.cpp
    ReshapeAndCacheNzKernel)

add_kernel(reshape_and_cache_omni_compress ascend910b vector
    op_kernel/reshape_and_cache_omni_compress.cpp
    ReshapeAndCacheOmniCompressKernel)