# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
 
set(rms_norm_and_rope_and_reshape_and_cache_srcs
    ${CMAKE_CURRENT_LIST_DIR}/rms_norm_and_rope_and_reshape_and_cache_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rms_norm_and_rope_and_reshape_and_cache_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/rms_norm_and_rope_and_reshape_and_cache_tiling.cpp
)
 
add_operation(RmsNormAndRopeAndReshapeAndCacheOperation "${rms_norm_and_rope_and_reshape_and_cache_srcs}")
 
add_kernel(rms_norm_and_rope_and_reshape_and_cache ascend910b vector
    op_kernel/rms_norm_and_rope_and_reshape_and_cache.cpp
    RmsNormAndRopeAndReshapeAndCacheKernel)