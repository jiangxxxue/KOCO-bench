# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set(kvcache_srcs
    ${CMAKE_CURRENT_LIST_DIR}/kvcache_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/kvcache_kernel.cpp
)

add_operation(KVCacheOperation "${kvcache_srcs}")

add_kernel(kvcache ascend910b vector
    op_kernel/kvcache.cpp
    KVCacheNdKernel)

add_kernel(kvcache_nz ascend310p vector
    op_kernel/kvcache_nz.cpp
    KVCacheNzKernel)

add_kernel(kvcache_dynamic_batch ascend910b vector
    op_kernel/kvcache_dynamic_batch.cpp
    KVCacheNdDynamicBatchKernel)

add_kernel(kvcache_params ascend910b vector
    op_kernel/kvcache_params.cpp
    KVCacheNdParamsKernel)

add_kernel(kvcache_nz_params ascend310p vector
    op_kernel/kvcache_nz_params.cpp
    KVCacheNzParamsKernel)

add_kernel(kvcache_dynamic_batch_params ascend910b vector
    op_kernel/kvcache_dynamic_batch_params.cpp
    KVCacheNdDynamicBatchParamsKernel)