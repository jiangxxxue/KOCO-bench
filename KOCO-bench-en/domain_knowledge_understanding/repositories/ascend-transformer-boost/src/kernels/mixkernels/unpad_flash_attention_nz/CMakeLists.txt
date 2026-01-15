# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set(unpad_flash_attention_nz_srcs
    ${CMAKE_CURRENT_LIST_DIR}/unpad_flash_attention_nz_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unpad_flash_attention_nz_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/flash_attention_nz_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/flash_attention_nz_tiling_dependency.cpp
)

add_operation(UnpadFlashAttentionNzOperation "${unpad_flash_attention_nz_srcs}")

add_kernel(flash_attention_prefill ascend310p cube
    op_kernel/unpad_flash_attention_prefill.cce
    UnpadFlashAttentionNzEncoderKernel)

add_kernel(flash_attention_prefill ascend910 cube
    op_kernel/unpad_flash_attention_prefill.cce
    UnpadFlashAttentionNzEncoderKernel)

add_kernel(flash_attention_decoder ascend310p cube
    op_kernel/unpad_flash_attention_decoder.cce
    UnpadFlashAttentionNzDecoderKernel)
