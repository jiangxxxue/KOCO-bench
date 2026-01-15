# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set(unpad_flash_attention_srcs
    ${CMAKE_CURRENT_LIST_DIR}/unpad_flash_attention_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unpad_flash_attention_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/flash_attention_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/flash_attention_tiling_dependency.cpp
)

add_operation(UnpadFlashAttentionOperation "${unpad_flash_attention_srcs}")


add_kernel(unpad_flashattention ascend910b mix
    op_kernel/unpad_flash_attention_mix.cce
    UnpadFlashAttentionNdKernel)

add_kernel(unpad_flashattention_bf16 ascend910b mix
    op_kernel/unpad_flashattention_bf16_mix.cce
    UnpadFlashAttentionBF16NdKernel)

add_kernel(unpad_flashattention_bf16_prefix_cache ascend910b mix
    op_kernel/unpad_flashattention_bf16_prefix_cache.cce
    UnpadFlashAttentionBF16PrefixCacheNdKernel)
    
add_kernel(unpad_flashattention_decoder ascend910b mix
    op_kernel/unpad_flash_attention_decoder_hp_mix.cce
    UnpadFlashAttentionDecoderNdKernel)

add_kernel(unpad_flashattention_mla ascend910b mix
    op_kernel/unpad_flash_attention_mla_mix.cce
    UnpadFlashAttentionMlaNdKernel)

add_kernel(unpad_flashattention_mla_bf16 ascend910b mix
    op_kernel/unpad_flash_attention_mla_bf16_mix.cce
    UnpadFlashAttentionMlaBF16NdKernel)

add_kernel(unpad_flashattention_mla_decoder ascend910b mix
    op_kernel/unpad_flash_attention_mla_decoder_hp_mix.cce
    UnpadFlashAttentionMlaDecoderNdKernel)
 
add_kernel(multi_latent_attention_mix ascend910b mix
    op_kernel/multi_latent_attention_mix.cce
    MultiLatentAttentionEncoderCombineCacheKernel)

add_kernel(relay_attention_decoder_mix ascend910b mix
    op_kernel/relay_attention_decoder_mix.cce
    RelayAttentionKernel)

add_kernel(unpad_flashattention_razor_fusion ascend910b mix
    op_kernel/unpad_flash_attention_razor_fusion.cce
    UnpadFlashAttentionRazorFusionKernel)