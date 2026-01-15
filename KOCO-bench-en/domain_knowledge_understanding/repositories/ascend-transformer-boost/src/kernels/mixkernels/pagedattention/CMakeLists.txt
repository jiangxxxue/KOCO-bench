# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set(paged_attention_srcs
    ${CMAKE_CURRENT_LIST_DIR}/paged_attention_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/paged_attention_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/paged_attention_tiling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/paged_attention_tiling_dependency.cpp
)

add_operation(PagedAttentionOperation "${paged_attention_srcs}")

add_kernel(paged_attention_mask ascend910b mix
    op_kernel/paged_attention_mask_mix.cce
    PagedAttentionMaskNdKernel)

add_kernel(paged_multi_latent_attention_split_cache_mask ascend910b mix
    op_kernel/paged_MLAttention_split_cache_mask_mix.cce
    PagedMultiLatentAttentionSplitCacheMaskNdKernel)

add_kernel(paged_multi_latent_attention_mask ascend910b mix
    op_kernel/paged_MLAttention_combine_cache_mask_mix.cce
    PagedMultiLatentAttentionCombineCacheMaskNdKernel)

add_kernel(paged_multi_latent_attention_multi_token_prediction_mix ascend910b mix
    op_kernel/paged_MLAttention_multi_token_prediction.cce
    PagedMultiLatentAttentionMultiTokenPredictionMaskNdKernel)

add_kernel(paged_attention_decoder_mask ascend310p cube
    op_kernel/paged_attention_decoder_mask.cce
    PagedAttentionDecoderNzMaskKernel)

add_kernel(paged_attention_decoder_mask ascend910 cube
    op_kernel/paged_attention_decoder_mask.cce
    PagedAttentionDecoderNzMaskKernel)