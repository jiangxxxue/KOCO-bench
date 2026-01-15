# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set(mla_preprocess_srcs
    ${CMAKE_CURRENT_LIST_DIR}/mla_preprocess_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/mla_preprocess_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/mla_preprocess_tiling.cpp
)

add_operation(MlaPreprocessOperation "${mla_preprocess_srcs}")


add_kernel(MLA_preprocess_attention ascend910b mix
    op_kernel/mla_preprocess_mix.cce
    MLAPreprocessKernel)

add_kernel(MLA_preprocess_attention_bf16 ascend910b mix
    op_kernel/mla_preprocess_mix_bf16.cce
    MLAPreprocessBF16Kernel)