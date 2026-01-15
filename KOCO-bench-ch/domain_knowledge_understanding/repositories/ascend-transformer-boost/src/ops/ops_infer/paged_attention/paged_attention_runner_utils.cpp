/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/utils/utils_internal.h"
#include "paged_attention_runner_utils.h"

namespace atb {
bool IsParamEqual(const infer::PagedAttentionParam &left, const infer::PagedAttentionParam &right)
{
    return left.headNum == right.headNum && UtilsInternal::IsFloatEqual(left.qkScale, right.qkScale) &&
           left.kvHeadNum == right.kvHeadNum && left.maskType == right.maskType &&
           left.batchRunStatusEnable == right.batchRunStatusEnable && left.quantType == right.quantType &&
           left.hasQuantOffset == right.hasQuantOffset && left.compressType == right.compressType &&
           left.calcType == right.calcType && left.scaleType == right.scaleType &&
           left.mlaVHeadSize == right.mlaVHeadSize;
}
} // namespace atb