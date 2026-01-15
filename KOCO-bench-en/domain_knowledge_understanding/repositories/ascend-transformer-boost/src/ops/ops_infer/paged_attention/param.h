/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_PAGED_ATTENTION_PARAM_H
#define ATB_PAGED_ATTENTION_PARAM_H

#include <mki/utils/SVector/SVector.h>
#include <mki/tensor.h>
#include <vector>
#include <acl/acl.h>
#include "atb/svector.h"

namespace atb {
struct PagedAttentionFusionVariantPackParam {
    std::vector<int32_t> batchRunStatus;
    std::vector<int32_t> contextLens;
    std::vector<int32_t> qLens;
    bool BuildFromTensor(const SVector<Mki::Tensor> &inTensors, bool batchRunStatusEnable, size_t batchRunStatusIndex,
                         bool needQLens, size_t qLensIndex, size_t contextLensTensorId);
    bool BuildFromTensor310P(const SVector<Mki::Tensor> &inTensors, bool needQLens, int qLensIndex);
    bool ParseQLens(const SVector<Mki::Tensor> &inTensors, int qLensIndex);
};
} // namespace atb
#endif