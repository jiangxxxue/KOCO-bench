/*
 * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATB_SELF_ATTENTION_PARAM_H
#define ATB_SELF_ATTENTION_PARAM_H

#include <mki/utils/SVector/SVector.h>
#include <mki/tensor.h>
#include <vector>
#include <acl/acl.h>
#include "atb/svector.h"
#include "atb/types.h"

namespace atb {
struct SelfAttentionFusionVariantPackParam {
    std::vector<int32_t> tokenOffset;
    std::vector<int32_t> seqLen;
    std::vector<int32_t> batchRunStatus;
    std::vector<int32_t> kvSeqLen;
    std::vector<Tensor> kCache;
    std::vector<Tensor> vCache;
    aclFormat attnMaskFormat = ACL_FORMAT_UNDEFINED;
    bool isUseTensorList = false;
    bool BuildFromTensor(const SVector<Mki::Tensor> &inTensors, bool hasBatch, bool hasMask, bool hasKV);
    bool BuildFromTensorEncoder(const SVector<Mki::Tensor> &inTensors, const size_t seqLenTensorId);
    bool BuildFromTensorPrefixEncoder(const SVector<Mki::Tensor> &inTensors, const size_t seqLenTensorId,
                                      const size_t kvSeqLenTensorId);
    void SetKVCacheParam(const SVector<Mki::Tensor> &inTensors);
    void SetTokenOffsetParam(const Mki::Tensor &tokenOffsetTensor);
};
} // namespace atb
#endif
