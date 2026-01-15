/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATB_RELAY_ATTENTION_PARAM_H
#define ATB_RELAY_ATTENTION_PARAM_H

#include <mki/utils/SVector/SVector.h>
#include <mki/tensor.h>
#include <vector>
#include <acl/acl.h>
#include "atb/svector.h"
#include "atb/types.h"

namespace atb {
struct RelayAttentionVariantPackParam {
    std::vector<int32_t> seqLen;
    std::vector<int32_t> kvSeqLen;
    std::vector<int32_t> kvShareMap;
    std::vector<int32_t> kvShareLen;
    std::vector<Tensor> key;
    std::vector<Tensor> value;
    std::vector<Tensor> keyShare;
    std::vector<Tensor> valueShare;
    bool BuildFromTensor(const SVector<Mki::Tensor> &inTensors);
    bool HostDataCheck(const SVector<Mki::Tensor> &inTensors);
    void ReintCastShapeFix(const Mki::Tensor tensor, std::vector<atb::Tensor> &tensorList) const;
};
} // namespace atb
#endif