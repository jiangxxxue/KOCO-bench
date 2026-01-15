/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "multi_latent_attention_binder.h"
#include "atb/utils/log.h"

MultiLatentAttentionBinder::MultiLatentAttentionBinder() {}

MultiLatentAttentionBinder::~MultiLatentAttentionBinder() {}

void MultiLatentAttentionBinder::ParseParam(const nlohmann::json &paramJson)
{
    contextLens_.clear();
    if (paramJson.contains("contextLens")) {
        for (auto item : paramJson["contextLens"]) {
            contextLens_.push_back(item.get<int32_t>());
        }
    }
    qSeqlen_.clear();
    if (paramJson.contains("qSeqlen")) {
        for (auto item : paramJson["qSeqlen"]) {
            qSeqlen_.push_back(item.get<int32_t>());
        }
    }
    isMask_ = false;
    if (paramJson.contains("maskType")) {
        if (paramJson["maskType"] != 0) {
            isMask_ = true;
        }
    }
}

void MultiLatentAttentionBinder::BindTensor(atb::VariantPack &variantPack)
{
    uint32_t contextLensTensorId = 5; // q,qr,kv,kvq,bolck,contex
    variantPack.inTensors.at(contextLensTensorId).hostData = contextLens_.data();
    uint32_t qSeqlenTensorId = 6; // q,qr,kv,kvq,bolck,contex,(mask),qseqlen
    if (isMask_) {
        qSeqlenTensorId++;
    }
    if (qSeqlen_.size() > 0) {
        variantPack.inTensors.at(qSeqlenTensorId).hostData = qSeqlen_.data();
    }
}