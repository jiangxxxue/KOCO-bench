/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "pagedattention_binder.h"

PagedAttentionBinder::PagedAttentionBinder() {}

PagedAttentionBinder::~PagedAttentionBinder() {}

void PagedAttentionBinder::ParseParam(const nlohmann::json &paramJson)
{
    contextLens_.clear();
    if (paramJson.contains("contextLens")) {
        for (auto item : paramJson["contextLens"]) {
            contextLens_.push_back(item.get<int32_t>());
        }
    }
    batchRunStatus_.clear();
    if (paramJson.contains("batchRunStatus")) {
        for (auto item : paramJson["batchRunStatus"]) {
            batchRunStatus_.push_back(item.get<int32_t>());
        }
    }
    qLens_.clear();
    if (paramJson.contains("qLens")) {
        for (auto item : paramJson["qLens"]) {
            qLens_.push_back(item.get<int32_t>());
        }
    }
    isMask_ = false;
    if (paramJson.contains("maskType")) {
        if (paramJson["maskType"] != 0) {
            isMask_ = true;
        }
    }
    isMla_ = false;
    if (paramJson.contains("mlaVHeadSize")) {
        if (paramJson["mlaVHeadSize"] > 0) {
            isMla_ = true;
        }
    }
}

void PagedAttentionBinder::BindTensor(atb::VariantPack &variantPack)
{
    uint32_t contextLensTensorId = 4;    // q,k,v,bolck,contex
    uint32_t batchRunStatusTensorId = 5; // q,k,v,bolck,contex,(mask),batch
    uint32_t qLensTensorId = 5;          // q,k,v,bolck,contex,(mask),qLens
    if (isMla_) {
        contextLensTensorId -= 1;
        batchRunStatusTensorId -= 1;
        qLensTensorId -= 1;
    }
    if (isMask_) {
        batchRunStatusTensorId += 1;
        qLensTensorId += 1;
    }
    if (batchRunStatus_.size() > 0) {
        variantPack.inTensors.at(batchRunStatusTensorId).hostData = batchRunStatus_.data();
    }
    if (qLens_.size() > 0) {
        variantPack.inTensors.at(qLensTensorId).hostData = qLens_.data();
    }
    variantPack.inTensors.at(contextLensTensorId).hostData = contextLens_.data();
}