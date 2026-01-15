/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "llama65blayer_binder.h"
#include <atb/utils/log.h>
 
Llama65BLayerBinder::Llama65BLayerBinder() {}
 
Llama65BLayerBinder::~Llama65BLayerBinder() {}
 
void Llama65BLayerBinder::ParseParam(const nlohmann::json &paramJson)
{
    seqLen_.clear();
    if (paramJson.contains("seqLen")) {
        for (auto item : paramJson["seqLen"]) {
            seqLen_.push_back(item.get<int32_t>());
        }
    }
    tokenOffset_.clear();
    if (paramJson.contains("tokenOffset")) {
        for (auto item : paramJson["tokenOffset"]) {
            tokenOffset_.push_back(item.get<int32_t>());
        }
    }
}
 
void Llama65BLayerBinder::BindTensor(atb::VariantPack &variantPack)
{
    const uint32_t tokenOffsetTensorId = 12;
    const uint32_t seqLenTensorId = 13;
    variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
}