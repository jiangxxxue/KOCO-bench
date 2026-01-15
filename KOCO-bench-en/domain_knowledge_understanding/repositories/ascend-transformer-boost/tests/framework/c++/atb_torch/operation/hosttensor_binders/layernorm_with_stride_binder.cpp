/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "layernorm_with_stride_binder.h"
#include "atb/utils/log.h"
static const uint64_t IN_TENSOR_COUNT_FOUR = 4;
static const uint64_t IN_TENSOR_COUNT_THREE = 3;
LayerNormWithStrideBinder::LayerNormWithStrideBinder() {}

LayerNormWithStrideBinder::~LayerNormWithStrideBinder() {}

void LayerNormWithStrideBinder::ParseParam(const nlohmann::json &paramJson)
{
    strides.clear();
    if (paramJson.contains("strides")) {
        for (auto item : paramJson["strides"]) {
            strides.push_back(item.get<int64_t>());
        }
    }
    if (paramJson.contains("offset")) {
        for (auto item : paramJson["offset"]) {
            offset.push_back(item.get<int64_t>());
        }
    }
}

void LayerNormWithStrideBinder::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(IN_TENSOR_COUNT_THREE).hostData = strides.data();
    variantPack.inTensors.at(IN_TENSOR_COUNT_FOUR).hostData = offset.data();
}