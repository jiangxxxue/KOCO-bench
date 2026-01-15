/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reduce_scatterv_binder.h"
#include "atb/utils/log.h"

ReduceScatterVBinder::ReduceScatterVBinder() {}

ReduceScatterVBinder::~ReduceScatterVBinder() {}

void ReduceScatterVBinder::ParseParam(const nlohmann::json &paramJson)
{
    sendCounts.clear();
    if (paramJson.contains("sendCounts")) {
        for (auto item : paramJson["sendCounts"]) {
            sendCounts.push_back(item.get<int64_t>());
        }
    }
    for(auto array: sendCounts) {
        std::cout << array << " ";
    }
    std::cout << std::endl;
    sdispls.clear();
    if (paramJson.contains("sdispls")) {
        for (auto item : paramJson["sdispls"]) {
            sdispls.push_back(item.get<int64_t>());
        }
    }
    recvCount = paramJson["recvCount"].get<int64_t>();
}

void ReduceScatterVBinder::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(1).hostData = sendCounts.data();
    variantPack.inTensors.at(2).hostData = sdispls.data();
    variantPack.inTensors.at(3).hostData = &recvCount;
}