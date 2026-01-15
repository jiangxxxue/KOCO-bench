/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "all_gatherv_binder.h"
#include "atb/utils/log.h"

AllGatherVBinder::AllGatherVBinder() {}

AllGatherVBinder::~AllGatherVBinder() {}

void AllGatherVBinder::ParseParam(const nlohmann::json &paramJson)
{
    sendCount = paramJson["sendCount"].get<int64_t>();
    recvCounts.clear();
    if (paramJson.contains("recvCounts")) {
        for (auto item : paramJson["recvCounts"]) {
            recvCounts.push_back(item.get<int64_t>());
        }
    }
    rdispls.clear();
    if (paramJson.contains("rdispls")) {
        for (auto item : paramJson["rdispls"]) {
            rdispls.push_back(item.get<int64_t>());
        }
    }
}

void AllGatherVBinder::BindTensor(atb::VariantPack &variantPack)
{
    variantPack.inTensors.at(1).hostData = &sendCount;
    variantPack.inTensors.at(2).hostData = recvCounts.data();
    variantPack.inTensors.at(3).hostData = rdispls.data();
}