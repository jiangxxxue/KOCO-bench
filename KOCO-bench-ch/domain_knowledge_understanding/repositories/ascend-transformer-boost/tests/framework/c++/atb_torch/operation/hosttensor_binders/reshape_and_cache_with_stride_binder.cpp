/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reshape_and_cache_with_stride_binder.h"
#include "atb/utils/log.h"
 
ReshapeAndCacheWithStrideBinder::ReshapeAndCacheWithStrideBinder() {}
 
ReshapeAndCacheWithStrideBinder::~ReshapeAndCacheWithStrideBinder() {}
 
void ReshapeAndCacheWithStrideBinder::ParseParam(const nlohmann::json &paramJson)
{
    kstrides.clear();
    vstrides.clear();
    koffset.clear();
    voffset.clear();
    is_SISO = false;
    if (paramJson.contains("kvCacheCfg") && paramJson["kvCacheCfg"] == 1) {
        if (paramJson.contains("kstrides")) {
            is_SISO = true;
            for (auto item : paramJson["kstrides"]) {
                kstrides.push_back(item.get<int64_t>());
            }
        }
        if (paramJson.contains("koffset")) {
            for (auto item : paramJson["koffset"]) {
                koffset.push_back(item.get<int64_t>());
            }
        }
    }else if (paramJson.contains("compressType")) {
        compressType = paramJson["compressType"];
        for (auto item : paramJson["kstrides"]) {
            kstrides.push_back(item.get<int64_t>());
        }
        for (auto item : paramJson["vstrides"]) {
            vstrides.push_back(item.get<int64_t>());
        }
        for (auto item : paramJson["koffset"]) {
            koffset.push_back(item.get<int64_t>());
        }
        for (auto item : paramJson["voffset"]) {
            voffset.push_back(item.get<int64_t>());
        }
    }
}
 
void ReshapeAndCacheWithStrideBinder::BindTensor(atb::VariantPack &variantPack)
{
    ATB_LOG(ERROR) << "BindTensor(atb::VariantPack &variantPack)";
    ATB_LOG(INFO) << "BindTensor(atb::VariantPack &variantPack)";
    ATB_LOG(DEBUG) << "BindTensor(atb::VariantPack &variantPack)";
    if (is_SISO) {
        variantPack.inTensors.at(3).hostData = kstrides.data();
        variantPack.inTensors.at(4).hostData = koffset.data();
    } else {
        if (compressType == 0) {
            variantPack.inTensors.at(5).hostData = kstrides.data();
            variantPack.inTensors.at(6).hostData = vstrides.data();
            variantPack.inTensors.at(7).hostData = koffset.data();
            variantPack.inTensors.at(8).hostData = voffset.data();
        } else if (compressType == 1) {
            variantPack.inTensors.at(7).hostData = kstrides.data();
            variantPack.inTensors.at(8).hostData = vstrides.data();
            variantPack.inTensors.at(9).hostData = koffset.data();
            variantPack.inTensors.at(10).hostData = voffset.data();
        } else if (compressType == 2) {
            variantPack.inTensors.at(8).hostData = kstrides.data();
            variantPack.inTensors.at(9).hostData = vstrides.data();
            variantPack.inTensors.at(10).hostData = koffset.data();
            variantPack.inTensors.at(11).hostData = voffset.data();
        }
    }
}