/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "relayattention_binder.h"
#include "atb/utils/log.h"
 
RelayAttentionBinder::RelayAttentionBinder() {}
 
RelayAttentionBinder::~RelayAttentionBinder() {}
 
void RelayAttentionBinder::ParseParam(const nlohmann::json &paramJson)
{
    tokenOffset_.clear();
    if (paramJson.contains("tokenOffset")) {
        for (auto item : paramJson["tokenOffset"]) {
            tokenOffset_.push_back(item.get<int32_t>());
        }
    }
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int32_t>());
    }
    kvShareMap_.clear();
    for (auto item : paramJson["kvShareMap"]) {
        kvShareMap_.push_back(item.get<int32_t>());
    }
    kvShareLen_.clear();
    for (auto item : paramJson["kvShareLen"]) {
        kvShareLen_.push_back(item.get<int32_t>());
    }
}
 
void RelayAttentionBinder::BindTensor(atb::VariantPack &variantPack)
{
    const uint32_t seqLenTensorId = 6; // 6: 设置seqLen的tensor位置
    variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    const uint32_t tokenOffsetTensorId = 7; // 7: 设置tokenOffset的tensor位置
    variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
    const uint32_t kvShareMapTensorId = 8; // 8: 设置kvShareMap的tensor位置
    variantPack.inTensors.at(kvShareMapTensorId).hostData = kvShareMap_.data();
    const uint32_t kvShareLenTensorId = 9; // 9: 设置kvShareLen的tensor位置
    variantPack.inTensors.at(kvShareLenTensorId).hostData = kvShareLen_.data();
    SetTensorListToPack(variantPack);
}
 
void RelayAttentionBinder::SetTensorList(const std::vector<atb::Tensor> &tensorList, const std::string &tensorListName)
{
    if (tensorListName == "key") {
        ATB_LOG(INFO) << "key SetTensorList";
        key_ = tensorList;
    }
    if (tensorListName == "value") {
        ATB_LOG(INFO) << "value SetTensorList";
        value_ = tensorList;
    }
    if (tensorListName == "keyShare") {
        ATB_LOG(INFO) << "keyShare SetTensorList";
        keyShare_ = tensorList;
    }
    if (tensorListName == "valueShare") {
        ATB_LOG(INFO) << "valueShare SetTensorList";
        valueShare_ = tensorList;
    }
}
 
void RelayAttentionBinder::SetTensorListToPack(atb::VariantPack &variantPack)
{
    const uint32_t keyTensor = 1;
    const uint32_t valueTensor = 2;
    const uint32_t keyShareTensor = 3;
    const uint32_t valueShareTensor = 4;
    if (key_.size() > 0) {
        variantPack.inTensors.at(keyTensor).hostData = &key_;
        variantPack.inTensors.at(keyTensor).deviceData = nullptr;
    }
    if (value_.size() > 0) {
        variantPack.inTensors.at(valueTensor).hostData = &value_;
        variantPack.inTensors.at(valueTensor).deviceData = nullptr;
    }
    if (keyShare_.size() > 0) {
        variantPack.inTensors.at(keyShareTensor).hostData = &keyShare_;
        variantPack.inTensors.at(keyShareTensor).deviceData = nullptr;
    }
    if (valueShare_.size() > 0) {
        variantPack.inTensors.at(valueShareTensor).hostData = &valueShare_;
        variantPack.inTensors.at(valueShareTensor).deviceData = nullptr;
    }
}