/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include "selfattention_binder.h"
#include "atb/utils/log.h"

namespace {
    const int CALCTYPE_PREFIX_ENCODER = 4;
}


SelfAttentionBinder::SelfAttentionBinder() {}

SelfAttentionBinder::~SelfAttentionBinder() {}

void SelfAttentionBinder::ParseParam(const nlohmann::json &paramJson)
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
    if (paramJson.contains("CalcType")) {
        calcType_ = static_cast<atb::infer::SelfAttentionParam::CalcType>(paramJson["CalcType"]);
    }
    if (calcType_ == atb::infer::SelfAttentionParam::PREFIX_ENCODER) {
        usePrefixEncoder_ = true;
    }
    kvSeqLen_.clear();
    if (paramJson.contains("kvSeqLen")) {
        for (auto item : paramJson["kvSeqLen"]) {
            kvSeqLen_.push_back(item.get<int32_t>());
        }
    }
    if (paramJson.contains("byPass")) {
        if (paramJson["byPass"] == "true") {
            byPass_ = true;
        }
    }
    useLogN_ = false;
    if (paramJson.contains("scaleType")) {
        if (paramJson["scaleType"] == atb::infer::SelfAttentionParam::SCALE_TYPE_LOGN) {
            useLogN_ = true;
        }
    }
    kvCacheWithParam_ = false;
    if (paramJson.contains("kvCacheWithParam")) {
        if (paramJson["kvCacheWithParam"]) {
            kvCacheWithParam_ = true;
        }
    }
    useQKVQuantOffline_ = false;
    useQKVQuantOnline_ = false;
    if (paramJson.contains("quantType")) {
        if (paramJson["quantType"] == atb::infer::SelfAttentionParam::TYPE_QUANT_QKV_OFFLINE) {
            useQKVQuantOffline_ = true;
        }else if (paramJson["quantType"] == atb::infer::SelfAttentionParam::TYPE_QUANT_QKV_ONLINE) {
            useQKVQuantOnline_ = true;
        }
    }
    if (paramJson.contains("maskType")) {
        if (paramJson["maskType"] == atb::infer::SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_NORM ||
            paramJson["maskType"] == atb::infer::SelfAttentionParam::MASK_TYPE_SLIDING_WINDOW_COMPRESS) {
            useSwa_ = true;
        }
        ATB_LOG(INFO) << "paramJson[maskType]:" << paramJson["maskType"];
        if (paramJson["maskType"] == atb::infer::SelfAttentionParam::MASK_TYPE_CAUSAL_MASK) {
            useCausalMask_ = true;
        }
    }
}

void SelfAttentionBinder::BindTensorPrefixEncoder(atb::VariantPack &variantPack)
{
    // query, key, value, blockTables, mask, qSeqLen, kvSeqLen, slopes
    const uint32_t kSeqLenTensorId = useCausalMask_ ? 4 : 5;
    ATB_LOG(INFO) << "useCausalMask_:" << useCausalMask_;
    variantPack.inTensors.at(kSeqLenTensorId).hostData = seqLen_.data();
    variantPack.inTensors.at(kSeqLenTensorId + 1).hostData = kvSeqLen_.data();
}

void SelfAttentionBinder::BindTensor(atb::VariantPack &variantPack)
{
    if (usePrefixEncoder_) {
        return BindTensorPrefixEncoder(variantPack);
    }
    int32_t baseIdx = 0;
    uint32_t tokenOffsetTensorId = 0;
    uint32_t seqLenTensorId = 0;
    if (useLogN_) baseIdx++; // szie judgement correct
    if (useQKVQuantOffline_) baseIdx += 5; // 5: pa encoder with offline QKV quant
    if (useQKVQuantOnline_) baseIdx += 4; // 4: pa encoder with online QKV quant
    if (variantPack.inTensors.size() == (baseIdx + 3) || // 3: pa encoder MLA without mask
        variantPack.inTensors.size() == (baseIdx + 4) || // 4: pa encoder without mask
        variantPack.inTensors.size() == (baseIdx + 5)) { // 5: pa encoder input num
        seqLenTensorId = variantPack.inTensors.size() - 1 - baseIdx;
    } else if (variantPack.inTensors.size() == (baseIdx + 6)) { //6: pa encoder with mask & slopes
        if (useSwa_) {
            tokenOffsetTensorId = 3; // 3: 设置tokenOffset的tensor位置
            seqLenTensorId = 4;      // 4: 设置seqLen的tensor位置

        } else {
            seqLenTensorId = 4;      // 4: 设置seqLen的tensor位置
            if (byPass_) {
                SetTensorListToPack(variantPack);
            }
        }
    } else if (variantPack.inTensors.size() == (baseIdx + 7)) { //7: bypass
        tokenOffsetTensorId = 4; // 4: 设置tokenOffset的tensor位置
        seqLenTensorId = 5;      // 5: 设置seqLen的tensor位置
        if (byPass_) {
            SetTensorListToPack(variantPack);
        }
    } else if (variantPack.inTensors.size() == 8) { //8: Deocoder norm without mask
        tokenOffsetTensorId = 5; // 5: 设置tokenOffset的tensor位置
        seqLenTensorId = 6;      // 6: 设置seqLen的tensor位置
        if (byPass_) {
            SetTensorListToPack(variantPack);
        }
    }
    else {
        tokenOffsetTensorId = 6; // 6: 设置tokenOffset的tensor位置
        seqLenTensorId = 7;      // 7: 设置seqLen的tensor位置
        if (byPass_) {
            SetTensorListToPack(variantPack);
        }
    }
    if (tokenOffsetTensorId > 0) {
        variantPack.inTensors.at(tokenOffsetTensorId).hostData = tokenOffset_.data();
        if (kvCacheWithParam_) {
            variantPack.inTensors.at(tokenOffsetTensorId).deviceData = nullptr;
        }
    }
    variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    if (kvCacheWithParam_) {
        variantPack.inTensors.at(seqLenTensorId).deviceData = nullptr;
    }
}

void SelfAttentionBinder::SetTensorList(const std::vector<atb::Tensor> &tensorList, const std::string &tensorListName)
{
    if (tensorListName == "kCache") {
        ATB_LOG(INFO) << "kCache SetTensorList";
        kCache_ = tensorList;
    }
    if (tensorListName == "vCache") {
        ATB_LOG(INFO) << "vCache SetTensorList";
        vCache_ = tensorList;
    }
}
 
void SelfAttentionBinder::SetTensorListToPack(atb::VariantPack &variantPack)
{
    const uint32_t kCacheTensor = 1;
    const uint32_t vCacheTensor = 2;
    if (kCache_.size() > 0) {
        variantPack.inTensors.at(kCacheTensor).hostData = &kCache_;
        variantPack.inTensors.at(kCacheTensor).deviceData = nullptr;
    }
    if (vCache_.size() > 0) {
        variantPack.inTensors.at(vCacheTensor).hostData = &vCache_;
        variantPack.inTensors.at(vCacheTensor).deviceData = nullptr;
    }
}