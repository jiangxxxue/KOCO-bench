/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef SELFATTENTIONBINDER_H
#define SELFATTENTIONBINDER_H
#include "hosttensor_binder.h"
#include <vector>

class SelfAttentionBinder : public HostTensorBinder {
public:
    SelfAttentionBinder();
    virtual ~SelfAttentionBinder();
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;
    void SetTensorList(const std::vector<atb::Tensor> &tensorList, const std::string &tensorListName) override;
 
private:
    void SetTensorListToPack(atb::VariantPack &variantPack);
    void BindTensorPrefixEncoder(atb::VariantPack &variantPack);

private:
    std::vector<int32_t> tokenOffset_;
    std::vector<int32_t> seqLen_;
    std::vector<int32_t> kvSeqLen_;
    std::vector<atb::Tensor> kCache_;
    std::vector<atb::Tensor> vCache_;
    bool byPass_ = false;
    bool useLogN_ = false;
    bool useQKVQuantOffline_ = false;
    bool useQKVQuantOnline_ = false;
    bool useSwa_ = false;
    bool useCausalMask_ = false;
    bool usePrefixEncoder_ = false;
    bool kvCacheWithParam_ = false;
    atb::infer::SelfAttentionParam::CalcType calcType_ = atb::infer::SelfAttentionParam::UNDEFINED;
};

#endif