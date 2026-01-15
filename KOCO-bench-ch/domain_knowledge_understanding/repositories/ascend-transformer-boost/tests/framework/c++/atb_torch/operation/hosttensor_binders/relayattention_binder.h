/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RELAYATTENTIONBINDER_H
#define RELAYATTENTIONBINDER_H
#include "hosttensor_binder.h"
#include <vector>
 
class RelayAttentionBinder : public HostTensorBinder {
public:
    RelayAttentionBinder();
    virtual ~RelayAttentionBinder();
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;
    void SetTensorList(const std::vector<atb::Tensor> &tensorList, const std::string &tensorListName) override;
 
private:
    void SetTensorListToPack(atb::VariantPack &variantPack);
 
private:
    std::vector<int32_t> tokenOffset_;
    std::vector<int32_t> seqLen_;
    std::vector<int32_t> kvShareMap_;
    std::vector<int32_t> kvShareLen_;
    std::vector<atb::Tensor> key_;
    std::vector<atb::Tensor> value_;
    std::vector<atb::Tensor> keyShare_;
    std::vector<atb::Tensor> valueShare_;
};
 
#endif