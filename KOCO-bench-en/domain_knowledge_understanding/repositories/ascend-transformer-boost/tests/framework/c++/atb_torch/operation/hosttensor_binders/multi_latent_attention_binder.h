/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MULTILATENTBINDER_H
#define MULTILATENTBINDER_H
#include "hosttensor_binder.h"
#include <iostream>
#include <vector>

class MultiLatentAttentionBinder : public HostTensorBinder {
public:
    MultiLatentAttentionBinder();
    virtual ~MultiLatentAttentionBinder();
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int32_t> contextLens_;
    std::vector<int32_t> qSeqlen_;
    bool isMask_ = false;
};

#endif