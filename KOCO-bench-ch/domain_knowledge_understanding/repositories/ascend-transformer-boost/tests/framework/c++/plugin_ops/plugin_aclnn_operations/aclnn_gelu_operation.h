/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TEST_ACLNN_GELU_OPERATION_H
#define TEST_ACLNN_GELU_OPERATION_H

#include "aclnn_operation_base.h"
namespace atb {
struct AclnnGeluParam
{
    int64_t geluApproximate = -1; // gelu_v2计算的入参，指定高斯近似算法，0: "none", 1: "tanh" , -1: 不使用gelu_v2
};

class GeluOperation : public AclnnBaseOperation
{
public:
    GeluOperation(const std::string &name, AclnnGeluParam param);
    atb::Status InferShape(
        const atb::SVector<atb::TensorDesc> &inTensorDesc, atb::SVector<atb::TensorDesc> &outTensorDesc) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

    atb::Status CreateAclnnVariantPack(const atb::VariantPack &variantPack) override;
    atb::Status SetAclnnWorkspaceExecutor() override;
    atb::Status ExecuteAclnnOp(uint8_t *workspace, aclrtStream &stream) override;

private:
    atb::Status CreateAclnnInTensor(const atb::VariantPack &variantPack);
    atb::Status CreateAclnnOutTensor(const atb::VariantPack &variantPack);
    std::shared_ptr<AclnnTensor> CreateAclnnTensor(atb::Tensor atbTensor, size_t tensorIdx);

    AclnnGeluParam param_;
};
}

#endif