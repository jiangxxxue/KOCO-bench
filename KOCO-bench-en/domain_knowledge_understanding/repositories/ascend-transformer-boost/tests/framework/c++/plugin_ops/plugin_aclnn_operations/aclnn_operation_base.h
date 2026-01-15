/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TEST_ACLNN_OPERATION_BASE_H
#define TEST_ACLNN_OPERATION_BASE_H

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <atb/atb_infer.h>
#include <atb/types.h>
#include <atb/utils.h>
#include "atb/infer_op_params.h"
#include "atb/operation_infra.h"

// 对atb::tensor的一层封装
namespace atb {
struct AclnnTensor
{
public:
    atb::Tensor atbTensor; //
    aclTensor *tensor = nullptr;
    int tensorIdx = -1; // aclTensor在aclExecutor中的index
    bool needUpdateTensorDataPtr = false;
    atb::SVector<int64_t> strides = {};
};

// 保持与atb的算子的统一接口调用
class AclnnBaseOperation : public atb::OperationInfra
{
public:
    explicit AclnnBaseOperation(const std::string &opName);
    ~AclnnBaseOperation() override;
    std::string GetName() const override;

    // 仿atb接口，获取workspace的大小
    atb::Status Setup(const atb::VariantPack &variantPack, uint64_t &workspaceSize, atb::Context *context) override;

    // 仿atb接口，算子执行
    atb::Status Execute(const atb::VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
                        atb::Context *context) override;

    // 创建输入aclnntensor
    virtual atb::Status CreateAclnnVariantPack(const atb::VariantPack &variantPack) = 0;

    // 计算workspace大小
    virtual atb::Status SetAclnnWorkspaceExecutor() = 0;

    // 执行Aclnn op
    virtual atb::Status ExecuteAclnnOp(uint8_t *workspace, aclrtStream &stream) = 0;

    // 更新aclnn输入和输出tensor的地址
    atb::Status UpdateAclnnVariantPack(const atb::VariantPack &variantPack);

    std::string opName_;
    aclOpExecutor *aclExecutor_ = nullptr;
    atb::SVector<std::shared_ptr<AclnnTensor>> aclInTensors_;
    atb::SVector<std::shared_ptr<AclnnTensor>> aclOutTensors_;
    uint64_t workspaceSize_;
    int workspaceBlockId_ = -1;
};
}
#endif