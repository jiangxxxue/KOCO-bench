/*
* Copyright (c) 2024 Huawei Technologies Co., Ltd.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "aclnn_operation_base.h"
#include "atb/utils/log.h"

namespace atb {
AclnnBaseOperation::AclnnBaseOperation(const std::string &opName) : opName_(opName)
{}

AclnnBaseOperation::~AclnnBaseOperation()
{
    aclExecutor_ = nullptr;
}

std::string AclnnBaseOperation::GetName() const
{
    return opName_;
}

atb::Status AclnnBaseOperation::Setup(
    const atb::VariantPack &variantPack, uint64_t &workspaceSize, atb::Context *context)
{
    ATB_LOG(INFO) << opName_ << " setup start";

    // 调用子类，创建输入输出tensor，并存入VariantPack
    int ret = CreateAclnnVariantPack(variantPack);
    if (ret != 0) {
        ATB_LOG(ERROR) << opName_ << " call CreateAclnnVariantPack fail, error: " << std::to_string(ret);
        return atb::ERROR_INVALID_PARAM;
    }

    // 调用子类，获取Executor和Workspace
    ret = SetAclnnWorkspaceExecutor();
    if (ret != 0) {
        ATB_LOG(ERROR) << opName_ << " call CreateAclnnVaSetAclnnWorkspaceExecutorriantPack fail, error: "
            << std::to_string(ret);
        return atb::ERROR_INVALID_PARAM;
    }
    // 返回计算出的workspaceSize
    workspaceSize = workspaceSize_;
    ATB_LOG(INFO) << opName_ << " setup end";
    return ret;
}

atb::Status AclnnBaseOperation::Execute(
    const atb::VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize, atb::Context *context)
{
    ATB_LOG(INFO) << opName_ << " execute start";
    if (!context) {
        ATB_LOG(ERROR) << opName_ << " execute fail, context param is null";
        return atb::ERROR_INVALID_PARAM;
    }

    aclrtStream stream = GetExecuteStream(context);
    if (!stream) {
        ATB_LOG(ERROR) << opName_ << " execute fail, execute stream in context is null";
        return atb::ERROR_INVALID_PARAM;
    }

    // 更新数据传入的地址
    int ret = UpdateAclnnVariantPack(variantPack);
    if (ret != 0) {
        ATB_LOG(ERROR) << opName_ << " call UpdateAclnnVariantPack fail, error: " << std::to_string(ret);
        return atb::ERROR_CANN_ERROR;
    }

    ATB_LOG(INFO) << "Input workspaceSize " << std::to_string(workspaceSize) << " localCache workspaceSize " <<
             std::to_string(workspaceSize_);
    ret = ExecuteAclnnOp(workspace, stream); // 调用aclnn接口
    if (ret != 0) {
        ATB_LOG(ERROR) << opName_ << " call ExecuteAclnnOp fail, error: " << std::to_string(ret);
        return atb::ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << opName_ << " execute start";

    return ret;
}

atb::Status AclnnBaseOperation::UpdateAclnnVariantPack(const atb::VariantPack &variantPack)
{
    // 更新inTensor的device地址
    for (size_t i = 0; i < aclInTensors_.size(); ++i) {
        int ret = -1;
        if (!aclInTensors_[i]->needUpdateTensorDataPtr) {
            continue;
        }
        aclInTensors_[i]->atbTensor = variantPack.inTensors.at(i);
        ret = aclSetInputTensorAddr(aclExecutor_,
            aclInTensors_[i]->tensorIdx,
            aclInTensors_[i]->tensor,
            aclInTensors_[i]->atbTensor.deviceData);

        if (ret != 0) {
            ATB_LOG(ERROR) << "inTensor " << std::to_string(i) << " call UpdateAclTensorDataPtr fail, error: "
                << std::to_string(ret);
            return atb::ERROR_CANN_ERROR;
        }
    }

    // 更新outTensor的device地址
    for (size_t i = 0; i < aclOutTensors_.size(); ++i) {
        int ret = -1;
        if (!aclOutTensors_[i]->needUpdateTensorDataPtr) {
            continue;
        }
        aclOutTensors_[i]->atbTensor = variantPack.outTensors.at(i);
        ret = aclSetOutputTensorAddr(aclExecutor_,
            aclOutTensors_[i]->tensorIdx,
            aclOutTensors_[i]->tensor,
            aclOutTensors_[i]->atbTensor.deviceData);

        if (ret != 0) {
            ATB_LOG(ERROR) << "outTensor " << std::to_string(i) << " call UpdateAclTensorDataPtr fail, error: "
                << std::to_string(ret);
            return atb::ERROR_CANN_ERROR;
        }
    }
    return atb::NO_ERROR;
}
}