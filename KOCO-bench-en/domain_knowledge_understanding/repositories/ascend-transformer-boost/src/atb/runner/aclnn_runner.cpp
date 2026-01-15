/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/runner/aclnn_runner.h"
#include "atb/kernel_cache/aclnn_executor_cache.h"
#include "atb/utils/aclnn_util.h"
#include "atb/utils/log.h"
#include "atb/utils/singleton.h"

namespace atb {

AclnnRunner::AclnnRunner(const std::string &name, RunnerType runnerType) : Runner(name), runnerType_(runnerType) {}

AclnnRunner::~AclnnRunner() {}

Status AclnnRunner::SetupImpl(RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "aclnn runner setupImpl";
    if (!runnerVariantPack.context) {
        ATB_LOG(ERROR) << GetLogPrefix() << "context is not ContextBase, setup fail";
        return ERROR_INVALID_CONTEXT_ADDR;
    }
    Status ret = NO_ERROR;
    const std::string &opName = this->GetName();
    AclnnCacheSlot aclnnCacheSlot = {};
    // executorCache hit
    if (executorRepeatable_ &&
        GetSingleton<AclnnExecutorCache>().FetchCacheSlot(opName, runnerVariantPack, aclnnCacheSlot) == NO_ERROR) {
        if (!IsAclnnRunnerVariankPackEqual(this->aclnnVariantPack_, runnerVariantPack)) {
            ATB_LOG(INFO) << GetLogPrefix()
                          << "fetched cached runnerVariantPack not same as aclnnVariantPack_, build again";
            ret = BuildAclnnVariantPack(runnerVariantPack);
            if (ret != NO_ERROR) {
                ATB_LOG(ERROR) << GetLogPrefix() << "BuildAclnnVariantPack failed!";
                return ret;
            }
        }
        this->atbVariantPack_.workspaceBufferSize = aclnnCacheSlot.workspaceSize;
        this->aclnnExecutor_ = aclnnCacheSlot.executor;
        return NO_ERROR;
    }
    // cache miss，创建新的executor
    ATB_LOG(INFO) << GetLogPrefix() << "ExecutorCache miss, BuildAclnnVariantPack directly";
    ret = BuildAclnnVariantPack(runnerVariantPack);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "BuildAclnnVariantPack failed!";
        return ret;
    }
    aclnnStatus aclnnRet = ACL_SUCCESS;
    aclnnRet = SetAclNNWorkspaceExecutor();
    if (aclnnRet != ACL_SUCCESS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Atb aclnn op set workspace failed with return value: " << aclnnRet;
        return ERROR_CANN_ERROR;
    }
    ATB_LOG(INFO) << GetLogPrefix()
                  << "getWorkspace success, workspaceSize: " << this->atbVariantPack_.workspaceBufferSize
                  << ", workspace addr: " << this->atbVariantPack_.workspaceBuffer;
    aclnnRet = aclSetAclOpExecutorRepeatable(this->aclnnExecutor_.get());
    if (aclnnRet != 0) {
        // 设置算子可复用失败，标记cache中executor不可复用
        ATB_LOG(INFO) << this->GetName() << " call aclSetAclOpExecutorRepeatable fail with error code: " << aclnnRet;
        this->executorRepeatable_ = false;
    } else {
        // 设置算子可复用成功，标记cache中executor可复用
        ATB_LOG(INFO) << this->GetName() << " call aclSetAclOpExecutorRepeatable success: ";
        this->executorRepeatable_ = true;
    }
    aclnnCacheSlot = {this->atbVariantPack_.workspaceBufferSize, aclnnExecutor_};
    ret = GetSingleton<AclnnExecutorCache>().AddCacheSlot(opName, runnerVariantPack, aclnnCacheSlot);
    if (ret != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "AclnnExecutorCache update cache failed!";
    }
    ATB_LOG(INFO) << GetLogPrefix() << "AclnnExecutorCache AddCacheSlot success opName: " << opName
                  << ", runnerVariantPack: " << runnerVariantPack.ToString();
    return ret;
}

uint64_t AclnnRunner::GetWorkspaceBufferSizeImpl()
{
    return this->atbVariantPack_.workspaceBufferSize;
}

Status AclnnRunner::PreExecuteImpl(RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << GetLogPrefix() << "AclNNOpCacheUpdateAclNNVariantPack";
    for (size_t i = 0; i < this->aclnnVariantPack_.aclInTensors.size(); ++i) {
        int ret = -1;
        if (!this->aclnnVariantPack_.aclInTensors[i]->needUpdateTensorDataPtr) {
            continue;
        }
        this->aclnnVariantPack_.aclInTensors[i]->atbTensor = runnerVariantPack.inTensors.at(i);
        if (this->aclnnVariantPack_.aclInTensors[i]->tensorListidx == AclNNTensor::notInTensorList) {
            ret = aclSetInputTensorAddr(this->aclnnExecutor_.get(), this->aclnnVariantPack_.aclInTensors[i]->tensorIdx,
                                        this->aclnnVariantPack_.aclInTensors[i]->tensor,
                                        this->aclnnVariantPack_.aclInTensors[i]->atbTensor.deviceData);
        } else {
            ret = aclSetDynamicInputTensorAddr(
                this->aclnnExecutor_.get(), this->aclnnVariantPack_.aclInTensors[i]->tensorListidx,
                this->aclnnVariantPack_.aclInTensors[i]->tensorIdx,
                this->aclnnVariantPack_.aclInTensorList[this->aclnnVariantPack_.aclInTensors[i]->tensorListidx],
                this->aclnnVariantPack_.aclInTensors[i]->atbTensor.deviceData);
        }
        if (ret != 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << "inTensor " << i << " call UpdateAclTensorDataPtr fail, error: " << ret;
            return atb::ERROR_CANN_ERROR;
        }
    }

    for (size_t i = 0; i < this->aclnnVariantPack_.aclOutTensors.size(); ++i) {
        int ret = -1;
        if (!this->aclnnVariantPack_.aclOutTensors[i]->needUpdateTensorDataPtr) {
            continue;
        }
        this->aclnnVariantPack_.aclOutTensors[i]->atbTensor = runnerVariantPack.outTensors.at(i);
        if (this->aclnnVariantPack_.aclOutTensors[i]->tensorListidx == AclNNTensor::notInTensorList) {
            ret =
                aclSetOutputTensorAddr(this->aclnnExecutor_.get(), this->aclnnVariantPack_.aclOutTensors[i]->tensorIdx,
                                       this->aclnnVariantPack_.aclOutTensors[i]->tensor,
                                       this->aclnnVariantPack_.aclOutTensors[i]->atbTensor.deviceData);
        } else {
            ret = aclSetDynamicOutputTensorAddr(
                this->aclnnExecutor_.get(), this->aclnnVariantPack_.aclOutTensors[i]->tensorListidx,
                this->aclnnVariantPack_.aclOutTensors[i]->tensorIdx,
                this->aclnnVariantPack_.aclOutTensorList[this->aclnnVariantPack_.aclOutTensors[i]->tensorListidx],
                this->aclnnVariantPack_.aclOutTensors[i]->atbTensor.deviceData);
        }
        if (ret != 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << "outTensor " << i
                           << " call UpdateAclTensorDataPtr fail, error: " << ret;
            return atb::ERROR_CANN_ERROR;
        }
    }

    return atb::NO_ERROR;
}

void AclnnRunner::UpdateWorkspace(const RunnerVariantPack &runnerVariantPack)
{
    this->atbVariantPack_.workspaceBufferSize = runnerVariantPack.workspaceBufferSize;
    this->atbVariantPack_.workspaceBuffer = runnerVariantPack.workspaceBuffer;
}

Status AclnnRunner::ExecuteImpl(RunnerVariantPack &runnerVariantPack)
{
    UpdateWorkspace(runnerVariantPack);
    return LaunchAclnnKernel();
}

} // namespace atb
