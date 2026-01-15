/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atb/operation/if_operation.h"
#include "atb/types.h"
#include "atb/utils/log.h"
#include "atb/operation/plugin_operation.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/common_utils.h"

namespace atb {

Status IfOperation::GetOperationFromCondition(Operation *&op) const
{
    bool cond = true;
    try {
        cond = param_.handle(param_.userData);
    } catch (const std::exception &e) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Get condition failed, please check handle function";
        return ERROR_INVALID_PARAM;
    }

    if (cond && param_.opA) {
        ATB_LOG(INFO) << GetLogPrefix() << "Condition met (true), selecting opA...";
        op = param_.opA;
    } else if (!cond && param_.opB) {
        ATB_LOG(INFO) << GetLogPrefix() << "Condition not met (false), selecting opB...";
        op = param_.opB;
    } else {
        ATB_LOG(ERROR) << GetLogPrefix() << "Please check the intended operation is valid, opA: " << param_.opA
                       << " opB: " << param_.opB;
        return ERROR_INVALID_PARAM;
    }
    return NO_ERROR;
}

static Status ParamCheck(const common::IfCondParam &param)
{
    if (!param.userData) {
        ATB_LOG(ERROR) << "userData is null, please check the param";
        return ERROR_INVALID_PARAM;
    }
    if (!param.handle) {
        ATB_LOG(ERROR) << "Handle is null, please check the param";
        return ERROR_INVALID_PARAM;
    }
    if (param.opA->GetInputNum() != param.opB->GetInputNum()) {
        ATB_LOG(ERROR) << "Input num of opA and opB are not equal, please check the param";
        return ERROR_INVALID_PARAM;
    }
    if (param.opA->GetOutputNum() != param.opB->GetOutputNum()) {
        ATB_LOG(ERROR) << "Output num of opA and opB are not equal, please check the param";
        return ERROR_INVALID_PARAM;
    }
    return NO_ERROR;
}

template <> Status CreateOperation(const common::IfCondParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        ATB_LOG(ERROR) << "Invalid param, operation is nullptr";
        return ERROR_INVALID_PARAM;
    }
    Status st = ParamCheck(opParam);
    if (st != NO_ERROR) {
        return st;
    }
    *operation = new (std::nothrow) IfOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "Failed to new conditional operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

IfOperation::IfOperation(const common::IfCondParam &param) : OperationBase("IfOperation"), param_(param) {}

IfOperation::~IfOperation() {}

std::string IfOperation::GetName() const
{
    return "IfOperation";
}

Status IfOperation::Setup(const VariantPack &variantPack, uint64_t &workspaceSize, Context *context)
{
    if (!opSelected_) {
        ATB_LOG(INFO) << GetLogPrefix() << "Operation not selected yet, setting opSelected_...";
    } else {
        ATB_LOG(WARN) << GetLogPrefix() << "Operation already selected, resetting opSelected_...";
    }
    Status st = GetOperationFromCondition(opSelected_);
    if (st != NO_ERROR) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Failed to select operation based on condition!";
    }
    ATB_LOG(INFO) << GetLogPrefix() << "Calling Setup...";
    return opSelected_->Setup(variantPack, workspaceSize, context);
}

Status IfOperation::Execute(const VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
                            Context *context)
{
    ATB_LOG(INFO) << GetLogPrefix() << "Calling Execute...";
    return opSelected_->Execute(variantPack, workspace, workspaceSize, context);
}

uint32_t IfOperation::GetInputNum() const
{
    ATB_LOG(INFO) << GetLogPrefix() << "Calling GetInputNum...";
    return param_.opA->GetInputNum();
}

uint32_t IfOperation::GetOutputNum() const
{
    ATB_LOG(INFO) << GetLogPrefix() << "Calling GetOutputNum...";
    return param_.opA->GetOutputNum();
}

void IfOperation::SetExecuteStreamId(uint32_t streamId)
{
    ATB_LOG(INFO) << GetLogPrefix() << "Calling SetExecuteStreamId...";
    Status st;
    if (param_.opA) {
        st = atb::SetExecuteStreamId(param_.opA, streamId);
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "Calling SetExecuteStreamId for opA failed!";
            return;
        }
        ATB_LOG(INFO) << GetLogPrefix() << "Setting execute streamId for opA success.";
    }
    if (param_.opB) {
        st = atb::SetExecuteStreamId(param_.opB, streamId);
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "Calling SetExecuteStreamId for opB failed!";
            return;
        }
        ATB_LOG(INFO) << GetLogPrefix() << "Setting execute streamId for opB success.";
    }
}

Status IfOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs, SVector<TensorDesc> &outTensorDescs) const
{
    ATB_LOG(INFO) << GetLogPrefix() << "Calling InferShape...";
    return param_.opA->InferShape(inTensorDescs, outTensorDescs);
}

std::shared_ptr<Runner> IfOperation::CreateRunner(Context &context) const
{
    if (!opSelected_) {
        ATB_LOG(INFO) << GetLogPrefix()
                      << "Operation not selected yet, executing create runner as part of graph, setting opSelected_...";
        Status st = GetOperationFromCondition(opSelected_);
        if (st != NO_ERROR) {
            ATB_LOG(ERROR) << GetLogPrefix() << "Failed to select operation based on condition!";
        }
    }
    OperationBase *opBase = dynamic_cast<OperationBase *>(opSelected_);
    if (!opBase) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Failed to convert Operation to OperationBase";
        return nullptr;
    }
    ATB_LOG(INFO) << GetLogPrefix() << "Calling CreateRunner...";
    return opBase->CreateRunner(context);
}
} // namespace atb