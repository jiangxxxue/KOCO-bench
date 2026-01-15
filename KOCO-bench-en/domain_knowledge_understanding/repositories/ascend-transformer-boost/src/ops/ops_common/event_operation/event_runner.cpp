/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "event_runner.h"
#include <atb/utils/log.h>
#include <acl/acl.h>
#include <asdops/params/params.h>
#include "atb/utils/tensor_util.h"
#include "atb/operation.h"

namespace atb {
EventRunner::EventRunner(const common::EventParam &param) : Runner("EventRunner"), param_(param)
{
    ATB_LOG(INFO) << "EventRunner called, param_.operatorType: " << param_.operatorType;
}

EventRunner::~EventRunner() {}

Status EventRunner::SetupImpl(RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << "EventRunner start setupImpl";
    if (!runnerVariantPack.context) {
        ATB_LOG(ERROR) << GetLogPrefix() << "context is not ContextBase, setup fail";
        return ERROR_INVALID_PARAM;
    }
    ATB_LOG(INFO) << "EventRunner setupImpl success";

    return NO_ERROR;
}

Status EventRunner::ExecuteImpl(RunnerVariantPack &runnerVariantPack)
{
    ATB_LOG(INFO) << "EventRunner start executeImpl";
    if (!runnerVariantPack.context) {
        ATB_LOG(ERROR) << GetLogPrefix() << "context is not ContextBase, execute fail";
        return ERROR_INVALID_PARAM;
    }
    if (param_.operatorType != atb::common::EventParam::UNDEFINED && !param_.event) {
        ATB_LOG(ERROR) << "param.event is null, please check the param.";
        return ERROR_INVALID_PARAM;
    }
    aclrtStream currentStream = GetExecuteStream(runnerVariantPack.context);
    uint32_t streamId = GetExecuteStreamId(operation_);
    if (param_.operatorType == common::EventParam::OperatorType::RECORD) {
        ATB_LOG(INFO) << GetLogPrefix() << "start recording event at StreamId " << streamId;
        aclError ret = aclrtRecordEvent(param_.event, currentStream);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << "aclrtRecordEvent fail, ret: " << ret;
            return ERROR_CANN_ERROR;
        }
        ATB_LOG(INFO) << "call aclrtRecordEvent success";
    }
    if (param_.operatorType == common::EventParam::OperatorType::WAIT) {
        ATB_LOG(INFO) << GetLogPrefix() << "start waiting for event at StreamId " << streamId;
        aclError ret = aclrtStreamWaitEvent(currentStream, param_.event);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << "aclrtStreamWaitEvent fail, ret: " << ret;
            return ERROR_CANN_ERROR;
        }
        ATB_LOG(INFO) << "call aclrtStreamWaitEvent success";
        ATB_LOG(INFO) << GetLogPrefix() << "start resetting event at StreamId " << streamId;
        ret = aclrtResetEvent(param_.event, currentStream);
        if (ret != ACL_SUCCESS) {
            ATB_LOG(ERROR) << "aclrtResetEvent fail, ret: " << ret;
            return ERROR_CANN_ERROR;
        }
        ATB_LOG(INFO) << "call aclrtResetEvent success";
    }
    ATB_LOG(INFO) << "EventRunner executeImpl success";

    return NO_ERROR;
}

void EventRunner::SetParam(const Mki::Any &param)
{
    ATB_LOG(INFO) << "EventRunner Start update the runner param";
    common::EventParam newParam = Mki::AnyCast<common::EventParam>(param);
    param_ = newParam;
    ATB_LOG(INFO) << "EventRunner update runner param success";
}
} // namespace atb