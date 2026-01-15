/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "event_operation.h"
#include <acl/acl.h>
#include "atb/utils.h"
#include "atb/utils/log.h"
#include "atb/utils/config.h"
#include "event_runner.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/op_param_funcs.h"
#include "atb/utils/operation_util.h"

namespace atb {
static const uint32_t IN_TENSOR_NUM = 0;
static const uint32_t OUT_TENSOR_NUM = 0;

bool ParamCheck(const atb::common::EventParam &opParam)
{
    if (opParam.operatorType < atb::common::EventParam::UNDEFINED ||
        opParam.operatorType > atb::common::EventParam::WAIT) {
        ATB_LOG(ERROR) << "operatorType: " << opParam.operatorType << " should be UNDEFINED, RECORD and WAIT";
        return false;
    }
    return true;
}

OPERATION_PARAM_FUNCS(EventOperation, common::EventParam)

EventOperation::EventOperation(const common::EventParam &param) : OperationBase("EventOperation"), param_(param) {}

EventOperation::~EventOperation() {}

uint32_t EventOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t EventOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status EventOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                      SVector<TensorDesc> &outTensorDescs) const
{
    if (inTensorDescs.size() != 0) {
        ATB_LOG(ERROR) << "EventOperation not need inTensors.";
        return ERROR_INVALID_TENSOR_NUM;
    }
    outTensorDescs = {};
    return NO_ERROR;
}

std::shared_ptr<Runner> EventOperation::CreateRunner(Context &context) const
{
    (void)context;
    ATB_LOG(INFO) << "start create runner";
    return std::make_shared<EventRunner>(param_);
}

nlohmann::json EventOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}

common::EventParam EventOperation::GetParam() const
{
    return param_;
}

void EventOperation::SetParam(const common::EventParam &param)
{
    param_ = param;
    if (!runner_) {
        ATB_LOG(WARN) << "EventOperation runner_ is nullptr, can not update the runner param";
    } else {
        runner_->SetParam(param);
    }
}
} // namespace atb