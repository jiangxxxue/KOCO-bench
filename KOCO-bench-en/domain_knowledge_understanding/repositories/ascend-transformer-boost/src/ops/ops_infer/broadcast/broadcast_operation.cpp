/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "broadcast_operation.h"
#include <atb/utils/log.h>
#include "atb/utils/config.h"
#include "broadcast_hccl_runner.h"
#include "broadcast_lccl_runner.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const int32_t IN_TENSOR_NUM = 1;
static const int32_t OUT_TENSOR_NUM = 0;

template <> Status CreateOperation(const infer::BroadcastParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (GetSingleton<Config>().Is310P()) {
        ATB_LOG(ERROR) << "Broadcast is not support in Atlas inference products";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.backend != "hccl" && opParam.backend != "lccl") {
        ATB_LOG(ERROR) << "backend is " << opParam.backend << "backend must either be hccl or lccl";
        return ERROR_INVALID_PARAM;
    }
    if (OperationUtil::DistributedInitCheck<infer::BroadcastParam>(opParam) != NO_ERROR) {
        ATB_LOG(ERROR) << "BroadcastParam DistributedInitCheck failed";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) BroadcastOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

BroadcastOperation::BroadcastOperation(const infer::BroadcastParam &param)
    : OperationBase("BroadcastOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("BroadcastOperation");
}

BroadcastOperation::~BroadcastOperation() {}

uint32_t BroadcastOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t BroadcastOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status BroadcastOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                          SVector<TensorDesc> &outTensorDescs) const
{
    ATB_LOG(INFO) << GetLogPrefix() << " inTensorDescs Size:" << inTensorDescs.size()
                  << "outTensorDescs Size:" << outTensorDescs.size();
    return NO_ERROR;
}

std::shared_ptr<Runner> BroadcastOperation::CreateRunner(Context &context) const
{
    (void)context;
    if (param_.backend == "hccl") {
        if (param_.hcclComm == nullptr) {
            return std::make_shared<BroadcastHcclRunner>(param_, !param_.rankTableFile.empty());
        } else {
            return std::make_shared<BroadcastHcclRunner>(param_, param_.hcclComm);
        }
    } else if (param_.backend == "lccl") {
        return std::make_shared<BroadcastLcclRunner>(param_, context);
    }
    return std::shared_ptr<Runner>();
}

nlohmann::json BroadcastOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb