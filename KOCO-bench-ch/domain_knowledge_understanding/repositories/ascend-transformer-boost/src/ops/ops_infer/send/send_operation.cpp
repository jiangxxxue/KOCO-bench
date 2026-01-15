/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "send_operation.h"
#include "atb/utils/config.h"
#include "send_hccl_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/log.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils/singleton.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const int32_t IN_TENSOR_NUM = 1;
static const int32_t OUT_TENSOR_NUM = 0;

template <> Status CreateOperation(const infer::SendParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (GetSingleton<Config>().Is310P()) {
        ATB_LOG(ERROR) << "send is not support in Atlas inference products";
        return ERROR_INVALID_PARAM;
    }
    if (static_cast<int>(opParam.destRank) == opParam.rank || static_cast<int>(opParam.destRank) >= opParam.rankSize) {
        ATB_LOG(ERROR) << "SendParam destRank should not equal to rank, and should be smaller than rankSize";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.backend != "hccl") {
        ATB_LOG(ERROR) << "backend is " << opParam.backend << "backend must either be hccl";
        return ERROR_INVALID_PARAM;
    }
    if (OperationUtil::DistributedInitCheck<infer::SendParam>(opParam) != NO_ERROR) {
        ATB_LOG(ERROR) << "SendOperation DistributedInitCheck failed";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) SendOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

SendOperation::SendOperation(const infer::SendParam &param) : OperationBase("SendOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("SendOperation");
}

SendOperation::~SendOperation() {}

uint32_t SendOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t SendOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status SendOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                     SVector<TensorDesc> &outTensorDescs) const
{
    ATB_LOG(INFO) << GetLogPrefix() << "inTensorDescs Size:" << inTensorDescs.size()
                  << "outTensorDescs Size:" << outTensorDescs.size();
    return NO_ERROR;
}

std::shared_ptr<Runner> SendOperation::CreateRunner(Context &context) const
{
    (void)context;
    if (param_.backend == "hccl") {
        if (param_.hcclComm == nullptr) {
            return std::make_shared<SendHcclRunner>(param_, !param_.rankTableFile.empty());
        } else {
            return std::make_shared<SendHcclRunner>(param_, param_.hcclComm);
        }
    }
    return std::shared_ptr<Runner>();
}

nlohmann::json SendOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb