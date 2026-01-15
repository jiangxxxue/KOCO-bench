/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cumsum_operation.h"
#include "atb/utils/log.h"
#include "cumsum_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
template <> Status CreateOperation(const infer::CumsumParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (opParam.axes.size() != 1) {
        ATB_LOG(ERROR) << "cumsumParam axes size must be 1";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.reverse) {
        ATB_LOG(ERROR) << "cumsum does not support reverse yet";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.exclusive) {
        ATB_LOG(ERROR) << "cumsum does not support exclusive yet";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) CumsumOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

CumsumOperation::CumsumOperation(const infer::CumsumParam &param) : OperationBase("CumsumOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("CumsumOperation");
}

CumsumOperation::~CumsumOperation() {}

uint32_t CumsumOperation::GetInputNum() const
{
    return 1;
}

uint32_t CumsumOperation::GetOutputNum() const
{
    return 1;
}

Status CumsumOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                       SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    return NO_ERROR;
}

Status CumsumOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    if (param_.axes.at(0) < 0 || param_.axes.at(0) >= static_cast<int64_t>(inTensorDescs.at(0).shape.dimNum)) {
        ATB_LOG(ERROR) << "cumsumParam axes value [" << param_.axes.at(0) << "] is invalid";
        return ERROR_INVALID_PARAM;
    }
    return NO_ERROR;
}

Status CumsumOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    if (param_.axes.at(0) < 0 || param_.axes.at(0) >= static_cast<int64_t>(inTensors.at(0).desc.shape.dimNum)) {
        ATB_LOG(ERROR) << "cumsumParam axes value [" << param_.axes.at(0) << "] is invalid";
        return ERROR_INVALID_PARAM;
    }
    ATB_LOG(DEBUG) << "outTensors size:" << outTensors.size();
    return NO_ERROR;
}

std::shared_ptr<Runner> CumsumOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<CumsumOpsRunner>(param_);
}

nlohmann::json CumsumOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb
