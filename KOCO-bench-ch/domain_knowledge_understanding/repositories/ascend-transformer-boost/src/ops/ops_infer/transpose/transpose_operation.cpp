/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "transpose_operation.h"
#include "transpose_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/singleton.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"

static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_NUM = 1;

namespace atb {
template <> Status CreateOperation(const infer::TransposeParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    *operation = new TransposeOperation(opParam);
    return NO_ERROR;
}

TransposeOperation::TransposeOperation(const infer::TransposeParam &param)
    : OperationBase("TransposeOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("TransposeOperation");
}

TransposeOperation::~TransposeOperation() {}

uint32_t TransposeOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t TransposeOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status TransposeOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                          SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; i++) {
        outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[param_.perm.at(i)];
    }
    return NO_ERROR;
}

Status TransposeOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    if (param_.perm.size() != inTensorDescs.at(0).shape.dimNum) {
        ATB_LOG(ERROR) << "input perm size error";
        return ERROR_INVALID_PARAM;
    }
    Status st = CheckPerm();
    if (st != NO_ERROR) {
        return st;
    }
    return NO_ERROR;
}

Status TransposeOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    if (param_.perm.size() != inTensors.at(0).desc.shape.dimNum) {
        ATB_LOG(ERROR) << "input perm size error";
        return ERROR_INVALID_PARAM;
    }
    Status st = CheckPerm();
    if (st != NO_ERROR) {
        return st;
    }
    ATB_LOG(DEBUG) << "outTensors size:" << outTensors.size();
    return NO_ERROR;
}

Status TransposeOperation::CheckPerm() const
{
    if (param_.perm.size() == 0) {
        ATB_LOG(ERROR) << "input perm cannot be empty";
        return ERROR_INVALID_PARAM;
    }
    SVector<int32_t> permElements;
    permElements.resize(param_.perm.size());
    for (size_t i = 0; i < permElements.size(); i++) {
        permElements.at(i) = 0;
    }
    for (size_t i = 0; i < param_.perm.size(); i++) {
        if (param_.perm.at(i) >= int32_t(param_.perm.size()) || param_.perm.at(i) < 0 ||
            permElements.at(param_.perm.at(i)) == 1) {
            ATB_LOG(ERROR) << "input perm invalid";
            return ERROR_INVALID_PARAM;
        }
        permElements.at(param_.perm.at(i)) = 1;
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> TransposeOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<TransposeOpsRunner>(param_);
}

nlohmann::json TransposeOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb