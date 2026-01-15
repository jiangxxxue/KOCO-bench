/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "repeat_operation.h"
#include "repeat_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils/singleton.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"

static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_NUM = 1;

namespace atb {
template <> Status CreateOperation(const infer::RepeatParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (opParam.multiples.size() > MAX_DIM) {
        ATB_LOG(ERROR) << "The dimNum of param.multiples should <= MAX_DIM(8)";
        return ERROR_INVALID_PARAM;
    }

    for (size_t i = 0; i < opParam.multiples.size(); ++i) {
        if (opParam.multiples.at(i) <= 0) {
            ATB_LOG(ERROR) << "Each value of param.multiples should be greater than 0";
            return ERROR_INVALID_PARAM;
        }
    }

    *operation = new (std::nothrow) RepeatOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

RepeatOperation::RepeatOperation(const infer::RepeatParam &param) : OperationBase("RepeatOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("RepeatOperation");
}

RepeatOperation::~RepeatOperation() {}

uint32_t RepeatOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t RepeatOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status RepeatOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                       SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).shape.dimNum = param_.multiples.size();
    int64_t idx = inTensorDescs.at(0).shape.dimNum;
    int64_t multipleSize = static_cast<int64_t>(param_.multiples.size()) - 1;
    for (int64_t i = multipleSize; i >= 0; --i) {
        if (idx > 0) {
            if (std::numeric_limits<int64_t>::max() / inTensorDescs.at(0).shape.dims[idx - 1] < param_.multiples[i]) {
                ATB_LOG(ERROR) << "Repeat inferShape outTensor Size Overflow.";
                return ERROR_INVALID_PARAM;
            }
            outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[idx - 1] * param_.multiples[i];
            idx--;
        } else {
            outTensorDescs.at(0).shape.dims[i] = param_.multiples[i];
        }
    }
    return NO_ERROR;
}

Status RepeatOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(0).shape.dimNum > param_.multiples.size()) {
        ATB_LOG(WARN) << "inTensor dimNum is invalid, should <= param_.multiples.size";
        return ERROR_INVALID_TENSOR_DIM;
    }
    uint32_t repeatDimNum = 0;
    for (int64_t i = inTensorDescs.at(0).shape.dimNum - 1; i >= 0; --i) {
        if (param_.multiples.at(i) > 1 && inTensorDescs.at(0).shape.dims[i] > 1) {
            repeatDimNum++;
        }
    }
    if (repeatDimNum + inTensorDescs.at(0).shape.dimNum > MAX_DIM) {
        ATB_LOG(ERROR) << "inTensor dimNum is invalid, repeat dims + inTensor dimNum should <= 8";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status RepeatOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    if (inTensors.at(0).desc.shape.dimNum > param_.multiples.size()) {
        ATB_LOG(WARN) << "inTensor dimNum is invalid, should <= param_.multiples.size";
        return ERROR_INVALID_TENSOR_DIM;
    }
    uint32_t repeatDimNum = 0;
    for (int64_t i = inTensors.at(0).desc.shape.dimNum - 1; i >= 0; --i) {
        if (param_.multiples.at(i) > 1 && inTensors.at(0).desc.shape.dims[i] > 1) {
            repeatDimNum++;
        }
    }
    if (repeatDimNum + inTensors.at(0).desc.shape.dimNum > MAX_DIM) {
        ATB_LOG(ERROR) << "inTensor dimNum is invalid, repeat dims + inTensor dimNum should <= 8";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (outTensors.at(0).desc.shape.dimNum != param_.multiples.size()) {
        ATB_LOG(ERROR) << "inTensor/outTensor dimNum does not match, should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> RepeatOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<RepeatOpsRunner>(param_);
}

nlohmann::json RepeatOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb