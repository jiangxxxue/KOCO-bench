/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "concat_operation.h"
#include "atb/utils/log.h"
#include "concat_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
template <> Status CreateOperation(const infer::ConcatParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    *operation = new ConcatOperation(opParam);
    return NO_ERROR;
}

ConcatOperation::ConcatOperation(const infer::ConcatParam &param) : OperationBase("ConcatOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("ConcatOperation");
}

ConcatOperation::~ConcatOperation() {}

uint32_t ConcatOperation::GetInputNum() const
{
    const uint32_t inTensorNum = 2;
    return inTensorNum;
}

uint32_t ConcatOperation::GetOutputNum() const
{
    return 1;
}

Status ConcatOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                       SVector<TensorDesc> &outTensorDescs) const
{
    int64_t dim = InferDimFromInTensorDesc(inTensorDescs.at(0).shape);
    Dims shape = inTensorDescs.at(0).shape;
    Dims shape1 = inTensorDescs.at(1).shape;
    aclDataType dtype = inTensorDescs.at(0).dtype;
    shape.dims[dim] = shape.dims[dim] + shape1.dims[dim];
    outTensorDescs.at(0) = {dtype, inTensorDescs.at(0).format, shape};
    return NO_ERROR;
}

Status ConcatOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    int64_t dim = InferDimFromInTensorDesc(inTensorDescs.at(0).shape);
    if (dim < 0 || static_cast<int64_t>(inTensorDescs.at(0).shape.dimNum) <= dim ||
        static_cast<int64_t>(inTensorDescs.at(1).shape.dimNum) <= dim) {
        ATB_LOG(ERROR) << "inTensor dimNum is invalid, should > param_.concatDim and param_.concatDim should > 0";
        return ERROR_INVALID_PARAM;
    }
    if (CheckInTensorShape(inTensorDescs.at(0).shape, inTensorDescs.at(1).shape, dim) != NO_ERROR) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status ConcatOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    int64_t dim = InferDimFromInTensorDesc(inTensors.at(0).desc.shape);
    if (dim < 0 || static_cast<int64_t>(inTensors.at(0).desc.shape.dimNum) <= dim ||
        static_cast<int64_t>(inTensors.at(1).desc.shape.dimNum) <= dim) {
        ATB_LOG(ERROR) << "inTensor dimNum is invalid, should > param_.concatDim and param_.concatDim should > 0";
        return ERROR_INVALID_PARAM;
    }
    if (CheckInTensorShape(inTensors.at(0).desc.shape, inTensors.at(1).desc.shape, dim) != NO_ERROR) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    ATB_LOG(DEBUG) << "outTensors size:" << outTensors.size();
    return NO_ERROR;
}

int64_t ConcatOperation::InferDimFromInTensorDesc(const Dims &inTensorShape) const
{
    int64_t dim = param_.concatDim;
    uint64_t dimNum = inTensorShape.dimNum;
    if (dim < 0) {
        dim = static_cast<int64_t>(dimNum) + dim;
    }
    return dim;
}

Status ConcatOperation::CheckInTensorShape(const Dims &shape1, const Dims &shape2, int64_t concatDim) const
{
    if (shape1.dimNum != shape2.dimNum) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor shape does not match";
        return ERROR_INVALID_TENSOR_DIM;
    }
    for (int64_t i = 0; i < static_cast<int64_t>(shape1.dimNum); i++) {
        if (i != concatDim && shape1.dims[i] != shape2.dims[i]) {
            ATB_LOG(ERROR) << GetLogPrefix() << "inTensor shape does not match";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> ConcatOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<ConcatOpsRunner>(param_);
}

nlohmann::json ConcatOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb
