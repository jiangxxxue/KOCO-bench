/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gather_operation.h"
#include <cmath>
#include "atb/utils/log.h"
#include "gather_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const uint32_t IN_TENSOR_COUNT = 2;
static const uint32_t OUT_TENSOR_COUNT = 1;

template <> Status CreateOperation(const infer::GatherParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (opParam.axis < 0 || opParam.batchDims < 0) {
        ATB_LOG(ERROR) << "param_.axis and param_.batchdims should >= 0";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.batchDims > opParam.axis) {
        ATB_LOG(ERROR) << "param_.batchdims should  <= param_.axis ";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) GatherOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

GatherOperation::GatherOperation(const infer::GatherParam &param) : OperationBase("GatherOperation"), param_(param)
{
    if (GetSingleton<Config>().Is310B()) {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("GatherOperationAtlas200I500A2");
    } else {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("GatherOperation");
    }
}

GatherOperation::~GatherOperation() {}

uint32_t GatherOperation::GetInputNum() const
{
    return IN_TENSOR_COUNT;
}

uint32_t GatherOperation::GetOutputNum() const
{
    return OUT_TENSOR_COUNT;
}

Status GatherOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                       SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    uint64_t dim1 = static_cast<uint64_t>(param_.axis);
    uint64_t dim2 = inTensorDescs.at(1).shape.dimNum - static_cast<uint64_t>(param_.batchDims);
    uint64_t dim3 = inTensorDescs.at(0).shape.dimNum - dim1 - 1;
    outTensorDescs.at(0).shape.dimNum = dim1 + dim2 + dim3;
    if (outTensorDescs.at(0).shape.dimNum > MAX_DIM) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensorDescs dimNum is out of range";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    size_t dim0 = 0;
    for (size_t i = 0; i < static_cast<size_t>(dim1); i++) {
        outTensorDescs.at(0).shape.dims[dim0++] = inTensorDescs.at(0).shape.dims[i];
    }
    for (size_t i = static_cast<size_t>(param_.batchDims); i < inTensorDescs.at(1).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[dim0++] = inTensorDescs.at(1).shape.dims[i];
    }
    for (size_t i = static_cast<size_t>(dim1) + 1; i < inTensorDescs.at(0).shape.dimNum; ++i) {
        outTensorDescs.at(0).shape.dims[dim0++] = inTensorDescs.at(0).shape.dims[i];
    }
    for (size_t i = 0; i < outTensorDescs.at(0).shape.dimNum; i++) {
        ATB_LOG(DEBUG) << "GatherV2InferShape OutTensor dims[" << i << "] = " << outTensorDescs.at(0).shape.dims[i];
    }
    return NO_ERROR;
}

Status GatherOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    const TensorDesc &xTensorDesc = inTensorDescs.at(0);
    const TensorDesc &indicesTensorDesc = inTensorDescs.at(1);
    Status status = ParamCheck(xTensorDesc, indicesTensorDesc);
    if (status != NO_ERROR) {
        return status;
    }
    return NO_ERROR;
}

Status GatherOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    const TensorDesc &xTensorDesc = inTensors.at(0).desc;
    const TensorDesc &indicesTensorDesc = inTensors.at(1).desc;
    Status status = ParamCheck(xTensorDesc, indicesTensorDesc);
    if (status != NO_ERROR) {
        return status;
    }
    ATB_LOG(DEBUG) << "outTensors size:" << outTensors.size();
    return NO_ERROR;
}

Status GatherOperation::ParamCheck(const TensorDesc &xTensorDesc, const TensorDesc &indicesTensorDesc) const
{
    if (static_cast<uint64_t>(param_.axis) >= xTensorDesc.shape.dimNum) {
        ATB_LOG(ERROR) << GetLogPrefix() << "param_.axis should  < inTensorDescs(0) dimNum";
        return ERROR_INVALID_PARAM;
    }
    if (static_cast<uint64_t>(param_.batchDims) > indicesTensorDesc.shape.dimNum) {
        ATB_LOG(ERROR) << GetLogPrefix() << "param_.batchDims should  <= inTensorDescs(1) dimNum";
        return ERROR_INVALID_PARAM;
    }
    if (xTensorDesc.shape.dimNum + indicesTensorDesc.shape.dimNum - 1 - param_.batchDims > 8) {
        ATB_LOG(ERROR) << GetLogPrefix() << "x dim + indices dim should - 1 - batchDims <= 8";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> GatherOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<GatherOpsRunner>(param_);
}

nlohmann::json GatherOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb
