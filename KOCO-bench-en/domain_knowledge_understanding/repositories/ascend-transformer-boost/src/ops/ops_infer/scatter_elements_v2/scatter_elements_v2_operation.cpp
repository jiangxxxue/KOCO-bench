/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_elements_v2_operation.h"
#include <cmath>
#include "atb/utils/log.h"
#include "scatter_elements_v2_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const uint32_t IN_TENSOR_COUNT = 3;
static const uint32_t OUT_TENSOR_COUNT = 0;

template <> Status CreateOperation(const infer::ScatterElementsV2Param &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);

    if (!GetSingleton<Config>().Is910B()) {
        ATB_LOG(ERROR) << "ScatterElementsV2Operation only supports Atlas 800I A2/A3 ";
        return ERROR_INVALID_PARAM;
    }

    // axis 只能等于 -1
    if (opParam.axis != -1) {
        ATB_LOG(ERROR) << "param_.axis should == -1";
        return ERROR_INVALID_PARAM;
    }

    // reduction 只支持add和none
    if (opParam.reduction != infer::ScatterElementsV2Param::ReductionType::NONE &&
        opParam.reduction != infer::ScatterElementsV2Param::ReductionType::ADD) {
        ATB_LOG(ERROR) << "param_.reduction only support add or none ";
        return ERROR_INVALID_PARAM;
    }

    *operation = new (std::nothrow) ScatterElementsV2Operation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

ScatterElementsV2Operation::ScatterElementsV2Operation(const infer::ScatterElementsV2Param &param)
    : OperationBase("ScatterElementsV2Operation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("ScatterElementsV2Operation");
}

ScatterElementsV2Operation::~ScatterElementsV2Operation() {}

uint32_t ScatterElementsV2Operation::GetInputNum() const
{
    return IN_TENSOR_COUNT;
}

uint32_t ScatterElementsV2Operation::GetOutputNum() const
{
    return OUT_TENSOR_COUNT;
}

Status ScatterElementsV2Operation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                                  SVector<TensorDesc> &outTensorDescs) const
{
    (void)inTensorDescs;
    (void)outTensorDescs;
    return NO_ERROR;
}

Status ScatterElementsV2Operation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    const TensorDesc &inputTensorDesc = inTensorDescs.at(0);
    const TensorDesc &indicesTensorDesc = inTensorDescs.at(1);
    const TensorDesc &updateTensorDesc = inTensorDescs.at(2);

    Status status = ParamCheck(inputTensorDesc, indicesTensorDesc, updateTensorDesc);
    return status;
}

Status ScatterElementsV2Operation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                                  const SVector<Tensor> &outTensors) const
{
    const TensorDesc &inputTensorDesc = inTensors.at(0).desc;
    const TensorDesc &indicesTensorDesc = inTensors.at(1).desc;
    const TensorDesc &updateTensorDesc = inTensors.at(2).desc;

    Status status = ParamCheck(inputTensorDesc, indicesTensorDesc, updateTensorDesc);
    (void)outTensors;
    return status;
}

Status ScatterElementsV2Operation::ParamCheck(const TensorDesc &inputTensorDesc, const TensorDesc &indicesTensorDesc,
                                              const TensorDesc &updateTensorDesc) const
{
    // input_tensor 和 update 的dtype必须相同
    if (inputTensorDesc.dtype != updateTensorDesc.dtype) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "ScatterElementsV2Operation: The dtype values of input_tensor and update must be the same.";
        return ERROR_INVALID_TENSOR_DTYPE;
    }

    Status dimsStatus = CheckDims(inputTensorDesc, indicesTensorDesc, updateTensorDesc);
    if (dimsStatus != NO_ERROR) {
        return dimsStatus;
    }
    Status shapeStatus = CheckShape(inputTensorDesc, indicesTensorDesc, updateTensorDesc);
    if (shapeStatus != NO_ERROR) {
        return shapeStatus;
    }
    return NO_ERROR;
}

Status ScatterElementsV2Operation::CheckDims(const TensorDesc &inputTensorDesc, const TensorDesc &indicesTensorDesc,
                                             const TensorDesc &updateTensorDesc) const
{
    // indice和update的dimNum需要完全相同
    if (indicesTensorDesc.shape.dimNum != updateTensorDesc.shape.dimNum) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "ScatterElementsV2Operation: The dimNum of indice and update must be the same.";
        return ERROR_INVALID_TENSOR_DIM;
    }

    // indice的维度需要和input_tensor的维度相同
    if (indicesTensorDesc.shape.dimNum != inputTensorDesc.shape.dimNum) {
        ATB_LOG(ERROR)
            << GetLogPrefix()
            << "ScatterElementsV2Operation: The dimension of indice must be the same as that of input_tensor.";
        return ERROR_INVALID_TENSOR_DIM;
    }

    return NO_ERROR;
}


Status ScatterElementsV2Operation::CheckShape(const TensorDesc &inputTensorDesc, const TensorDesc &indicesTensorDesc,
                                              const TensorDesc &updateTensorDesc) const
{
    // indice和update的shape需要完全相同
    for (std::size_t i = 0; i < indicesTensorDesc.shape.dimNum; ++i) {
        if (indicesTensorDesc.shape.dims[i] != updateTensorDesc.shape.dims[i]) {
            ATB_LOG(ERROR) << GetLogPrefix()
                           << "ScatterElementsV2Operation: The dims of indice and update must be the same.";
            return ERROR_INVALID_TENSOR_DIM;
        }

        if (indicesTensorDesc.shape.dims[i] == 0) {
            ATB_LOG(ERROR) << GetLogPrefix()
                           << "ScatterElementsV2Operation:  The value of indice_tensor cannot be 0 for any dimension.";
            return ERROR_INVALID_TENSOR_DIM;
        }

        if (inputTensorDesc.shape.dims[i] == 0) {
            ATB_LOG(ERROR) << GetLogPrefix()
                           << "ScatterElementsV2Operation:  The value of input_tensor cannot be 0 for any dimension.";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }

    // input_tensor 和 indice 在非尾轴和非0轴上的shape必须一致
    for (std::size_t i = 1; i < inputTensorDesc.shape.dimNum - 1; ++i) {
        if (inputTensorDesc.shape.dims[i] != indicesTensorDesc.shape.dims[i]) {
            ATB_LOG(ERROR) << GetLogPrefix()
                           << "ScatterElementsV2Operation: The shapes of input_tensor and indices_tensor on the "
                              "non-tail axis and non-zero axis must be the same.";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }

    // // indices_tensor 0轴和尾轴不大于inpute_tensor的0轴和尾轴
    if (inputTensorDesc.shape.dims[0] < indicesTensorDesc.shape.dims[0]) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "ScatterElementsV2Operation: The value of indicatorsTensor must be smaller than the value of "
                          "inputTensor for non-zero axis and non-tail axis.";
        return ERROR_INVALID_TENSOR_DIM;
    }

    // // indices_tensor 0轴和尾轴不大于inpute_tensor的0轴和尾轴
    if (inputTensorDesc.shape.dims[inputTensorDesc.shape.dimNum - 1] <
        indicesTensorDesc.shape.dims[inputTensorDesc.shape.dimNum - 1]) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "ScatterElementsV2Operation: The value of indicatorsTensor must be smaller than the value of "
                          "inputTensor for non-zero axis and non-tail axis.";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> ScatterElementsV2Operation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<ScatterElementsV2OpsRunner>(param_);
}

nlohmann::json ScatterElementsV2Operation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb