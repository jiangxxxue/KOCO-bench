/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "layer_norm_with_stride_operation.h"
#include <cmath>
#include <sstream>
#include "layer_norm_with_stride_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/config.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"
namespace atb {
static const uint32_t IN_TENSOR_COUNT_FIVE = 5;
static const uint32_t OUT_TENSOR_COUNT_ONE = 1;
static const uint32_t IN_TENSOR_SCALE_NORMNORM = 3;
static const uint32_t IN_TENSOR_OFFSET_NORMNORM = 4;
static const uint32_t ALIGNMENT = 32;
static const uint32_t STRIDE_TENSOR_COUNT = 2;
static const float THRESHOLD = 2e-38; // The non-negative minimum value of float type is 1.17549e-038.
static const uint32_t DYNAMIC_QUANT_MAX_TENSOR_LENGTH = 12288;

template <> Status CreateOperation(const infer::LayerNormWithStrideParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    if (!GetSingleton<Config>().Is910B()) {
        ATB_LOG(ERROR) << "RmsNormWithStride only support Atlas 800I A2 inference product";
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    OP_PARAM_RSV_CHECK(opParam.normParam);
    OP_PARAM_RSV_CHECK(opParam.preNormParam);
    OP_PARAM_RSV_CHECK(opParam.postNormParam);
    if (opParam.layerType != infer::LayerNormWithStrideParam::LAYER_NORM_NORM) {
        ATB_LOG(ERROR) << "LayerNormWithStrideParam only support LAYER_NORM_NORM";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.normParam.quantType != infer::QUANT_UNQUANT) {
        ATB_LOG(ERROR) << "quantType is not supported, only support QUANT_UNQUANT";
        return ERROR_INVALID_PARAM;
    }
    if (std::fabs(opParam.normParam.epsilon) < THRESHOLD) {
        ATB_LOG(ERROR) << "Invalid epsilon, it's recommended to init a nonzero value for eps.";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_ASYMMETRIC) {
        ATB_LOG(ERROR) << "dynamicQuantType not support DYNAMIC_QUANT_ASYMMETRIC";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) LayerNormWithStrideOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

LayerNormWithStrideOperation::LayerNormWithStrideOperation(const infer::LayerNormWithStrideParam &param)
    : OperationBase("LayerNormWithStrideOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("LayerNormWithStrideOperation");
}

LayerNormWithStrideOperation::~LayerNormWithStrideOperation() {}

uint32_t LayerNormWithStrideOperation::GetInputNum() const
{
    return IN_TENSOR_COUNT_FIVE;
}

uint32_t LayerNormWithStrideOperation::GetOutputNum() const
{
    return OUT_TENSOR_COUNT_ONE;
}

Status LayerNormWithStrideOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                                    SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    return NO_ERROR;
}

Status LayerNormWithStrideOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    const TensorDesc &xTensorDesc = inTensorDescs.at(0);
    const TensorDesc &gammaTensorDesc = inTensorDescs.at(1);
    Status checkStatus = ParamCheck(xTensorDesc, gammaTensorDesc);
    if (checkStatus != NO_ERROR) {
        return checkStatus;
    }
    checkStatus = LastDimCheck(inTensorDescs);
    if (checkStatus != NO_ERROR) {
        return checkStatus;
    }
    return NO_ERROR;
}

Status LayerNormWithStrideOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                                    const SVector<Tensor> &outTensors) const
{
    const TensorDesc &xTensorDesc = inTensors.at(0).desc;
    const TensorDesc &gammaTensorDesc = inTensors.at(1).desc;
    Status paramCheckStatus = ParamCheck(xTensorDesc, gammaTensorDesc);
    if (paramCheckStatus != NO_ERROR) {
        return paramCheckStatus;
    }
    SVector<TensorDesc> inTensorDescs;
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    Status lastDimCheckStatus = LastDimCheck(inTensorDescs);
    if (lastDimCheckStatus != NO_ERROR) {
        return lastDimCheckStatus;
    }
    (void)outTensors;
    return NO_ERROR;
}

Status LayerNormWithStrideOperation::ParamCheck(const TensorDesc &xTensorDesc, const TensorDesc &gammaTensorDesc) const
{
    int32_t beginNormAxis = param_.normParam.beginNormAxis;
    uint64_t xTensorDimNum = xTensorDesc.shape.dimNum;
    uint64_t gammaTensorDimNum = gammaTensorDesc.shape.dimNum;
    if (param_.layerType == infer::LayerNormWithStrideParam::LAYER_NORM_NORM &&
        param_.normParam.quantType == infer::QUANT_UNQUANT) {
        if (gammaTensorDimNum > xTensorDimNum) {
            ATB_LOG(ERROR) << GetLogPrefix() << "gammaTensor dimNum should be smaller or equal to xTensor.";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if ((beginNormAxis >= 0 && static_cast<uint64_t>(beginNormAxis) + gammaTensorDimNum != xTensorDimNum) ||
            (beginNormAxis < 0 && static_cast<uint64_t>(-beginNormAxis) != gammaTensorDimNum)) {
            ATB_LOG(ERROR) << GetLogPrefix()
                           << "Invalid param: beginNormAxis,"
                              "beginNormAxis should not larger than xTensorDimNum or beginNormAxis can be negative,"
                              "but beginNormAxis+xTensorDimNum cannot be negative.";
            return ERROR_INVALID_PARAM;
        }
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> LayerNormWithStrideOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<LayerNormWithStrideOpsRunner>(param_);
}

nlohmann::json LayerNormWithStrideOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}

bool LayerNormWithStrideOperation::QuantTensorCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    uint32_t scaleIndex = 0;
    uint32_t offsetIndex = 0;
    scaleIndex = IN_TENSOR_SCALE_NORMNORM;
    offsetIndex = IN_TENSOR_OFFSET_NORMNORM;
    const TensorDesc scaleInTensorDesc = inTensorDescs.at(scaleIndex);
    const TensorDesc offsetInTensorDesc = inTensorDescs.at(offsetIndex);
    if (scaleInTensorDesc.shape.dimNum != 1 || offsetInTensorDesc.shape.dimNum != 1) {
        ATB_LOG(ERROR) << GetLogPrefix() << "dim numbers of quant param intensors should be one";
        return false;
    }
    if (scaleInTensorDesc.shape.dims[0] != 1 || offsetInTensorDesc.shape.dims[0] != 1) {
        ATB_LOG(ERROR) << GetLogPrefix() << "dim sizes of quant param intensors should be one";
        return false;
    }
    return true;
}

Status LayerNormWithStrideOperation::LastDimCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    uint64_t dimNum = inTensorDescs.at(0).shape.dimNum;
    int64_t lastDim = inTensorDescs.at(0).shape.dims[dimNum - 1];
    // only layernorm no quant has no restriction on alignment
    if (!(param_.layerType == infer::LayerNormWithStrideParam::LAYER_NORM_NORM &&
          param_.normParam.quantType == infer::QUANT_UNQUANT)) {
        if ((static_cast<uint64_t>(lastDim) % static_cast<uint64_t>(ALIGNMENT / sizeof(uint16_t))) != 0) {
            ATB_LOG(ERROR) << GetLogPrefix() << "inTensor shape is not support, should align 32";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    uint32_t inTensorCount = GetInputNum();
    for (uint32_t i = 1; i < inTensorCount - STRIDE_TENSOR_COUNT; i++) {
        dimNum = inTensorDescs.at(i).shape.dimNum;
        if ((inTensorDescs.at(i).shape.dims[dimNum - 1]) != lastDim) {
            ATB_LOG(ERROR) << GetLogPrefix() << "last dims of inTensors does not match";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    if (param_.layerType == infer::LayerNormWithStrideParam::LAYER_NORM_NORM &&
        param_.normParam.dynamicQuantType == infer::DYNAMIC_QUANT_SYMMETRIC &&
        lastDim > DYNAMIC_QUANT_MAX_TENSOR_LENGTH) {
        ATB_LOG(ERROR) << GetLogPrefix() << "Length of the last dim is invalid, it should be no larger than "
                       << DYNAMIC_QUANT_MAX_TENSOR_LENGTH << ".";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}
} // namespace atb