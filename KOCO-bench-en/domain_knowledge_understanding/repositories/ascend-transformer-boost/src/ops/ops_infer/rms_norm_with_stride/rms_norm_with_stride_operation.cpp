/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rms_norm_with_stride_operation.h"
#include <cmath>
#include "rms_norm_with_stride_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"
#include "atb/utils/operation_util.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const uint32_t IN_TENSOR_COUNT_FOUR = 4;
static const uint32_t OUT_TENSOR_COUNT_ONE = 1;
static const uint32_t IN_TENSOR_X = 0;
static const uint32_t ALIGNMENT = 32;
static const uint32_t IN_TENSOR_STRIDE_INDEX = 2;
static const float THRESHOLD = 2e-38; // The non-negative minimum value of float type is 1.17549e-038.

bool WithStrideEpsilonCheck(const infer::RmsNormWithStrideParam &opParam)
{
    if (std::fabs(opParam.normParam.epsilon) < THRESHOLD || std::fabs(opParam.preNormParam.epsilon) < THRESHOLD ||
        std::fabs(opParam.normParam.layerNormEps) < THRESHOLD || std::fabs(opParam.postNormParam.epsilon) < THRESHOLD) {
        return false;
    }
    return true;
}

template <> Status CreateOperation(const infer::RmsNormWithStrideParam &opParam, Operation **operation)
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
    if (opParam.layerType != infer::RmsNormWithStrideParam::RMS_NORM_NORM) {
        ATB_LOG(ERROR) << "RmsNormWithStrideParam only support RMS_NORM_NORM";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.normParam.quantType != infer::QUANT_UNQUANT) {
        ATB_LOG(ERROR) << "RmsNormWithStrideParam normParam only support QUANT_UNQUANT";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.normParam.rstd) {
        ATB_LOG(ERROR) << "RmsNormWithStrideParam normParam rstd only support false";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.normParam.dynamicQuantType != infer::DynamicQuantType::DYNAMIC_QUANT_UNDEFINED) {
        ATB_LOG(ERROR) << "RmsNormWithStrideParam normParam dynamicQuantType only support DYNAMIC_QUANT_UNDEFINED";
        return ERROR_INVALID_PARAM;
    }
    if (!WithStrideEpsilonCheck(opParam)) {
        ATB_LOG(ERROR) << "Invalid epsilon, it's recommended to init a nonzero value for eps.";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) RmsNormWithStrideOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

RmsNormWithStrideOperation::RmsNormWithStrideOperation(const infer::RmsNormWithStrideParam &param)
    : OperationBase("RmsNormOperation"), param_(param)
{
    operationIr_ =
        GetSingleton<AtbOperationIrCfg>().GetOperationIr("RmsNormWithStrideOperationNORMAndQUANTUNDEFINEDWithoutRSTD");
}

RmsNormWithStrideOperation::~RmsNormWithStrideOperation() {}

uint32_t RmsNormWithStrideOperation::GetInputNum() const
{
    return IN_TENSOR_COUNT_FOUR;
}

uint32_t RmsNormWithStrideOperation::GetOutputNum() const
{
    return OUT_TENSOR_COUNT_ONE;
}

Status RmsNormWithStrideOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                                  SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    if (param_.layerType == infer::RmsNormWithStrideParam::RMS_NORM_PRENORM &&
        param_.preNormParam.quantType == infer::QUANT_UNQUANT) {
        outTensorDescs.at(1) = inTensorDescs.at(0);
    }
    return NO_ERROR;
}

Status RmsNormWithStrideOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    if (param_.layerType == infer::RmsNormWithStrideParam::RMS_NORM_NORM) {
        if (!NormInferShapeCheckImpl(inTensorDescs)) {
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    if (!CheckLastDims(inTensorDescs)) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status RmsNormWithStrideOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                                  const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs;
    for (size_t i = 0; i < inTensors.size(); i++) {
        inTensorDescs.push_back(inTensors.at(i).desc);
    }
    aclDataType targetType = inTensorDescs[IN_TENSOR_X].dtype;
    Status st = CheckOutTensorSame(inTensorDescs[IN_TENSOR_X], outTensors[0].desc, targetType);
    if (st != NO_ERROR) {
        return st;
    }
    return InferShapeCheckImpl(inTensorDescs);
}

std::shared_ptr<Runner> RmsNormWithStrideOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<RmsNormWithStrideOpsRunner>(param_);
}

bool RmsNormWithStrideOperation::NormInferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    TensorDesc xTensorDesc = inTensorDescs.at(0);
    TensorDesc gammaTensorDesc = inTensorDescs.at(1);
    uint64_t xDimNum = xTensorDesc.shape.dimNum;
    uint64_t gammaDimNum = gammaTensorDesc.shape.dimNum;
    if (xTensorDesc.shape.dims[xDimNum - 1] != gammaTensorDesc.shape.dims[gammaDimNum - 1]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensors dims error, inTensor0 lastDim == inTensor1 lastDim";
        return false;
    }
    for (uint64_t i = 0; i < gammaTensorDesc.shape.dimNum - 1; i++) {
        if (gammaTensorDesc.shape.dims[i] != 1) {
            ATB_LOG(ERROR) << GetLogPrefix() << "inTensors dims error, inTensor1 dimsExceptLastDim == 1";
            return false;
        }
    }
    return true;
}

bool WithStrideTensorDescsDiff(const TensorDesc &tensorDesc1, const TensorDesc &tensorDesc2)
{
    if (tensorDesc1.dtype != tensorDesc2.dtype || tensorDesc1.format != tensorDesc2.format ||
        tensorDesc1.shape.dimNum != tensorDesc2.shape.dimNum) {
        return true;
    }
    for (uint64_t i = 0; i < tensorDesc1.shape.dimNum; i++) {
        if (tensorDesc1.shape.dims[i] != tensorDesc2.shape.dims[i]) {
            return true;
        }
    }
    return false;
}

Status RmsNormWithStrideOperation::CheckOutTensorSame(const TensorDesc &tensorDesc1, const TensorDesc &tensorDesc2,
                                                      const aclDataType &targetType) const
{
    if (targetType != tensorDesc2.dtype) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor dtype is invalid";
        return ERROR_INVALID_TENSOR_DTYPE;
    }
    if (tensorDesc1.format != tensorDesc2.format) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor format is invalid";
        return ERROR_INVALID_TENSOR_FORMAT;
    }
    if (tensorDesc1.shape.dimNum != tensorDesc2.shape.dimNum) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor dimNum is invalid";
        return ERROR_INVALID_TENSOR_SIZE;
    }
    for (uint64_t i = 0; i < tensorDesc1.shape.dimNum; i++) {
        if (tensorDesc1.shape.dims[i] != tensorDesc2.shape.dims[i]) {
            ATB_LOG(ERROR) << GetLogPrefix() << "outTensor dims are invalid";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

bool RmsNormWithStrideOperation::CheckLastDims(const SVector<TensorDesc> &inTensorDescs) const
{
    uint32_t inTensorCount = GetInputNum();
    uint64_t dimNum = inTensorDescs.at(IN_TENSOR_X).shape.dimNum;
    int64_t lastDim = inTensorDescs.at(IN_TENSOR_X).shape.dims[dimNum - 1];
    uint32_t dtypeSize = inTensorDescs.at(IN_TENSOR_X).dtype == ACL_FLOAT ? sizeof(uint32_t) : sizeof(uint16_t);
    if ((static_cast<uint64_t>(lastDim) % static_cast<uint64_t>(ALIGNMENT / dtypeSize) != 0)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor shape is not support, should align 32";
        return false;
    }
    for (uint32_t i = 1; i < inTensorCount - IN_TENSOR_STRIDE_INDEX; i++) {
        dimNum = inTensorDescs.at(i).shape.dimNum;
        if ((inTensorDescs.at(i).shape.dims[dimNum - 1]) != lastDim) {
            ATB_LOG(ERROR) << GetLogPrefix() << "last dims of inTensors does not match";
            return false;
        }
    }
    return true;
}

nlohmann::json RmsNormWithStrideOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb
