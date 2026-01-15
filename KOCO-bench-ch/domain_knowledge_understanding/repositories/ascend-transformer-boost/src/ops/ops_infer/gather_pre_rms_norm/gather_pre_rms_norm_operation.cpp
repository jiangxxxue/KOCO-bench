/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gather_pre_rms_norm_operation.h"
#include <cmath>
#include "gather_pre_rms_norm_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils/config.h"
#include "atb/utils/singleton.h"
#include "atb/utils/operation_util.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const uint32_t IN_TENSOR_COUNT_FOUR = 4;
static const uint32_t OUT_TENSOR_COUNT_TWO = 2;
static const uint32_t IN_TENSOR_X = 0;
static const uint32_t OUT_TENSOR_X = 0;
static const uint32_t DN_MAX_VALUE = 7680;
static const float THRESHOLD = 2e-38; // The non-negative minimum value of float type is 1.17549e-038.


template <> Status CreateOperation(const infer::GatherPreRmsNormParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (std::fabs(opParam.epsilon) < THRESHOLD) {
        ATB_LOG(ERROR) << "Invalid epsilon, it's recommended to init a nonzero value for eps.";
        return ERROR_INVALID_PARAM;
    }
    if (!GetSingleton<Config>().Is910B()) {
        ATB_LOG(ERROR) << "GatherPreRmsNormOperation only supports the Atlas 800I A2 inference product";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) GatherPreRmsNormOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

GatherPreRmsNormOperation::GatherPreRmsNormOperation(const infer::GatherPreRmsNormParam &param)
    : OperationBase("GatherPreRmsNormOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("GatherPreRmsNormOperation");
}

GatherPreRmsNormOperation::~GatherPreRmsNormOperation() {}

uint32_t GatherPreRmsNormOperation::GetInputNum() const
{
    return IN_TENSOR_COUNT_FOUR;
}

uint32_t GatherPreRmsNormOperation::GetOutputNum() const
{
    return OUT_TENSOR_COUNT_TWO;
}

Status GatherPreRmsNormOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                                 SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    outTensorDescs.at(0).dtype = ACL_FLOAT;

    outTensorDescs.at(1) = inTensorDescs.at(0);
    return NO_ERROR;
}

Status GatherPreRmsNormOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    if (!GatherPreRMSNormCheckImpl910B(inTensorDescs)) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs[0].shape.dims[inTensorDescs[0].shape.dimNum - 1] > DN_MAX_VALUE) { // 0: 0th dim 0: 0th dim
        ATB_LOG(ERROR) << GetLogPrefix() << "The last dimNum of the first input tensor"
                                         << " should be less than or equal 7680";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status GatherPreRmsNormOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                                 const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs;
    for (size_t i = 0; i < inTensors.size(); i++) {
        inTensorDescs.push_back(inTensors.at(i).desc);
    }
    aclDataType targetType = outTensors[0].desc.dtype;
    Status st = CheckOutTensorSame(inTensorDescs[IN_TENSOR_X], outTensors[OUT_TENSOR_X].desc, targetType);
    if (st != NO_ERROR) {
        return st;
    }
    return InferShapeCheckImpl(inTensorDescs);
}

std::shared_ptr<Runner> GatherPreRmsNormOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<GatherPreRmsNormOpsRunner>(param_);
}


bool GatherPreRmsNormOperation::GatherPreRMSNormCheckImpl910B(const SVector<TensorDesc> &inTensorDescs) const
{
    uint32_t idx = 0;
    TensorDesc xTensorDesc = inTensorDescs.at(idx++);
    TensorDesc resInTensorDesc = inTensorDescs.at(idx++);
    TensorDesc indices = inTensorDescs.at(idx++);
    TensorDesc gammaTensorDesc = inTensorDescs.at(idx++);
    if (!GammaBetaTensorCheck(xTensorDesc, gammaTensorDesc)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "gammaTensorDesc error";
        return false;
    }
    if (xTensorDesc.shape.dims[1] != resInTensorDesc.shape.dims[1]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "The second dimension num of the first input tensor \
                                            and the second input tensor should be equal";
        return false;
    }
    if (xTensorDesc.shape.dims[0] != indices.shape.dims[0]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "The first dimension num of the first input tensor \
                                            and the third input tensor should be equal";
        return false;
    }
    if (xTensorDesc.shape.dims[1] % 16 != 0 || // 1: 2nd dim 16: Align units
        resInTensorDesc.shape.dims[1] % 16 != 0 || // 1: 2nd dim 16: Align units
        gammaTensorDesc.shape.dims[1] % 16 != 0) { // 1: 2nd dim 16: Align units
        ATB_LOG(ERROR) << GetLogPrefix() << "The second dimension num of the first input tensor,"
                                         << " the second input tensor and"
                                         << " the third input tensor should be a multiple of 16";
        return false;
    }
    return true;
}

Status GatherPreRmsNormOperation::CheckOutTensorSame(const TensorDesc &tensorDesc1, const TensorDesc &tensorDesc2,
                                                     const aclDataType &targetType) const
{
    if (targetType != ACL_FLOAT) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor dtype is invalid, outTensor dtype should be float.";
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

bool GatherPreRmsNormOperation::GammaBetaTensorCheck(const TensorDesc &xTensorDesc, const TensorDesc &tensorDesc2) const
{
    int embedDim = xTensorDesc.shape.dims[xTensorDesc.shape.dimNum - 1];
    if (xTensorDesc.dtype != tensorDesc2.dtype || xTensorDesc.format != tensorDesc2.format) {
        ATB_LOG(ERROR) << GetLogPrefix() << "gamma and bias should have the same dtype and format as x";
        return false;
    }
    if (tensorDesc2.shape.dims[tensorDesc2.shape.dimNum - 1] != embedDim) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensors dims error, inTensor last dim is not equal to embed_dim";
        return false;
    }
    for (uint64_t i = 0; i < tensorDesc2.shape.dimNum - 1; i++) {
        if (tensorDesc2.shape.dims[i] != 1) {
            ATB_LOG(ERROR) << GetLogPrefix() << "inTensors dims error, inTensor non-last dim is not equal to 1";
            return false;
        }
    }
    return true;
}

nlohmann::json GatherPreRmsNormOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb
