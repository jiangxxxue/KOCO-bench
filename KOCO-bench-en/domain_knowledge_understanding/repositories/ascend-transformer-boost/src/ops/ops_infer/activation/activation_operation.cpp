/*
 * Copyright (c) 2024-2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "activation_operation.h"
#include "atb/utils/param_to_json.h"
#include "activation_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/config.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

namespace {
static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t SWIGLU_BACKWARD_IN_TENSOR_NUM = 2;
static const uint32_t OUT_TENSOR_NUM = 1;
static const uint32_t SPLIT_NUM = 2;
static const int32_t HIDDEN_SIZE_DIM_BASE = 32;
} // namespace

namespace atb {
template <> Status CreateOperation(const infer::ActivationParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (opParam.activationType <= atb::infer::ActivationType::ACTIVATION_UNDEFINED ||
        opParam.activationType >= atb::infer::ActivationType::ACTIVATION_MAX) {
        ATB_LOG(ERROR) << "activationType:" << opParam.activationType << "is invalid ActivationType";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.dim != -1) {
        if ((opParam.activationType == atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD ||
             opParam.activationType == atb::infer::ActivationType::ACTIVATION_SWIGLU_BACKWARD)) {
            ATB_LOG(ERROR) << "activationType: " << opParam.activationType << " only support splitting dim -1";
        } else {
            ATB_LOG(ERROR) << "activationType: " << opParam.activationType << " does not support splitting dim";
        }
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) ActivationOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

static Mki::OperationIr *GetOperationIrForActivation(const infer::ActivationType activationType)
{
    switch (activationType) {
        case atb::infer::ActivationType::ACTIVATION_RELU:
            return GetSingleton<AtbOperationIrCfg>().GetOperationIr("ActivationOperationRELU");
        case atb::infer::ActivationType::ACTIVATION_LOG:
            return GetSingleton<AtbOperationIrCfg>().GetOperationIr("ActivationOperationLOG");
        case atb::infer::ActivationType::ACTIVATION_GELU:
            return GetSingleton<AtbOperationIrCfg>().GetOperationIr("ActivationOperationGELU");
        case atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD:
            return GetSingleton<AtbOperationIrCfg>().GetOperationIr("ActivationOperationSWIGLUFORWARD");
        case atb::infer::ActivationType::ACTIVATION_FAST_GELU:
            return GetSingleton<AtbOperationIrCfg>().GetOperationIr("ActivationOperationFASTGELU");
        case atb::infer::ActivationType::ACTIVATION_SIGMOID:
            return GetSingleton<AtbOperationIrCfg>().GetOperationIr("ActivationOperationSIGMOID");
        case atb::infer::ActivationType::ACTIVATION_SWISH:
            return GetSingleton<AtbOperationIrCfg>().GetOperationIr("ActivationOperationSWISH");
        case atb::infer::ActivationType::ACTIVATION_SWIGLU_BACKWARD:
            return GetSingleton<AtbOperationIrCfg>().GetOperationIr("ActivationOperationSWIGLUBACKWARD");
        case atb::infer::ActivationType::ACTIVATION_FASTER_GELU_FORWARD:
            return GetSingleton<AtbOperationIrCfg>().GetOperationIr("ActivationOperationFasterGeluForward");
        default:
            ATB_LOG(ERROR) << "not support activationType:" << activationType;
    }
    return nullptr;
}

ActivationOperation::ActivationOperation(const infer::ActivationParam &param)
    : OperationBase("ActivationOperation"), param_(param)
{
    operationIr_ = GetOperationIrForActivation(param_.activationType);
    if (!operationIr_) {
        ATB_LOG(ERROR) << "GetOperationIrForActivation failed.";
    }
}

ActivationOperation::~ActivationOperation() {}

uint32_t ActivationOperation::GetInputNum() const
{
    if (param_.activationType == atb::infer::ActivationType::ACTIVATION_SWIGLU_BACKWARD) {
        return SWIGLU_BACKWARD_IN_TENSOR_NUM;
    } else {
        return IN_TENSOR_NUM;
    }
}

uint32_t ActivationOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status ActivationOperation::CheckSwigluBackwardInTensor(const SVector<TensorDesc> &inTensorDescs) const
{
    int32_t dimNums = static_cast<int32_t>(inTensorDescs.at(0).shape.dimNum);
    int32_t splitDim = param_.dim < 0 ? param_.dim + dimNums : param_.dim;
    if (splitDim < 0 || splitDim >= dimNums) {
        ATB_LOG(ERROR) << "The value of attr [dim] must be in the range [-" << dimNums << ", " << (dimNums - 1)
                       << "], but got [" << splitDim << "].";
        return ERROR_INVALID_PARAM;
    }
    if (inTensorDescs.at(0).shape.dimNum != inTensorDescs.at(1).shape.dimNum) {
        ATB_LOG(ERROR) << "inTensor0 & inTensor1 dimNum does not match, should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    for (uint64_t i = 0; i < inTensorDescs.at(0).shape.dimNum; i++) {
        if (static_cast<uint64_t>(splitDim) == i) {
            if (inTensorDescs.at(1).shape.dims[i] != SPLIT_NUM * inTensorDescs.at(0).shape.dims[i]) {
                ATB_LOG(ERROR) << GetLogPrefix() << "Dims[" << i << "] of inTensor0 should be half of inTensor1.";
                return ERROR_INVALID_TENSOR_DIM;
            }
        } else {
            if (inTensorDescs.at(0).shape.dims[i] != inTensorDescs.at(1).shape.dims[i]) {
                ATB_LOG(ERROR) << GetLogPrefix() << "Dims[" << i
                               << "] of inTensor0 and inTensor1 does not match, should be same.";
                return ERROR_INVALID_TENSOR_DIM;
            }
        }
    }
    return NO_ERROR;
}

Status ActivationOperation::CheckSwigluForwardInTensor(const SVector<TensorDesc> &inTensorDescs) const
{
    int32_t splitDim = param_.dim;
    int32_t dimNums = inTensorDescs.at(0).shape.dimNum; // 0: x
    if (splitDim < 0) {
        splitDim += dimNums;
    }
    if (splitDim < 0 || splitDim >= dimNums) {
        ATB_LOG(ERROR) << "The value of param [dim] must be in the range [-" << dimNums << ", " << (dimNums - 1)
                       << "], but got [" << splitDim << "].";
        return ERROR_INVALID_PARAM;
    }
    int32_t hiddenSizeDimNums = inTensorDescs.at(0).shape.dims[dimNums - 1]; // dimNums - 1: hiddenSize
    if (GetSingleton<Config>().Is310P() && hiddenSizeDimNums % HIDDEN_SIZE_DIM_BASE != 0) {
        ATB_LOG(ERROR) << "Atlas inference products only support inTensor[0] with last dim "
                       << "being multiple of 32, but got " << hiddenSizeDimNums;
        return ERROR_INVALID_TENSOR_DIM; // HIDDEN_SIZE_DIM_BASE: multiple of 32
    }
    return NO_ERROR;
}

Status ActivationOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    if (param_.activationType == atb::infer::ActivationType::ACTIVATION_SWIGLU_BACKWARD) {
        return CheckSwigluBackwardInTensor(inTensorDescs);
    }
    if (param_.activationType == atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD) {
        return CheckSwigluForwardInTensor(inTensorDescs);
    }
    return NO_ERROR;
}

Status ActivationOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                           SVector<TensorDesc> &outTensorDescs) const
{
    if (param_.activationType == atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD) {
        int32_t splitDim = param_.dim;
        int32_t dimNums = inTensorDescs.at(0).shape.dimNum; // 0: x
        if (splitDim < 0) {
            splitDim += dimNums;
        }
        outTensorDescs.at(0) = inTensorDescs.at(0); // 0: x
        outTensorDescs.at(0).shape.dims[splitDim] = inTensorDescs.at(0).shape.dims[splitDim] / SPLIT_NUM;
    } else if (param_.activationType == atb::infer::ActivationType::ACTIVATION_SWIGLU_BACKWARD) {
        outTensorDescs.at(0) = inTensorDescs.at(1); // // 0: y_grad, 1: x
    } else {
        outTensorDescs.at(0) = inTensorDescs.at(0);
    }
    return NO_ERROR;
}

Status ActivationOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs = {};
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    if (inTensors.at(0).desc.shape.dimNum != outTensors.at(0).desc.shape.dimNum) {
        ATB_LOG(ERROR) << "inTensor/outTensor dimNum does not match, should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (param_.activationType != atb::infer::ActivationType::ACTIVATION_SWIGLU_BACKWARD &&
        param_.activationType != atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD) {
        for (uint64_t i = 0; i < inTensors.at(0).desc.shape.dimNum; i++) {
            if (inTensors.at(0).desc.shape.dims[i] != outTensors.at(0).desc.shape.dims[i]) {
                ATB_LOG(ERROR) << GetLogPrefix() << "inTensor/outTensor dims does not match, should be same";
                return ERROR_INVALID_TENSOR_DIM;
            }
        }
        return NO_ERROR;
    }
    // Swiglu
    if (param_.activationType == atb::infer::ActivationType::ACTIVATION_SWIGLU_BACKWARD) {
        return CheckSwigluBackwardInTensor(inTensorDescs);
    }
    Status st;
    if ((st = CheckSwigluForwardInTensor(inTensorDescs)) != NO_ERROR) {
        return st;
    }
    int32_t dimNums = static_cast<int32_t>(inTensorDescs.at(0).shape.dimNum);
    int32_t splitDim = param_.dim < 0 ? param_.dim + dimNums : param_.dim;
    for (int32_t i = 0; i < dimNums; ++i) {
        if (i == splitDim) {
            if (SPLIT_NUM * outTensors.at(0).desc.shape.dims[splitDim] != inTensors.at(0).desc.shape.dims[splitDim]) {
                ATB_LOG(ERROR) << GetLogPrefix() << "outTensor dims[" << splitDim
                               << "] should be half of inTensor dims[" << splitDim << "]";
                return ERROR_INVALID_TENSOR_DIM;
            }
        } else if (outTensors.at(0).desc.shape.dims[i] != inTensors.at(0).desc.shape.dims[i]) {
            ATB_LOG(ERROR) << GetLogPrefix() << "Dims[" << i
                           << "] of inTensor0 and inTensor1 does not match, should be same.";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> ActivationOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<ActivationOpsRunner>(param_);
}

nlohmann::json ActivationOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb