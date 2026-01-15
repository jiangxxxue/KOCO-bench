/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fill_operation.h"
#include "fill_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"
namespace {
static const size_t FILL_PARAM_VALUE_NUM = 1;
static const uint32_t IN_TENSOR_NUM_WITH_MASK = 2;
static const uint32_t IN_TENSOR_NUM_WITHOUT_MASK = 0;
static const uint32_t OUT_TENSOR_NUM = 1;
bool ParamCheck(const atb::infer::FillParam &opParam)
{
    if (opParam.value.size() != FILL_PARAM_VALUE_NUM) {
        ATB_LOG(ERROR) << "fillParam value size should be 1, actually: ", opParam.value.size();
        return false;
    }
    if (!opParam.withMask && !atb::TensorCheck::IsDimNumValid(opParam.outDim.size())) {
        ATB_LOG(ERROR) << "fillParam withMask is false, outDim.size() should >0 && <= MAX_DIM(8)";
        return false;
    }
    return true;
}
} // namespace

namespace atb {
OPERATION_PARAM_FUNCS(FillOperation, infer::FillParam)

FillOperation::FillOperation(const infer::FillParam &param) : OperationBase("FillOperation"), param_(param)
{
    std::string opIrKey = param_.withMask ? "FillOperationWithMask" : "FillOperationWithoutMask";
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr(opIrKey);
}

FillOperation::~FillOperation() {}

uint32_t FillOperation::GetInputNum() const
{
    return param_.withMask ? IN_TENSOR_NUM_WITH_MASK : IN_TENSOR_NUM_WITHOUT_MASK;
}

uint32_t FillOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status FillOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                     SVector<TensorDesc> &outTensorDescs) const
{
    if (param_.withMask) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
    } else {
        outTensorDescs.at(0).dtype = ACL_FLOAT16;
        outTensorDescs.at(0).format = ACL_FORMAT_ND;
        outTensorDescs.at(0).shape.dimNum = param_.outDim.size();
        for (size_t i = 0; i < param_.outDim.size(); i++) {
            outTensorDescs.at(0).shape.dims[i] = param_.outDim.at(i);
        }
    }
    return NO_ERROR;
}

Status FillOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    if (param_.withMask) {
        if (CheckIsTensorBroadcastable(inTensorDescs.at(0), inTensorDescs.at(1)) != NO_ERROR) {
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

Status FillOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    if (param_.withMask) {
        return SetupCheckWithMaskImpl(inTensors, outTensors);
    } else {
        if (outTensors.at(0).desc.shape.dimNum != static_cast<uint64_t>(param_.outDim.size())) {
            ATB_LOG(ERROR) << GetLogPrefix() << "outTensor dim does not match outDim in param";
            return ERROR_INVALID_TENSOR_DIM;
        }
        for (size_t i = 0; i < param_.outDim.size(); i++) {
            if (outTensors.at(0).desc.shape.dims[i] != param_.outDim.at(i)) {
                ATB_LOG(ERROR) << GetLogPrefix() << "outTensor dim does not match outDim in param";
                return ERROR_INVALID_TENSOR_DIM;
            }
        }
        return NO_ERROR;
    }
}

Status FillOperation::SetupCheckWithMaskImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    if (CheckIsTensorBroadcastable(inTensors.at(0).desc, inTensors.at(1).desc) != NO_ERROR) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (outTensors.at(0).desc.shape.dimNum != inTensors.at(0).desc.shape.dimNum) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor shape does not match inTensor shape in param";
        return ERROR_INVALID_TENSOR_DIM;
    }
    for (size_t i = 0; i < outTensors.at(0).desc.shape.dimNum; i++) {
        if (outTensors.at(0).desc.shape.dims[i] != inTensors.at(0).desc.shape.dims[i]) {
            ATB_LOG(ERROR) << GetLogPrefix() << "outTensor shape does not match inTensor shape in param";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

Status FillOperation::CheckIsTensorBroadcastable(const TensorDesc &input, const TensorDesc &mask) const
{
    if (input.shape.dimNum < mask.shape.dimNum) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensors should have same dimension";
        return ERROR_INVALID_TENSOR_DIM;
    }
    uint64_t lenDiff = input.shape.dimNum - mask.shape.dimNum;
    for (uint64_t j = 0; j < mask.shape.dimNum; j++) {
        uint64_t i = j + lenDiff;
        if (input.shape.dims[i] != mask.shape.dims[j] && mask.shape.dims[j] != static_cast<int64_t>(1)) {
            ATB_LOG(ERROR) << GetLogPrefix() << "inTensors' shapes are not broadcastable";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> FillOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<FillOpsRunner>(param_);
}

nlohmann::json FillOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}

infer::FillParam FillOperation::GetParam() const
{
    return param_;
}

void FillOperation::SetParam(const infer::FillParam &param)
{
    param_ = param;
    runner_ = nullptr;
    std::string opIrKey = param_.withMask ? "FillOperationWithMask" : "FillOperationWithoutMask";
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr(opIrKey);
}
} // namespace atb
