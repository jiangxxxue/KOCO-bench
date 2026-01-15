/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "multinomial_operation.h"
#include "atb/utils/log.h"
#include "multinomial_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const uint32_t IN_TENSOR_NUM = 1;
static const uint64_t INPUT_TENSOR_DIM_NUM = 2;
static const uint32_t OUT_TENSOR_NUM = 1;
static const uint64_t OUT_TENSOR_DIM_NUM = 2;
static const uint64_t MAX_NUMSAMPLES = 64;


bool ParamCheck(const infer::MultinomialParam &opParam)
{
    if (opParam.numSamples > MAX_NUMSAMPLES) {
        ATB_LOG(ERROR) << "multinomial expects numSamples not bigger than " << MAX_NUMSAMPLES
                       << ", but got numSamples: " << opParam.numSamples;
        return false;
    }

    return true;
}

template <> Status CreateOperation(const infer::MultinomialParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    ATB_LOG(INFO) << "CreateOperation MultinomialParam numSamples: " << opParam.numSamples
                  << ", randSeed: " << opParam.randSeed;
    if (!(GetSingleton<Config>().Is910B())) {
        ATB_LOG(ERROR) << "multinomial only supports Atlas 800I A2/A3 inference product!";
        return ERROR_INVALID_PARAM;
    }
    if (!ParamCheck(opParam)) {
        return ERROR_INVALID_PARAM;
    }
    *operation = new MultinomialOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new multinomial operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

MultinomialOperation::MultinomialOperation(const infer::MultinomialParam &param)
    : OperationBase("MultinomialOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("MultinomialOperation");
}

MultinomialOperation::~MultinomialOperation() {}

uint32_t MultinomialOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t MultinomialOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status MultinomialOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                            SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(0).shape.dims[1] = param_.numSamples;
    outTensorDescs.at(0).shape.dimNum = OUT_TENSOR_DIM_NUM;
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dtype = ACL_INT32;
    return NO_ERROR;
}

Status MultinomialOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    return DimNumCheck(inTensorDescs.at(0));
}

Status MultinomialOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    Status ret = DimNumCheck(inTensors.at(0).desc);
    if (ret != NO_ERROR) {
        return ret;
    }
    uint64_t dimNum = inTensors.at(0).desc.shape.dimNum;
    int64_t lastDim = inTensors.at(0).desc.shape.dims[dimNum - 1];
    if (param_.numSamples > static_cast<uint64_t>(lastDim)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "numSamples shoud not be bigger than last dim, numSamples: " << param_.numSamples
                       << ", last dim: " << lastDim;
        return ERROR_INVALID_PARAM;
    }
    if (outTensors.at(0).desc.shape.dimNum != OUT_TENSOR_DIM_NUM) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensors dims is invalid, dimNum should be " << OUT_TENSOR_DIM_NUM << "but got: "
                                         << outTensors.at(0).desc.shape.dimNum;
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (outTensors.at(0).desc.shape.dims[1] != param_.numSamples) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensors dims is invalid, dims[1] should be param_.numSamples, but got dims[1]: "
                                         << outTensors.at(0).desc.shape.dims[1] << ", param_.numSamples" << param_.numSamples;
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (outTensors.at(0).desc.shape.dims[0] != inTensors.at(0).desc.shape.dims[0]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensors dims[0] should be equal to inTensors dims[0], but got outTensors dims[0]: "
                                         << outTensors.at(0).desc.shape.dims[0] << ", inTensors dims[0]" << inTensors.at(0).desc.shape.dims[0];
        return ERROR_INVALID_TENSOR_DIM;
    }

    return NO_ERROR;
}

std::shared_ptr<Runner> MultinomialOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<MultinomialOpsRunner>(param_);
}

Status MultinomialOperation::DimNumCheck(const TensorDesc &inTensorDesc) const
{
    uint64_t dimNum = inTensorDesc.shape.dimNum;
    
    if (dimNum != INPUT_TENSOR_DIM_NUM) {
        ATB_LOG(ERROR) << GetLogPrefix() << "dim size of inTensor should be 2, but inTensor dimNum is : " << dimNum;
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

nlohmann::json MultinomialOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb