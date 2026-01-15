/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "topk_topp_sampling_operation.h"
#include "atb/utils/tensor_check.h"
#include "topk_topp_sampling_ops_runner.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/singleton.h"
#include "atb/utils/tensor_util.h"
#include "atb/operation/op_param_funcs.h"
namespace {
static const uint32_t SINGLE_TOPK_IN_TENSOR_NUM = 2;
static const uint32_t BATCH_TOPK_MULTI_IN_TENSOR_NUM = 3;
static const uint32_t BATCH_TOPK_EXP_IN_TENSOR_NUM = 4;
static const uint32_t BATCH_TOPK_MULTI_LOGPROBS_IN_TENSOR_NUM = 4;
static const uint32_t OUT_TENSOR_NUM = 2;
static const uint32_t LOG_PROBS_OUT_TENSOR_NUM = 3;
static const uint32_t INTENSOR_TOPK = 1;
static const uint32_t INTENSOR_TOPP = 2;
static const uint32_t INTENSOR_EXP = 3;
static const uint32_t OUT_TENSOR_LOGPROBS = 2;
static const uint32_t MAX_BATCH_SIZE = 512;
static const uint32_t PROBS_TENSOR_INDEX = 0;
static const uint32_t RAND_TENSOR_INDEX = 3;
static const uint32_t TENSER_FIRST_DIM = 0;
static const uint32_t TENSER_VOC_DIM = 1;
static const int32_t MAX_LOGPROBS_SIZE = 16384;
static const int32_t MIN_LOGPROBS_SIZE = 0;
static const uint32_t INTENSOR_RAND = 2;
static const uint32_t RAND_TENSOR_LAST_INDEX = 1;
static const uint32_t RAND_TENSOR_LAST_DIM = 1;
static const uint32_t LOG_PROBS_OUT_TENSOR_INDEX = 2;
static const uint32_t LOG_PROBS_OUT_TENSOR_DIM = 2;
static const uint32_t LAST_DIM = 1;

using atbInferTopkToppSamplingType = atb::infer::TopkToppSamplingParam::TopkToppSamplingType;

bool ParamCheck(const atb::infer::TopkToppSamplingParam &opParam)
{
    if (opParam.topkToppSamplingType <= atb::infer::TopkToppSamplingParam::SAMPLING_UNDEFINED ||
        opParam.topkToppSamplingType >= atb::infer::TopkToppSamplingParam::SAMPLING_MAX) {
        ATB_LOG(ERROR) << "topkToppSamplingType:" << opParam.topkToppSamplingType << "is invalid topkToppSamplingType";
        return false;
    }
    return true;
}
} // namespace

namespace atb {
OPERATION_PARAM_FUNCS(TopkToppSamplingOperation, infer::TopkToppSamplingParam)

static Mki::OperationIr *GetOperationIrForTopkToppSampling(const infer::TopkToppSamplingParam &param)
{
    switch (param.topkToppSamplingType) {
        case atbInferTopkToppSamplingType::BATCH_TOPK_EXPONENTIAL_SAMPLING:
            return GetSingleton<AtbOperationIrCfg>().GetOperationIr("TopkToppSamplingBatchTopKExpOperation");
        case atbInferTopkToppSamplingType::BATCH_TOPK_EXPONENTIAL_LOGPROBS_SAMPLING:
            return GetSingleton<AtbOperationIrCfg>().GetOperationIr("TopkToppSamplingBatchTopKLogProbsExpOperation");
        case atbInferTopkToppSamplingType::BATCH_TOPK_MULTINOMIAL_SAMPLING:
            return GetSingleton<AtbOperationIrCfg>().GetOperationIr("TopkToppSamplingBatchTopKMultiOperation");
        case atbInferTopkToppSamplingType::BATCH_TOPK_MULTINOMIAL_LOGPROBS_SAMPLING:
            return GetSingleton<AtbOperationIrCfg>().GetOperationIr("TopkToppSamplingBatchTopKLogProbsMultiOperation");
        case atbInferTopkToppSamplingType::SINGLE_TOPK_SAMPLING:
            return GetSingleton<AtbOperationIrCfg>().GetOperationIr("TopkToppSamplingSingleTopKOperation");
        default:
            ATB_LOG(ERROR) << "UnSupported TopkToppSamplingType: " << param.topkToppSamplingType;
    }
    return nullptr;
}

TopkToppSamplingOperation::TopkToppSamplingOperation(const infer::TopkToppSamplingParam &param)
    : OperationBase("TopkToppSamplingOperation"), param_(param)
{
    operationIr_ = GetOperationIrForTopkToppSampling(param_);
    if (!operationIr_) {
        ATB_LOG(ERROR) << "GetOperationIrForTopkToppSampling failed.";
    }
}

TopkToppSamplingOperation::~TopkToppSamplingOperation() {}

uint32_t TopkToppSamplingOperation::GetInputNum() const
{
    switch (param_.topkToppSamplingType) {
        case atbInferTopkToppSamplingType::BATCH_TOPK_EXPONENTIAL_SAMPLING:
            return BATCH_TOPK_EXP_IN_TENSOR_NUM;
        case atbInferTopkToppSamplingType::BATCH_TOPK_MULTINOMIAL_SAMPLING:
            return BATCH_TOPK_MULTI_IN_TENSOR_NUM;
        case atbInferTopkToppSamplingType::BATCH_TOPK_EXPONENTIAL_LOGPROBS_SAMPLING:
            return BATCH_TOPK_EXP_IN_TENSOR_NUM;
        case atbInferTopkToppSamplingType::BATCH_TOPK_MULTINOMIAL_LOGPROBS_SAMPLING:
            return BATCH_TOPK_MULTI_LOGPROBS_IN_TENSOR_NUM;
        case atbInferTopkToppSamplingType::SINGLE_TOPK_SAMPLING:
            return SINGLE_TOPK_IN_TENSOR_NUM;
        default:
            ATB_LOG(ERROR) << GetLogPrefix() << "UnSupported TopkToppSamplingType: " << param_.topkToppSamplingType;
            return 0;
    }
}

uint32_t TopkToppSamplingOperation::GetOutputNum() const
{
    switch (param_.topkToppSamplingType) {
        case atbInferTopkToppSamplingType::BATCH_TOPK_EXPONENTIAL_LOGPROBS_SAMPLING:
            return LOG_PROBS_OUT_TENSOR_NUM;
        case atbInferTopkToppSamplingType::BATCH_TOPK_MULTINOMIAL_LOGPROBS_SAMPLING:
            return LOG_PROBS_OUT_TENSOR_NUM;
        default:
            return OUT_TENSOR_NUM;
    }
}

Status TopkToppSamplingOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                                 SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    size_t dimNum = outTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).shape.dims[dimNum - 1] = 1;
    outTensorDescs.at(0).dtype = ACL_INT32;

    outTensorDescs.at(1) = inTensorDescs.at(0);
    outTensorDescs.at(1).shape.dims[dimNum - 1] = 1;

    if (param_.topkToppSamplingType == atbInferTopkToppSamplingType::BATCH_TOPK_EXPONENTIAL_LOGPROBS_SAMPLING ||
        param_.topkToppSamplingType == atbInferTopkToppSamplingType::BATCH_TOPK_MULTINOMIAL_LOGPROBS_SAMPLING) {
        outTensorDescs.at(OUT_TENSOR_LOGPROBS) = inTensorDescs.at(0);
        outTensorDescs.at(OUT_TENSOR_LOGPROBS).dtype = ACL_FLOAT;
        outTensorDescs.at(OUT_TENSOR_LOGPROBS).shape.dims[dimNum - 1] = param_.logProbsSize;
    }
    return NO_ERROR;
}

Status TopkToppSamplingOperation::CheckBatchSize(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(0).shape.dims[0] > MAX_BATCH_SIZE) {
        ATB_LOG(ERROR) << "batch size of inTensor0: " << inTensorDescs.at(0).shape.dims[0]
                       << " is greater than max batch size " << MAX_BATCH_SIZE;
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status TopkToppSamplingOperation::CheckSingleTopk(const SVector<TensorDesc> &inTensorDescs) const
{
    if (param_.topk > inTensorDescs.at(0).shape.dims[inTensorDescs.at(0).shape.dimNum - 1]) {
        ATB_LOG(ERROR) << "topk " << param_.topk << " should be 0 or less than or equal probs dimNum("
                       << inTensorDescs.at(0).shape.dimNum << ")";
        return ERROR_INVALID_PARAM;
    }
    if (inTensorDescs.at(1).shape.dims[1] != 1) {
        ATB_LOG(ERROR) << GetLogPrefix() << "size of topp: " << inTensorDescs.at(1).shape.dims[1] << " should be 1";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return CheckBatchSize(inTensorDescs);
}

Status TopkToppSamplingOperation::CheckBatchTopkTopp(const SVector<TensorDesc> &inTensorDescs) const
{
    int64_t probsBatchSize = inTensorDescs.at(0).shape.dims[0];
    int64_t topkBatchSize = inTensorDescs.at(INTENSOR_TOPK).shape.dims[0];
    int64_t toppBatchSize = inTensorDescs.at(INTENSOR_TOPP).shape.dims[0];
    int64_t topkSize = inTensorDescs.at(INTENSOR_TOPK).shape.dims[1];
    int64_t toppSize = inTensorDescs.at(INTENSOR_TOPP).shape.dims[1];
    if (!(probsBatchSize == topkBatchSize && topkBatchSize == toppBatchSize)) {
        ATB_LOG(ERROR) << "batch size of inTensor0: " << probsBatchSize
                       << ", batch size of inTensor1: " << topkBatchSize
                       << ", batch size of inTensor2: " << toppBatchSize << " should be same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (!(topkSize == 1 && toppSize == 1)) {
        ATB_LOG(ERROR) << "size of topk: " << topkSize << " and size of topp: " << toppSize << " should be 1";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return CheckBatchSize(inTensorDescs);
}

Status TopkToppSamplingOperation::CheckMutinomial(const SVector<TensorDesc> &inTensorDescs) const
{
    if (param_.randSeeds.size() != static_cast<size_t>(inTensorDescs.at(0).shape.dims[0])) {
        ATB_LOG(ERROR) << "size of randSeeds: " << param_.randSeeds.size() << " should equal to probs batch size("
                       << inTensorDescs.at(0).shape.dims[0] << ")";
        return ERROR_INVALID_PARAM;
    }
    return CheckBatchTopkTopp(inTensorDescs);
}

Status TopkToppSamplingOperation::CheckLogProbsSize(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(PROBS_TENSOR_INDEX).shape.dims[TENSER_VOC_DIM] < param_.logProbsSize) {
        ATB_LOG(ERROR) << "logProbsSize: " << param_.logProbsSize << " should be less than or equal voc_size("
                       << inTensorDescs.at(PROBS_TENSOR_INDEX).shape.dims[TENSER_VOC_DIM] << ")";
        return ERROR_INVALID_PARAM;
    }
    if (param_.logProbsSize < MIN_LOGPROBS_SIZE || param_.logProbsSize > MAX_LOGPROBS_SIZE) {
        ATB_LOG(ERROR) << "logProbsSize: " << param_.logProbsSize << " shoud be in range [" << MIN_LOGPROBS_SIZE << ", "
                       << MAX_LOGPROBS_SIZE << "]";
        return ERROR_INVALID_PARAM;
    }
    return NO_ERROR;
}

Status TopkToppSamplingOperation::CheckMutinomialLogProbs(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(PROBS_TENSOR_INDEX).shape.dims[TENSER_FIRST_DIM] !=
        inTensorDescs.at(RAND_TENSOR_INDEX).shape.dims[TENSER_FIRST_DIM]) {
        ATB_LOG(ERROR) << "randTensor batch size: " << inTensorDescs.at(RAND_TENSOR_INDEX).shape.dims[TENSER_FIRST_DIM]
                       << " should equal to probs batch size("
                       << inTensorDescs.at(PROBS_TENSOR_INDEX).shape.dims[TENSER_FIRST_DIM] << ")";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(RAND_TENSOR_INDEX).shape.dimNum != INTENSOR_RAND) {
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDescs.at(RAND_TENSOR_INDEX).shape.dims[RAND_TENSOR_LAST_INDEX] != RAND_TENSOR_LAST_DIM) {
        return ERROR_INVALID_TENSOR_DIM;
    }

    Status logProbsSizeCheckRes = CheckLogProbsSize(inTensorDescs);
    if (logProbsSizeCheckRes != NO_ERROR) {
        return logProbsSizeCheckRes;
    }
    return CheckBatchTopkTopp(inTensorDescs);
}

Status TopkToppSamplingOperation::CheckExponentialLogProbs(const SVector<TensorDesc> &inTensorDescs) const
{
    Status logProbsSizeCheckRes = CheckLogProbsSize(inTensorDescs);
    if (logProbsSizeCheckRes != NO_ERROR) {
        return logProbsSizeCheckRes;
    }
    return CheckExponential(inTensorDescs);
}

Status TopkToppSamplingOperation::CheckExponential(const SVector<TensorDesc> &inTensorDescs) const
{
    if (!TensorUtil::TensorDescEqual(inTensorDescs.at(0), inTensorDescs.at(INTENSOR_EXP))) {
        ATB_LOG(ERROR) << "Shape of intensor3 should be same as intensor0";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return CheckBatchTopkTopp(inTensorDescs);
}

Status TopkToppSamplingOperation::CheckIntensorAndParam(const SVector<TensorDesc> &inTensorDescs) const
{
    for (size_t i = 0; i < inTensorDescs.size(); i++) {
        if (!TensorCheck::IsTensorDescDimNumValid(inTensorDescs.at(i), 2)) { // 2 = [batch, voc_size] / [batch, 1]
            ATB_LOG(ERROR) << GetLogPrefix() << "inTensor dim num is not support, dim num of inTensor only support 2";
            return ERROR_INVALID_TENSOR_DIM_NUM;
        }
    }
    switch (param_.topkToppSamplingType) {
        case atbInferTopkToppSamplingType::SINGLE_TOPK_SAMPLING:
            return CheckSingleTopk(inTensorDescs);
        case atbInferTopkToppSamplingType::BATCH_TOPK_MULTINOMIAL_SAMPLING:
            return CheckMutinomial(inTensorDescs);
        case atbInferTopkToppSamplingType::BATCH_TOPK_EXPONENTIAL_SAMPLING:
            return CheckExponential(inTensorDescs);
        case atbInferTopkToppSamplingType::BATCH_TOPK_EXPONENTIAL_LOGPROBS_SAMPLING:
            return CheckExponentialLogProbs(inTensorDescs);
        case atbInferTopkToppSamplingType::BATCH_TOPK_MULTINOMIAL_LOGPROBS_SAMPLING:
            return CheckMutinomialLogProbs(inTensorDescs);
        default:
            ATB_LOG(ERROR) << GetLogPrefix() << "UnSupported TopkToppSamplingType: " << param_.topkToppSamplingType;
            return ERROR_INVALID_PARAM;
    }
}

Status TopkToppSamplingOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    return CheckIntensorAndParam(inTensorDescs);
}

Status TopkToppSamplingOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                                 const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs = {};
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    SVector<TensorDesc> outTensorDescs = {};
    OperationUtil::InTensorsToInTensorDescs(outTensors, outTensorDescs);
    ATB_LOG(DEBUG) << "outTensors size:" << outTensors.size();
    for (size_t i = 0; i < outTensorDescs.size(); i++) {
        if (!TensorCheck::IsTensorDescDimNumValid(outTensorDescs.at(i), 2)) {
            if ((param_.topkToppSamplingType == atbInferTopkToppSamplingType::BATCH_TOPK_EXPONENTIAL_LOGPROBS_SAMPLING ||
                 param_.topkToppSamplingType == atbInferTopkToppSamplingType::BATCH_TOPK_MULTINOMIAL_LOGPROBS_SAMPLING) &&
                 param_.logProbsSize == 0 && i == outTensorDescs.size() - 1) {
                    ATB_LOG(INFO) << GetLogPrefix() << "the dimNum of outTensor[" << i << "] allows not be 2 when logProbsSize is 0";
            } else {
                ATB_LOG(ERROR) << GetLogPrefix() << "the dimNum of outTensor[" << i <<"] should be 2";
                return ERROR_INVALID_TENSOR_DIM_NUM;
            }
        }
    }
    if (!(outTensorDescs.at(0).shape.dims[0] == outTensorDescs.at(1).shape.dims[0])) {
        ATB_LOG(ERROR) << "The batch size of outTensors should be same.";
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (!(outTensorDescs.at(0).shape.dims[1] == outTensorDescs.at(1).shape.dims[1])) {
        ATB_LOG(ERROR) << "The vocabulary size of outTensors should be same.";
        return ERROR_INVALID_TENSOR_DIM;
    } else {
        if (!(outTensorDescs.at(0).shape.dims[1] == 1)) {
            ATB_LOG(ERROR) << "The vocabulary size of outTensors should be 1.";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    if (param_.topkToppSamplingType == atbInferTopkToppSamplingType::BATCH_TOPK_EXPONENTIAL_LOGPROBS_SAMPLING ||
        param_.topkToppSamplingType == atbInferTopkToppSamplingType::BATCH_TOPK_MULTINOMIAL_LOGPROBS_SAMPLING) {
        if (param_.logProbsSize != 0 && outTensorDescs.at(LOG_PROBS_OUT_TENSOR_INDEX).shape.dims[LAST_DIM] != param_.logProbsSize) {
            ATB_LOG(ERROR) << "the last dim of logProbsTensor dim:" << outTensorDescs.at(LOG_PROBS_OUT_TENSOR_INDEX).shape.dims[LAST_DIM]
                           << ", which should be same as " << param_.logProbsSize;
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return CheckIntensorAndParam(inTensorDescs);
}

std::shared_ptr<Runner> TopkToppSamplingOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<TopkToppSamplingOpsRunner>(param_);
}

nlohmann::json TopkToppSamplingOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}

infer::TopkToppSamplingParam TopkToppSamplingOperation::GetParam() const
{
    return param_;
}

void TopkToppSamplingOperation::SetParam(const infer::TopkToppSamplingParam &param)
{
    param_ = param;
    runner_ = nullptr;
    operationIr_ = GetOperationIrForTopkToppSampling(param_);
    if (!operationIr_) {
        ATB_LOG(ERROR) << "GetOperationIrForTopkToppSampling failed.";
    }
}

} // namespace atb
