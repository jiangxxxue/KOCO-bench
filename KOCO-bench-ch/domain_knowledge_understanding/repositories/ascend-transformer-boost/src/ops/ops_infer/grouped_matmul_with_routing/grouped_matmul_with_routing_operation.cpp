/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_matmul_with_routing_operation.h"
#include <atb/utils/log.h>
#include "grouped_matmul_with_routing_runner.h"
#include "atb/utils/singleton.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"


namespace atb {
static const uint32_t IN_TENSOR_ACTENSOR = 0;
static const uint32_t IN_TENSOR_EXPERTWEIGHT = 1;
static const uint32_t IN_TENSOR_EXPERTCOUNT = 2;
static const uint32_t IN_TENSOR_EXPERTINDEX = 3;
static const uint32_t IN_TENSOR_NSCALE = 4;
static const uint32_t IN_TENSOR_MSCALE = 5;
static const uint32_t WEIGHT_FIRST_DIM = 1;
static const uint32_t WEIGHT_SECOND_DIM = 2;

static const int32_t IN_TENSOR_NUM = 4;
static const int32_t QUANT_IN_TENSOR_NUM = 6;
static const int32_t OUT_TENSOR_NUM = 1;
static const int32_t TOPK_MIN_VALUE = 2;
static const int32_t TOPK_MAX_VALUE = 10;
static const int32_t ACTIVATION_MIN_VALUE = 128;
static const int32_t ACTIVATION_MAX_VALUE = 512;
static const int32_t WEIGHT_MIN_VALUE = 128;
static const int32_t WEIGHT_MAX_VALUE = 256;
static const int32_t UP_HIDDEN_IN_MIN_VALUE = 32;
static const int32_t UP_HIDDEN_IN_MAX_VALUE = 5120;
static const int32_t UP_HIDDEN_OUT_MIN_VALUE = 32;
static const int32_t UP_HIDDEN_OUT_MAX_VALUE = 256;
static const int32_t DOWN_HIDDEN_IN_MIN_VALUE = 32;
static const int32_t DOWN_HIDDEN_IN_MAX_VALUE = 256;
static const int32_t DOWN_HIDDEN_OUT_MIN_VALUE = 32;
static const int32_t DOWN_HIDDEN_OUT_MAX_VALUE = 5120;
static const int32_t ALIGHMENT_NUMBER = 32;

template <> Status CreateOperation(const infer::GroupedMatmulWithRoutingParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        ATB_LOG(ERROR) << "operation nullptr";
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (!GetSingleton<Config>().Is910B()) {
        ATB_LOG(ERROR) << "only support Atlas 800I A2 inference product";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.topK < TOPK_MIN_VALUE || opParam.topK > TOPK_MAX_VALUE) {
        ATB_LOG(ERROR) << "topk was not given, or it is out of the range from 2 to 10.";
        return ERROR_INVALID_PARAM;
    }
    if (!(opParam.outDataType == ACL_BF16 || opParam.outDataType == ACL_FLOAT16 ||
          opParam.outDataType == ACL_DT_UNDEFINED)) {
        ATB_LOG(ERROR) << "outdatatype invalid. Only takes ACL_DT_UNDEFINED, ACL_FLOAT16, ACL_BF16";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) GroupedMatmulWithRoutingOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to create operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

GroupedMatmulWithRoutingOperation::GroupedMatmulWithRoutingOperation(const infer::GroupedMatmulWithRoutingParam &param)
    : OperationBase("GroupedMatmulWithRoutingOperation"), param_(param)
{
    if (param.outDataType != ACL_DT_UNDEFINED) {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("GroupedMatmulWithRoutingQuantOperation");
    } else {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("GroupedMatmulWithRoutingOperation");
    }
}

GroupedMatmulWithRoutingOperation::~GroupedMatmulWithRoutingOperation() {}

uint32_t GroupedMatmulWithRoutingOperation::GetInputNum() const
{
    if (param_.outDataType != ACL_DT_UNDEFINED) {
        return QUANT_IN_TENSOR_NUM;
    }
    return IN_TENSOR_NUM;
}

uint32_t GroupedMatmulWithRoutingOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status GroupedMatmulWithRoutingOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                                         SVector<TensorDesc> &outTensorDescs) const
{
    if (param_.groupedMatmulType == infer::GroupedMatmulWithRoutingParam::GroupedMatmulType::GROUPED_MATMUL_UP) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0] * param_.topK;
    } else {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0] / param_.topK;
    }
    outTensorDescs.at(0).shape.dims[1] = OperationUtil::GetYTensorN(inTensorDescs.at(1), param_.transposeB);
    if (param_.outDataType != ACL_DT_UNDEFINED) {
        outTensorDescs.at(0).dtype = param_.outDataType;
    }
    return NO_ERROR;
}

Status GroupedMatmulWithRoutingOperation::OutTensorDimCheck(const SVector<TensorDesc> &inTensorDescs,
                                                            const SVector<Tensor> &outTensors) const
{
    TensorDesc resultTensor = outTensors.at(0).desc;
    if (param_.groupedMatmulType == infer::GroupedMatmulWithRoutingParam::GroupedMatmulType::GROUPED_MATMUL_UP) {
        if (resultTensor.shape.dims[0] != inTensorDescs.at(0).shape.dims[0] * param_.topK) {
            ATB_LOG(ERROR) << GetLogPrefix() << "outTensor0 first dim error";
            return ERROR_INVALID_TENSOR_DIM;
        }
    } else {
        if (resultTensor.shape.dims[0] * param_.topK != inTensorDescs.at(0).shape.dims[0]) {
            ATB_LOG(ERROR) << GetLogPrefix() << "outTensor0 first dim error";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    uint32_t outHiddenSize = inTensorDescs.at(1).shape.dims[WEIGHT_FIRST_DIM];
    if (!param_.transposeB) {
        outHiddenSize = inTensorDescs.at(1).shape.dims[WEIGHT_SECOND_DIM];
    }
    if (resultTensor.shape.dims[1] != outHiddenSize) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor0 second dim should be equal to hidden_size_out";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

bool GroupedMatmulWithRoutingOperation::TokenExpertTensorCheck(const TensorDesc &acTensorDesc,
                                                               const TensorDesc &expertWeightDesc) const
{
    uint32_t acTensorHiddenSize = acTensorDesc.shape.dims[1];
    uint32_t expertWeightHiddenInSize = expertWeightDesc.shape.dims[WEIGHT_SECOND_DIM];
    uint32_t expertWeightHiddenOutSize = expertWeightDesc.shape.dims[WEIGHT_FIRST_DIM];
    if (!param_.transposeB) {
        expertWeightHiddenInSize = expertWeightDesc.shape.dims[WEIGHT_FIRST_DIM];
        expertWeightHiddenOutSize = expertWeightDesc.shape.dims[WEIGHT_SECOND_DIM];
    }
    if (acTensorHiddenSize != expertWeightHiddenInSize) {
        ATB_LOG(ERROR) << GetLogPrefix() << "the second dim of intensor0 is not match the hidden size of intensor1";
        return false;
    }
    if (expertWeightHiddenInSize % ALIGHMENT_NUMBER != 0 || expertWeightHiddenOutSize % ALIGHMENT_NUMBER != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "The hidden size of intensor1 is not 32-bit aligned";
        return false;
    }
    if (param_.groupedMatmulType == infer::GroupedMatmulWithRoutingParam::GroupedMatmulType::GROUPED_MATMUL_UP) {
        if (acTensorHiddenSize < UP_HIDDEN_IN_MIN_VALUE || acTensorHiddenSize > UP_HIDDEN_IN_MAX_VALUE) {
            ATB_LOG(ERROR) << GetLogPrefix() << "The dimension of hidden size in exceeds the range from 32 to 5120";
            return false;
        }
        if (expertWeightHiddenOutSize < UP_HIDDEN_OUT_MIN_VALUE ||
            expertWeightHiddenOutSize > UP_HIDDEN_OUT_MAX_VALUE) {
            ATB_LOG(ERROR) << GetLogPrefix() << "The dimension of hidden size out exceeds the range from 32 to 256";
            return false;
        }
    } else {
        if (acTensorHiddenSize < DOWN_HIDDEN_IN_MIN_VALUE || acTensorHiddenSize > DOWN_HIDDEN_IN_MAX_VALUE) {
            ATB_LOG(ERROR) << GetLogPrefix() << "The dimension of hidden size in exceeds the range from 32 to 256";
            return false;
        }
        if (expertWeightHiddenOutSize < DOWN_HIDDEN_OUT_MIN_VALUE ||
            expertWeightHiddenOutSize > DOWN_HIDDEN_OUT_MAX_VALUE) {
            ATB_LOG(ERROR) << GetLogPrefix() << "The dimension of hidden size out exceeds the range from 32 to 5120";
            return false;
        }
    }
    return true;
}

bool GroupedMatmulWithRoutingOperation::ExpertCountTensorCheck(const TensorDesc &expertCount,
                                                               const TensorDesc &expertWeightDesc) const
{
    uint32_t nExperts = expertCount.shape.dims[0];
    if (nExperts != expertWeightDesc.shape.dims[0]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor1 first dim not equal to inTensor2";
        return false;
    }
    if (nExperts < WEIGHT_MIN_VALUE || nExperts > WEIGHT_MAX_VALUE) {
        ATB_LOG(ERROR) << GetLogPrefix() << "The number of expert exceeds the range from 128 to 256";
        return false;
    }
    return true;
}

bool GroupedMatmulWithRoutingOperation::ExpertIndexTensorCheck(const TensorDesc &acTensorDesc,
                                                               const TensorDesc &expertIndex) const
{
    uint32_t nTokens = acTensorDesc.shape.dims[0];
    if (param_.groupedMatmulType == infer::GroupedMatmulWithRoutingParam::GroupedMatmulType::GROUPED_MATMUL_UP) {
        if (nTokens * param_.topK != expertIndex.shape.dims[0]) {
            ATB_LOG(ERROR) << GetLogPrefix() << "inTensor3 first dim not equal to inTensor0 multiply topK";
            return false;
        }
    } else {
        nTokens /= param_.topK;
        if (acTensorDesc.shape.dims[0] != expertIndex.shape.dims[0]) {
            ATB_LOG(ERROR) << GetLogPrefix() << "inTensor3 first dim not equal to inTensor0";
            return false;
        }
    }
    if (nTokens < ACTIVATION_MIN_VALUE || nTokens > ACTIVATION_MAX_VALUE) {
        ATB_LOG(ERROR) << GetLogPrefix() << "The number of activation token exceeds the range from 128 to 512";
        return false;
    }
    return true;
}

bool GroupedMatmulWithRoutingOperation::NScaleTensorCheck(const TensorDesc &nScaleDesc,
                                                          const TensorDesc &expertWeightDesc) const
{
    if (nScaleDesc.shape.dims[0] != expertWeightDesc.shape.dims[0]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor4 first dim not equal to inTensor1";
        return false;
    }
    uint32_t expertWeightHiddenSize = expertWeightDesc.shape.dims[WEIGHT_FIRST_DIM];
    if (!param_.transposeB) {
        expertWeightHiddenSize = expertWeightDesc.shape.dims[WEIGHT_SECOND_DIM];
    }
    if (nScaleDesc.shape.dims[1] != expertWeightHiddenSize) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor4 second dim not equal to hidden size of inTensor1";
        return false;
    }
    return true;
}

bool GroupedMatmulWithRoutingOperation::MScaleTensorCheck(const TensorDesc &mScaleDesc,
                                                          const TensorDesc &acTensorDesc) const
{
    if (mScaleDesc.shape.dims[0] != acTensorDesc.shape.dims[0]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor5 first dim not equal to inTensor0";
        return false;
    }
    return true;
}

Status GroupedMatmulWithRoutingOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    TensorDesc acTensorDesc = inTensorDescs.at(IN_TENSOR_ACTENSOR);
    TensorDesc expertWeightDesc = inTensorDescs.at(IN_TENSOR_EXPERTWEIGHT);
    if (!TokenExpertTensorCheck(acTensorDesc, expertWeightDesc)) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (!ExpertCountTensorCheck(inTensorDescs.at(IN_TENSOR_EXPERTCOUNT), expertWeightDesc)) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (!ExpertIndexTensorCheck(acTensorDesc, inTensorDescs.at(IN_TENSOR_EXPERTINDEX))) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (param_.outDataType == ACL_DT_UNDEFINED) {
        return NO_ERROR;
    }
    if (!NScaleTensorCheck(inTensorDescs.at(IN_TENSOR_NSCALE), expertWeightDesc)) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (!MScaleTensorCheck(inTensorDescs.at(IN_TENSOR_MSCALE), acTensorDesc)) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status GroupedMatmulWithRoutingOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                                         const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs;
    for (size_t i = 0; i < inTensors.size(); i++) {
        inTensorDescs.push_back(inTensors.at(i).desc);
    }
    Status st = OutTensorDimCheck(inTensorDescs, outTensors);
    if (st != NO_ERROR) {
        return st;
    }
    return InferShapeCheckImpl(inTensorDescs);
}

nlohmann::json GroupedMatmulWithRoutingOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}

std::shared_ptr<Runner> GroupedMatmulWithRoutingOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<GroupedMatmulWithRoutingRunner>(param_);
}

} // namespace atb