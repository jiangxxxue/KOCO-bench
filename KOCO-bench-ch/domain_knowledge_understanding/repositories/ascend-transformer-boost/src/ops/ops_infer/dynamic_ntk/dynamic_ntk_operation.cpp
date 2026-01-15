/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dynamic_ntk_operation.h"
#include "atb/utils/param_to_json.h"
#include "dynamic_ntk_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/config.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/operation_util.h"
#include "atb/operation/op_param_funcs.h"

static const uint32_t IN_TENSOR_SEQLENS = 2;
static const uint32_t OUT_TENSOR_NUM = 2;
static const uint32_t IN_TENSOR_NUM = 3;
static const uint32_t OUT_TENSOR_DIM_NUM = 2;
static const uint32_t ALIGNMENT = 32;
static const uint32_t MAX_BATCH_SIZE = 16;
static const uint32_t MAX_HEAD_DIM = 2048;
static const uint32_t MAX_NUM_TOKENS = 256000;

namespace atb {
template <> Status CreateOperation(const infer::DynamicNTKParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (opParam.outDataType != ACL_FLOAT16 && opParam.outDataType != ACL_BF16) {
        ATB_LOG(ERROR) << "not support aclOutTensorType:" << opParam.outDataType;
        return ERROR_INVALID_PARAM;
    }
    if (opParam.outDataType == ACL_BF16 && !GetSingleton<Config>().Is910B()) {
        ATB_LOG(ERROR) << "only Atlas 800I A2 inference product supports bf16";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) DynamicNTKOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

DynamicNTKOperation::DynamicNTKOperation(const infer::DynamicNTKParam &param)
    : OperationBase("DynamicNTKOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("DynamicNTKOperation");
}

DynamicNTKOperation::~DynamicNTKOperation() {}

uint32_t DynamicNTKOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t DynamicNTKOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status DynamicNTKOperation::InferShapeDimCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    int64_t batch = inTensorDescs.at(IN_TENSOR_SEQLENS).shape.dims[0];
    if (batch != inTensorDescs.at(1).shape.dims[0] || batch > MAX_BATCH_SIZE) {
        ATB_LOG(ERROR) << GetLogPrefix() << "first dim of InvFreqIn and seqlens should be equal"
                       << "and should be no greater than 16";
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t ntokens = inTensorDescs.at(0).shape.dims[0];
    if (ntokens > MAX_NUM_TOKENS) {
        ATB_LOG(ERROR) << GetLogPrefix() << "shape of positionIds should be no greater than 256000";
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t headDim = inTensorDescs.at(1).shape.dims[1] * 2; // 输入InvFreqIn的大小是[head_dim / 2]
    if (headDim > MAX_HEAD_DIM) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "head_dim should be no greater than 2048,"
                          "last dim of InvFreqIn tensor should be no greater than 1024.";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if ((static_cast<uint64_t>(headDim) % static_cast<uint64_t>(ALIGNMENT) != 0)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "head_dim is not support, should align 32";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status DynamicNTKOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(0).shape.dimNum != 1 || inTensorDescs.at(IN_TENSOR_SEQLENS).shape.dimNum != 1) {
        ATB_LOG(ERROR) << GetLogPrefix() << "dim numbers of positionIds and seqlens should be one";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDescs.at(1).shape.dimNum != 2) { // 2:[batch, head_dim/2]
        ATB_LOG(ERROR) << GetLogPrefix() << "dim number of InvFreqIn should be two";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    Status st = NO_ERROR;
    st = InferShapeDimCheck(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    return NO_ERROR;
}

Status DynamicNTKOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                           SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0).format = ACL_FORMAT_ND;
    outTensorDescs.at(0).dtype = param_.outDataType;
    outTensorDescs.at(0).shape.dimNum = OUT_TENSOR_DIM_NUM;
    outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
    outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(1).shape.dims[1] * 2; // 输入InvFreqIn的大小是[head_dim / 2]
    outTensorDescs.at(1) = outTensorDescs.at(0);
    return NO_ERROR;
}

Status DynamicNTKOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs = {};
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    Status st = NO_ERROR;
    st = InferShapeCheckImpl(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    TensorDesc sinTensorDesc = outTensors.at(0).desc;
    TensorDesc cosTensorDesc = outTensors.at(1).desc;
    if (cosTensorDesc.dtype != param_.outDataType) {
        ATB_LOG(ERROR) << "outTensor dtype should be equal to outDataType";
        return ERROR_INVALID_TENSOR_DTYPE;
    }
    if (cosTensorDesc.shape.dimNum != OUT_TENSOR_DIM_NUM) {
        ATB_LOG(ERROR) << "outTensor dimNum shoudld be 2";
        return ERROR_INVALID_TENSOR_SIZE;
    }
    if (cosTensorDesc.shape.dims[0] != inTensors.at(0).desc.shape.dims[0]) {
        ATB_LOG(ERROR) << "outTensors dims error, outTensor fistDim shoud equal to inTensor0 fistDim";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (cosTensorDesc.shape.dims[1] != inTensors.at(1).desc.shape.dims[1] * 2) { // 输入InvFreqIn的大小是[head_dim / 2]
        ATB_LOG(ERROR) << "outTensors dims error, outTensor lastDim shoud equal to inTensor1 lastDim * 2";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (!TensorUtil::TensorDescEqual(sinTensorDesc, cosTensorDesc)) {
        ATB_LOG(ERROR) << "output sinTensor shape should be the same as cosTensor";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> DynamicNTKOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<DynamicNTKOpsRunner>(param_);
}

nlohmann::json DynamicNTKOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb