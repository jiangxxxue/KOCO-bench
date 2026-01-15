/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rope_q_concat_operation.h"
#include "atb/utils/config.h"
#include "rope_q_concat_ops_runner.h"
#include "atb/utils/log.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils/singleton.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const int32_t IN_TENSOR_NUM = 4;
static const int32_t OUT_TENSOR_NUM = 1;
static const int32_t PARAM_COS = 2;
static const int64_t MAX_HEAD_DIM = 64;
static const int64_t FACTOR_16 = 16;

template <> Status CreateOperation(const infer::RopeQConcatParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    *operation = new (std::nothrow) RopeQConcatOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

RopeQConcatOperation::RopeQConcatOperation(const infer::RopeQConcatParam &param)
    : OperationBase("RopeQConcatOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("RopeQConcatOperation");
}

RopeQConcatOperation::~RopeQConcatOperation() {}

uint32_t RopeQConcatOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t RopeQConcatOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status RopeQConcatOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                            SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(3);                              // index 3: concatinput tensor
    outTensorDescs.at(0).shape.dims[2] += inTensorDescs.at(1).shape.dims[1]; // 2: last dim; 1: cos head_dim
    return NO_ERROR;
}

Status RopeQConcatOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    return DimCheck(inTensorDescs);
}

Status RopeQConcatOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs;
    for (size_t i = 0; i < inTensors.size(); i++) {
        inTensorDescs.push_back(inTensors.at(i).desc);
    }

    Status st = DimCheck(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }

    ATB_LOG(DEBUG) << "outTensors Size:" << outTensors.size();
    if (outTensors.at(0).desc.shape.dimNum != 3) { // outtensor0's dim num must be 3
        ATB_LOG(ERROR) << "outTensor's dim shoule be 3 but is " << outTensors.at(0).desc.shape.dimNum;
        return ERROR_INVALID_TENSOR_DIM;
    } else {
        auto inputTokens = inTensorDescs.at(0).shape.dims[0];
        auto headDim = inTensorDescs.at(1).shape.dims[1];
        auto concatSize = inTensorDescs.at(3).shape.dims[2];
        auto outputTokens = outTensors.at(0).desc.shape.dims[0];
        auto outputDim2 = outTensors.at(0).desc.shape.dims[2];
        if (inputTokens != outputTokens) {
            ATB_LOG(ERROR) << "outTensor's ntoken dim should be the same as intensor's ntoken dim";
            return ERROR_INVALID_TENSOR_DIM;
        }
        if (outputDim2 - concatSize != headDim) {
            ATB_LOG(ERROR) << "outTensor's dim 2 should be head_dim + concat_size";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return NO_ERROR;
}

Status RopeQConcatOperation::DimCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    if (inTensorDescs.at(0).shape.dimNum != 2 || // tensor q's dim num should be 2
        inTensorDescs.at(1).shape.dimNum != 2 || // tensor cos's dim num should be 2
        inTensorDescs.at(2).shape.dimNum != 2 || // tensor sin's dim num should be 2
        inTensorDescs.at(3).shape.dimNum != 3) { // tensor concat's dim num should be 3
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor dimNum is not supported, tensor0/1/2 only support 2 dimNum,"
                       << " and tensor3 only support 3 dimNum";
        return ERROR_INVALID_TENSOR_DIM;
    }
    auto ntokens = inTensorDescs.at(0).shape.dims[0];
    if (inTensorDescs.at(1).shape.dims[0] != ntokens || // intensor[1] is tensor cos
        inTensorDescs.at(2).shape.dims[0] != ntokens || // intensor[2] is tensor sin
        inTensorDescs.at(3).shape.dims[0] != ntokens) { // intensor[3] is tensor concat
        ATB_LOG(ERROR) << GetLogPrefix() << "all input tensor's ntokens(dims[0]) must be the same";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(1).shape.dims[1] != inTensorDescs.at(2).shape.dims[1]) { // intensor[2] is tensor sin
        ATB_LOG(ERROR) << GetLogPrefix() << "tensor1 and tensor2 must have the same head_dim(dims[1])";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(1).shape.dims[1] > MAX_HEAD_DIM ||
        inTensorDescs.at(1).shape.dims[1] % FACTOR_16 != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "head dim must be a multiple of 16 and less than or equal to 64.";
        return ERROR_INVALID_TENSOR_DIM;
    }
    // inTensorDescs.at(3).shape.dims[1] != 0 is already checked, and // intensor[2] is tensor sin
    if (inTensorDescs.at(0).shape.dims[1] / inTensorDescs.at(3).shape.dims[1] != inTensorDescs.at(2).shape.dims[1]) {
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "tensor0's hidden_size(dims[1]) must be equal to tensor1 and tensor2's head_dim(dims[1]) * "
                       << "tensor3's head_num(dims[1])";
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t maxUbSize = 196352;
    auto headDim = inTensorDescs.at(1).shape.dims[1];
    auto concatSize = inTensorDescs.at(3).shape.dims[2];
    if (headDim % 16 != 0 || concatSize % 16 != 0) { // 16: dim alignment(32 byte / sizeof(float16))
        ATB_LOG(ERROR) << "head_dim or concat_size must be devided by 16";
        return ERROR_INVALID_TENSOR_DIM;
    }
    // q:4+dataTypeSize、reverseq 4、neg 4、sin 4+dataTypeSize、cos 4+dataTypeSize concat dataTypeSize
    int32_t dataTypeSize = 2; // float16 bf16
    int32_t kTempSpaceMultiplier = 3; // q sin cos
    int32_t kTempSpace = dataTypeSize + 4; // q sin cos
    int32_t kAdditionalTempSize = 2 * 4; // reverse neg
    if (headDim * (kTempSpaceMultiplier * kTempSpace + kAdditionalTempSize) + concatSize * dataTypeSize >= maxUbSize) {
        ATB_LOG(ERROR)
            << "headDim * (3 * (4 + sizeof(float16)) + 2 * 4) + concatSize\
                * sizeof(float16) must be less than maxUbSize(196352)";
        return ERROR_INVALID_TENSOR_DIM;
    }
    for (size_t i = 0; i < IN_TENSOR_NUM; i++) {
        auto inTensor = inTensorDescs.at(i);
        size_t dimNum = inTensor.shape.dimNum;
        for (size_t dimIndex = 0; dimIndex < dimNum; dimIndex++) {
            if (inTensor.shape.dims[dimIndex] == 0) { // any dim should not be equal to zero !
                ATB_LOG(ERROR) << "Any dimension of the input vector cannot be equal to zero.";
                return ERROR_INVALID_TENSOR_DIM;
            }
        }
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> RopeQConcatOperation::CreateRunner(Context &context) const
{
    ContextBase *contextBase = dynamic_cast<ContextBase *>(&context);
    if (!contextBase) {
        ATB_LOG(DEBUG) << "context cast to contextBase failed!";
        return nullptr;
    }
    RunnerPool &pool = contextBase->GetRunnerPool(RUNNER_TYPE_ROPE_Q_CONCAT);
    Runner *runner = pool.MallocRunner<RopeQConcatOpsRunner, infer::RopeQConcatParam>(param_);
    if (!runner) {
        ATB_LOG(DEBUG) << "MallocRunner from pool failed!";
        return std::make_shared<RopeQConcatOpsRunner>(param_);
    }
    return std::shared_ptr<Runner>(runner, [poolPtr = &pool](Runner *runner) { poolPtr->FreeRunner(runner); });
}

nlohmann::json RopeQConcatOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb
