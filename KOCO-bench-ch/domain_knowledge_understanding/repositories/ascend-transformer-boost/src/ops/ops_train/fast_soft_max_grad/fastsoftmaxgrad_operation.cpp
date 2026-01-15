/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fastsoftmaxgrad_operation.h"
#include "fastsoftmaxgrad_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/config.h"
#include "atb/operation/op_param_funcs.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/singleton.h"

namespace {
constexpr int32_t MAX_SEQLEN = 4096;
bool ParamCheck(const atb::train::FastSoftMaxGradParam &opParam)
{
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        ATB_LOG(ERROR) << "FastSoftMaxGradOperation is only supported in Atlas 800I A2 inference product.";
        return false;
    }

    if (!atb::OperationUtil::QSeqLenCheck(opParam.qSeqLen, MAX_SEQLEN)) {
        ATB_LOG(ERROR) << "qSeqLen is invalid for FastSoftMaxGradOperation.";
        return false;
    }

    if (opParam.headNum <= 0) {
        ATB_LOG(ERROR) << "headNum is invalid for FastSoftMaxGradOperation.";
        return false;
    }

    int64_t nSquareToken = 0;
    for (int32_t sampleSeqLen : opParam.qSeqLen) {
        nSquareToken += static_cast<int64_t>(sampleSeqLen) * sampleSeqLen;
    }
    if (nSquareToken > std::numeric_limits<int64_t>::max() / opParam.headNum) {
        ATB_LOG(ERROR) << "nSquareToken * headNum is overflow for FastSoftMaxGradOperation.";
        return false;
    }
    return true;
}

atb::Status CheckInTensorDescs(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                               const atb::train::FastSoftMaxGradParam &param)
{
    const uint64_t inDimNum = 1;
    if (!atb::TensorCheck::IsTensorDescDimNumValid(inTensorDescs.at(0), inDimNum) ||
        !atb::TensorCheck::IsTensorDescDimNumValid(inTensorDescs.at(1), inDimNum)) {
        ATB_LOG(ERROR) << "inTensor dim num is not support for FastSoftMaxGradOperation.";
        return atb::ERROR_INVALID_TENSOR_DIM_NUM;
    }

    int64_t nSquareToken = 0;
    for (int32_t sampleSeqLen : param.qSeqLen) {
        nSquareToken += static_cast<int64_t>(sampleSeqLen) * sampleSeqLen;
    }
    nSquareToken *= param.headNum;
    if (inTensorDescs.at(0).shape.dims[0] != nSquareToken || inTensorDescs.at(1).shape.dims[0] != nSquareToken) {
        ATB_LOG(ERROR) << "inTensor shape is not consistent with param.";
        return atb::ERROR_INVALID_TENSOR_DIM;
    }
    return atb::NO_ERROR;
}

atb::Status CheckInTensors(const atb::SVector<atb::Tensor> &inTensors, const atb::train::FastSoftMaxGradParam &param)
{
    const atb::SVector<atb::TensorDesc> inTensorDescs{inTensors.at(0).desc, inTensors.at(1).desc};
    return CheckInTensorDescs(inTensorDescs, param);
}

atb::Status CheckOutTensors(const atb::SVector<atb::Tensor> &inTensors, const atb::SVector<atb::Tensor> &outTensors)
{
    const uint64_t outDimNum = 1;
    if (!atb::TensorCheck::IsTensorDescDimNumValid(outTensors.at(0).desc, outDimNum)) {
        ATB_LOG(ERROR) << "outTensor dim num is not support for FastSoftMaxGradOperation.";
        return atb::ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensors.at(0).desc.shape.dims[0] != outTensors.at(0).desc.shape.dims[0]) {
        ATB_LOG(ERROR) << "outTensor shape is not consistent with inTensor for FastSoftMaxGradOperation.";
        return atb::ERROR_INVALID_TENSOR_SIZE;
    }
    return atb::NO_ERROR;
}
} // namespace
namespace atb {
OPERATION_PARAM_FUNCS(FastSoftMaxGradOperation, train::FastSoftMaxGradParam)

FastSoftMaxGradOperation::FastSoftMaxGradOperation(const train::FastSoftMaxGradParam &param)
    : OperationBase("FastSoftMaxGradOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("FastSoftMaxGradOperation");
}

FastSoftMaxGradOperation::~FastSoftMaxGradOperation() {}

uint32_t FastSoftMaxGradOperation::GetInputNum() const
{
    const uint32_t inTensorNum = 2;
    return inTensorNum;
}

uint32_t FastSoftMaxGradOperation::GetOutputNum() const
{
    const uint32_t outTensorNum = 1;
    return outTensorNum;
}

Status FastSoftMaxGradOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                                SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    return NO_ERROR;
}

Status FastSoftMaxGradOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    return CheckInTensorDescs(inTensorDescs, param_);
}

Status FastSoftMaxGradOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                                const SVector<Tensor> &outTensors) const
{
    Status status = CheckInTensors(inTensors, param_);
    if (status != NO_ERROR) {
        return status;
    }
    return CheckOutTensors(inTensors, outTensors);
}

std::shared_ptr<Runner> FastSoftMaxGradOperation::CreateRunner(Context &context) const
{
    ContextBase *contextBase = dynamic_cast<ContextBase *>(&context);
    if (!contextBase) {
        ATB_LOG(DEBUG) << "context cast to contextBase failed!";
        return nullptr;
    }
    RunnerPool &pool = contextBase->GetRunnerPool(RUNNER_TYPE_FASTSOFTMAXGRAD);
    Runner *runner = pool.MallocRunner<FastSoftMaxGradOpsRunner, train::FastSoftMaxGradParam>(param_);
    if (!runner) {
        ATB_LOG(DEBUG) << "MallocRunner from pool failed!";
        return std::make_shared<FastSoftMaxGradOpsRunner>(param_);
    }
    return std::shared_ptr<Runner>(runner, [&pool](Runner *runner) { pool.FreeRunner(runner); });
}

nlohmann::json FastSoftMaxGradOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}

train::FastSoftMaxGradParam FastSoftMaxGradOperation::GetParam() const
{
    return param_;
}

void FastSoftMaxGradOperation::SetParam(const train::FastSoftMaxGradParam &param)
{
    param_ = param;
    runner_ = nullptr;
}
} // namespace atb