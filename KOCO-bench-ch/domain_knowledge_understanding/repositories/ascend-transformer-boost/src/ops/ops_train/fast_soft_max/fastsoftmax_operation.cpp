/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fastsoftmax_operation.h"
#include "fastsoftmax_ops_runner.h"
#include "atb/utils/config.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/singleton.h"

namespace {
constexpr int32_t MAX_SEQLEN = 4096;
bool ParamCheck(const atb::train::FastSoftMaxParam &opParam)
{
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        ATB_LOG(ERROR) << "FastSoftMaxOperation is only supported in Atlas 800I A2 inference product.";
        return false;
    }

    if (!atb::OperationUtil::QSeqLenCheck(opParam.qSeqLen, MAX_SEQLEN)) {
        ATB_LOG(ERROR) << "qSeqLen is invalid for FastSoftMaxOperation.";
        return false;
    }

    if (opParam.headNum <= 0) {
        ATB_LOG(ERROR) << "headNum is invalid for FastSoftMaxOperation.";
        return false;
    }

    int64_t nSquareToken = 0;
    for (int32_t sampleSeqLen : opParam.qSeqLen) {
        nSquareToken += static_cast<int64_t>(sampleSeqLen) * sampleSeqLen;
    }
    if (nSquareToken > std::numeric_limits<int64_t>::max() / opParam.headNum) {
        ATB_LOG(ERROR) << "nSquareToken * headNum is overflow for FastSoftMaxOperation.";
        return false;
    }
    return true;
}

atb::Status CheckInTensorDescs(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                               const atb::train::FastSoftMaxParam &param)
{
    const uint64_t inDimNum = 1;
    if (!atb::TensorCheck::IsTensorDescDimNumValid(inTensorDescs.at(0), inDimNum)) {
        ATB_LOG(ERROR) << "inTensor dim num is not support for FastSoftMaxOperation.";
        return atb::ERROR_INVALID_TENSOR_DIM_NUM;
    }

    int64_t nSquareToken = 0;
    for (int32_t sampleSeqLen : param.qSeqLen) {
        nSquareToken += static_cast<int64_t>(sampleSeqLen) * sampleSeqLen;
    }
    nSquareToken *= param.headNum;
    if (inTensorDescs.at(0).shape.dims[0] != nSquareToken) {
        ATB_LOG(ERROR) << "inTensor shape is not consistent with param.";
        return atb::ERROR_INVALID_TENSOR_SIZE;
    }
    return atb::NO_ERROR;
}

atb::Status CheckInTensors(const atb::SVector<atb::Tensor> &inTensors, const atb::train::FastSoftMaxParam &param)
{
    const atb::SVector<atb::TensorDesc> inTensorDescs{inTensors.at(0).desc};
    return CheckInTensorDescs(inTensorDescs, param);
}

atb::Status CheckOutTensors(const atb::SVector<atb::Tensor> &inTensors, const atb::SVector<atb::Tensor> &outTensors)
{
    const uint64_t outDimNum = 1;
    if (!atb::TensorCheck::IsTensorDescDimNumValid(outTensors.at(0).desc, outDimNum)) {
        ATB_LOG(ERROR) << "outTensor dim num is not support for FastSoftMaxOperation.";
        return atb::ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensors.at(0).desc.shape.dims[0] != outTensors.at(0).desc.shape.dims[0]) {
        ATB_LOG(ERROR) << "outTensor shape is not consistent with inTensor for FastSoftMaxOperation.";
        return atb::ERROR_INVALID_TENSOR_SIZE;
    }
    return atb::NO_ERROR;
}
} // namespace

namespace atb {
OPERATION_PARAM_FUNCS(FastSoftMaxOperation, train::FastSoftMaxParam)

FastSoftMaxOperation::FastSoftMaxOperation(const train::FastSoftMaxParam &param)
    : OperationBase("FastSoftMaxOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("FastSoftMaxOperation");
}

FastSoftMaxOperation::~FastSoftMaxOperation() {}

uint32_t FastSoftMaxOperation::GetInputNum() const
{
    const uint32_t inTensorNum = 1;
    return inTensorNum;
}

uint32_t FastSoftMaxOperation::GetOutputNum() const
{
    const uint32_t outTensorNum = 1;
    return outTensorNum;
}

Status FastSoftMaxOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                            SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    return NO_ERROR;
}

Status FastSoftMaxOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    return CheckInTensorDescs(inTensorDescs, param_);
}

Status FastSoftMaxOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    Status status = CheckInTensors(inTensors, param_);
    if (status != NO_ERROR) {
        return status;
    }
    return CheckOutTensors(inTensors, outTensors);
}

std::shared_ptr<Runner> FastSoftMaxOperation::CreateRunner(Context &context) const
{
    ContextBase *contextBase = dynamic_cast<ContextBase *>(&context);
    if (!contextBase) {
        ATB_LOG(DEBUG) << "context cast to contextBase failed!";
        return nullptr;
    }
    RunnerPool &pool = contextBase->GetRunnerPool(RUNNER_TYPE_FASTSOFTMAX);
    Runner *runner = pool.MallocRunner<FastSoftMaxOpsRunner, train::FastSoftMaxParam>(param_);
    if (!runner) {
        ATB_LOG(DEBUG) << "MallocRunner from pool failed!";
        return std::make_shared<FastSoftMaxOpsRunner>(param_);
    }
    return std::shared_ptr<Runner>(runner, [&pool](Runner *runner) { pool.FreeRunner(runner); });
}

nlohmann::json FastSoftMaxOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}

train::FastSoftMaxParam FastSoftMaxOperation::GetParam() const
{
    return param_;
}

void FastSoftMaxOperation::SetParam(const train::FastSoftMaxParam &param)
{
    param_ = param;
    runner_ = nullptr;
}
} // namespace atb