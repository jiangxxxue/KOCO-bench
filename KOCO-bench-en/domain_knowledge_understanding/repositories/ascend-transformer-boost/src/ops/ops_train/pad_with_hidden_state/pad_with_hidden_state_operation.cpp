/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "pad_with_hidden_state_operation.h"
#include "pad_with_hidden_state_ops_runner.h"
#include "atb/utils/config.h"
#include "atb/utils/tensor_check.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/singleton.h"

namespace {
constexpr int32_t DIM_2 = 2;
constexpr int32_t MAX_SEQLEN = 4096;
bool ParamCheck(const atb::train::PadWithHiddenStateParam &opParam)
{
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        ATB_LOG(ERROR) << "PadWithHiddenStateOperation is only supported in Atlas 800I A2 inference product.";
        return false;
    }

    if (opParam.maxSeqLen <= 0 || opParam.maxSeqLen > MAX_SEQLEN) {
        ATB_LOG(ERROR) << "PadWithHiddenStateParam maxSeqLen should be in (0, " << MAX_SEQLEN << "], but get "
                       << opParam.maxSeqLen;
        return false;
    }
    return atb::OperationUtil::QSeqLenCheck(opParam.qSeqLen, opParam.maxSeqLen);
}

atb::Status CheckInTensorDescs(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                               const atb::train::PadWithHiddenStateParam &param)
{
    const uint64_t inDimNum = 2;
    if (!atb::TensorCheck::IsTensorDescDimNumValid(inTensorDescs.at(0), inDimNum)) {
        ATB_LOG(ERROR) << "PadWithHiddenStateOperation inTensor dim number is not support, need 2, but get "
                       << inTensorDescs.at(0).shape.dimNum;
        return atb::ERROR_INVALID_TENSOR_DIM;
    }
    int64_t outerSize = 0;
    for (auto sampleSeqLen : param.qSeqLen) {
        outerSize += sampleSeqLen;
    }
    if (inTensorDescs.at(0).shape.dims[0] != outerSize) {
        ATB_LOG(ERROR) << "PaddingWithHiddenStateOperation inTensor shape is not consistent with param.";
        return atb::ERROR_INVALID_TENSOR_SIZE;
    }
    return atb::NO_ERROR;
}

atb::Status CheckInTensors(const atb::SVector<atb::Tensor> &inTensors, const atb::train::PadWithHiddenStateParam &param)
{
    const atb::SVector<atb::TensorDesc> inTensorDescs{inTensors.at(0).desc};
    return CheckInTensorDescs(inTensorDescs, param);
}

atb::Status CheckOutTensors(const atb::SVector<atb::Tensor> &inTensors, const atb::SVector<atb::Tensor> &outTensors,
                            const atb::train::PadWithHiddenStateParam &param)
{
    const uint64_t outDimNum = 3;
    if (!atb::TensorCheck::IsTensorDescDimNumValid(outTensors.at(0).desc, outDimNum)) {
        ATB_LOG(ERROR) << "PadWithHiddenStateOperation outTensor dimNum should be " << outDimNum << ", but get "
                       << outTensors.at(0).desc.shape.dimNum;
        return atb::ERROR_INVALID_TENSOR_DIM;
    }
    int64_t hiddenSize = inTensors.at(0).desc.shape.dims[1];
    if (static_cast<size_t>(outTensors.at(0).desc.shape.dims[0]) != param.qSeqLen.size() ||
        outTensors.at(0).desc.shape.dims[1] != param.maxSeqLen ||
        outTensors.at(0).desc.shape.dims[DIM_2] != hiddenSize) {
        ATB_LOG(ERROR) << "PadWithHiddenStateOperation outTensor shape is not consistent with inTensor or param.";
        return atb::ERROR_INVALID_TENSOR_SIZE;
    }
    return atb::NO_ERROR;
}
} // namespace

namespace atb {
OPERATION_PARAM_FUNCS(PadWithHiddenStateOperation, train::PadWithHiddenStateParam)

PadWithHiddenStateOperation::PadWithHiddenStateOperation(const train::PadWithHiddenStateParam &param)
    : OperationBase("PadWithHiddenStateOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("PadWithHiddenStateOperation");
}

PadWithHiddenStateOperation::~PadWithHiddenStateOperation() {}

uint32_t PadWithHiddenStateOperation::GetInputNum() const
{
    const uint32_t inTensorNum = 1;
    return inTensorNum;
}

uint32_t PadWithHiddenStateOperation::GetOutputNum() const
{
    const uint32_t outTensorNum = 1;
    return outTensorNum;
}

Status PadWithHiddenStateOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                                   SVector<TensorDesc> &outTensorDescs) const
{
    const uint64_t outDimNum = 3;
    int64_t batchSize = static_cast<int64_t>(param_.qSeqLen.size());
    int64_t maxSeqLen = param_.maxSeqLen;
    int64_t hiddenSize = inTensorDescs.at(0).shape.dims[1];
    auto &outDesc = outTensorDescs.at(0);
    outDesc.dtype = inTensorDescs.at(0).dtype;
    outDesc.format = inTensorDescs.at(0).format;
    outDesc.shape.dimNum = outDimNum;
    outDesc.shape.dims[0] = batchSize;
    outDesc.shape.dims[1] = maxSeqLen;
    outDesc.shape.dims[DIM_2] = hiddenSize;
    return NO_ERROR;
}

Status PadWithHiddenStateOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    return CheckInTensorDescs(inTensorDescs, param_);
}

Status PadWithHiddenStateOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                                   const SVector<Tensor> &outTensors) const
{
    Status status = CheckInTensors(inTensors, param_);
    if (status != NO_ERROR) {
        return status;
    }
    return CheckOutTensors(inTensors, outTensors, param_);
}

std::shared_ptr<Runner> PadWithHiddenStateOperation::CreateRunner(Context &context) const
{
    ContextBase *contextBase = dynamic_cast<ContextBase *>(&context);
    if (!contextBase) {
        ATB_LOG(DEBUG) << "context cast to contextBase failed!";
        return nullptr;
    }
    RunnerPool &pool = contextBase->GetRunnerPool(RUNNER_TYPE_PAD_WITH_HIDDEN_STATE);
    Runner *runner = pool.MallocRunner<PadWithHiddenStateOpsRunner, train::PadWithHiddenStateParam>(param_);
    if (!runner) {
        ATB_LOG(DEBUG) << "MallocRunner from pool failed!";
        return std::make_shared<PadWithHiddenStateOpsRunner>(param_);
    }
    return std::shared_ptr<Runner>(runner, [&pool](Runner *runner) { pool.FreeRunner(runner); });
}

train::PadWithHiddenStateParam PadWithHiddenStateOperation::GetParam() const
{
    return param_;
}

void PadWithHiddenStateOperation::SetParam(const train::PadWithHiddenStateParam &param)
{
    param_ = param;
    runner_ = nullptr;
}
} // namespace atb