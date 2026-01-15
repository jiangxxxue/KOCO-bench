/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rope_grad_operation.h"
#include "atb/utils/config.h"
#include "atb/utils/log.h"
#include "rope_grad_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/singleton.h"

namespace {
bool ParamCheck(const atb::train::RopeGradParam &opParam)
{
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        ATB_LOG(ERROR) << "RopeGradOperation is not supported in Atlas 800I A2 inference product.";
        return false;
    }

    return atb::OperationUtil::QSeqLenCheck(opParam.qSeqLen);
}
} // namespace

namespace atb {
static const uint32_t IN_TENSOR_NUM = 4;
static const uint32_t OUT_TENSOR_NUM = 2;
static const uint32_t INPUT_SHAPE_DIM = 2;
static const uint32_t SECOND_IN_TENSOR = 2;
static const uint32_t THIRD_IN_TENSOR = 3;
static const uint32_t HEAD_SIZE = 128;

OPERATION_PARAM_FUNCS(RopeGradOperation, train::RopeGradParam)

RopeGradOperation::RopeGradOperation(const train::RopeGradParam &param)
    : OperationBase("RopeGradOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("RopeGradOperation");
}

RopeGradOperation::~RopeGradOperation() {}

uint32_t RopeGradOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t RopeGradOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status RopeGradOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                         SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    outTensorDescs.at(1) = inTensorDescs.at(1);
    return NO_ERROR;
}

Status RopeGradOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    Status st = DimCheck(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }

    return ParamCheck(inTensorDescs);
}

Status RopeGradOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs;
    for (size_t i = 0; i < inTensors.size(); i++) {
        inTensorDescs.push_back(inTensors.at(i).desc);
    }
    ATB_LOG(DEBUG) << "outTensors size:" << outTensors.size();
    Status st = DimCheck(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    return ParamCheck(inTensorDescs);
}

Status RopeGradOperation::ParamCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    if (static_cast<int32_t>(param_.qSeqLen.size()) <= 0) {
        ATB_LOG(ERROR) << "Wrong Param, qSeqLen.size(): " << param_.qSeqLen.size()
                       << " size of qSeqLen should be more than zero";
        return ERROR_INVALID_PARAM;
    }

    int64_t headSize = inTensorDescs.at(SECOND_IN_TENSOR).shape.dims[1];
    if (headSize != HEAD_SIZE) {
        ATB_LOG(ERROR) << "Wrong Param, headSize only support 128, the headSize is: " << headSize;
        return ERROR_INVALID_TENSOR_DIM;
    }

    std::size_t size = param_.qSeqLen.size();
    for (std::size_t i = 0; i < size; i++) {
        if (param_.qSeqLen.at(i) <= 0 || param_.qSeqLen.at(i) > inTensorDescs.at(SECOND_IN_TENSOR).shape.dims[0]) {
            ATB_LOG(ERROR) << "Wrong Param, qSeqLen" << i << ": " << param_.qSeqLen.at(i);
            return ERROR_INVALID_PARAM;
        }
    }

    return NO_ERROR;
}

Status RopeGradOperation::DimCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    for (std::size_t i = 0; i < inTensorDescs.size(); i++) {
        if (inTensorDescs.at(i).shape.dimNum != INPUT_SHAPE_DIM) {
            ATB_LOG(ERROR) << "inTensor shape dimNum is not support, tensor" << i << " only support 2 dimNum"
                           << inTensorDescs.at(i).shape.dimNum;
            return ERROR_INVALID_TENSOR_DIM;
        }
    }

    if (inTensorDescs.at(0).shape.dims[0] != inTensorDescs.at(1).shape.dims[0] ||
        inTensorDescs.at(0).shape.dims[1] != inTensorDescs.at(1).shape.dims[1]) {
        ATB_LOG(ERROR) << "Q_grad1 Q_grad2 shape should be same. "
                       << "Q_grad1 : " << TensorUtil::TensorDescToString(inTensorDescs.at(0))
                       << ", Q_grad2: " << TensorUtil::TensorDescToString(inTensorDescs.at(1));
        return ERROR_INVALID_TENSOR_SIZE;
    }

    if (inTensorDescs.at(SECOND_IN_TENSOR).shape.dims[0] != inTensorDescs.at(THIRD_IN_TENSOR).shape.dims[0] ||
        inTensorDescs.at(SECOND_IN_TENSOR).shape.dims[1] != inTensorDescs.at(THIRD_IN_TENSOR).shape.dims[1]) {
        ATB_LOG(ERROR) << "cos sin shape should be same. "
                       << "cos : " << TensorUtil::TensorDescToString(inTensorDescs.at(SECOND_IN_TENSOR))  // index: 2
                       << ", sin: " << TensorUtil::TensorDescToString(inTensorDescs.at(THIRD_IN_TENSOR)); // index: 3
        return ERROR_INVALID_TENSOR_SIZE;
    }

    return NO_ERROR;
}

std::shared_ptr<Runner> RopeGradOperation::CreateRunner(Context &context) const
{
    ContextBase *contextBase = dynamic_cast<ContextBase *>(&context);
    if (!contextBase) {
        ATB_LOG(DEBUG) << "context cast to contextBase failed!";
        return nullptr;
    }
    RunnerPool &pool = contextBase->GetRunnerPool(RUNNER_TYPE_ROPE_GRAD);
    Runner *runner = pool.MallocRunner<RopeGradOpsRunner, train::RopeGradParam>(param_);
    if (!runner) {
        ATB_LOG(DEBUG) << "MallocRunner from pool failed!";
        return std::make_shared<RopeGradOpsRunner>(param_);
    }
    return std::shared_ptr<Runner>(runner, [&pool](Runner *runner) { pool.FreeRunner(runner); });
}

nlohmann::json RopeGradOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}

train::RopeGradParam RopeGradOperation::GetParam() const
{
    return param_;
}

void RopeGradOperation::SetParam(const train::RopeGradParam &param)
{
    param_ = param;
    runner_ = nullptr;
}
} // namespace atb