/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "genattentionmask_operation.h"
#include <climits>
#include "atb/utils/config.h"
#include "atb/utils/log.h"
#include "genattentionmask_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/singleton.h"

namespace {
constexpr size_t MAX_SEQLEN_SIZE = 32;
bool ParamCheck(const atb::train::GenAttentionMaskParam &opParam)
{
    if (!atb::GetSingleton<atb::Config>().Is910B()) {
        ATB_LOG(ERROR) << "GenAttentionMaskOperation is only supported in Atlas 800I A2 inference product.";
        return false;
    }

    if (opParam.headNum <= 0) {
        ATB_LOG(ERROR) << "headNum need bigger than 0";
        return false;
    }
    if (!atb::OperationUtil::QSeqLenCheck(opParam.seqLen)) {
        return false;
    }

    int64_t outdims = 0;
    for (std::size_t i = 0; i < opParam.seqLen.size(); i++) {
        if (opParam.headNum > LLONG_MAX / (opParam.seqLen.at(i) * static_cast<int64_t>(opParam.seqLen.at(i))) ||
            outdims > LLONG_MAX - opParam.headNum * opParam.seqLen.at(i) * static_cast<int64_t>(opParam.seqLen.at(i))) {
            ATB_LOG(ERROR) << "headNum or SeqLen is too large, which will result in overflow.";
            return false;
        }
        outdims += opParam.headNum * opParam.seqLen.at(i) * opParam.seqLen.at(i);
    }
    return true;
}
} // namespace

namespace atb {
static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_NUM = 1;
static const uint64_t OUT_TENSOR_DIM_NUM = 2;
static const uint64_t IN_DIM_SIZE = 4;
static const int64_t IN_DIM_DIM1 = 1;
static const int DIM3_VAL = 3;

OPERATION_PARAM_FUNCS(GenAttentionMaskOperation, train::GenAttentionMaskParam)

GenAttentionMaskOperation::GenAttentionMaskOperation(const train::GenAttentionMaskParam &param)
    : OperationBase("GenAttentionMaskOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("GenAttentionMaskOperation");
}

GenAttentionMaskOperation::~GenAttentionMaskOperation() {}

uint32_t GenAttentionMaskOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t GenAttentionMaskOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status GenAttentionMaskOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                                 SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    std::size_t size = param_.seqLen.size();
    Dims inShape = inTensorDescs.at(0).shape;
    uint64_t inDimNum = inShape.dimNum;
    auto inDimVec = inShape.dims;

    if (inDimNum != IN_DIM_SIZE) {
        ATB_LOG(ERROR) << "inTensor shape dimNum error, shape should be [batch,1,maxseqlen,maxseqlen]";
        return ERROR_INVALID_TENSOR_DIM;
    }
    int64_t maxseqlen = inDimVec[2];
    if (inDimVec[0] != static_cast<int64_t>(size) || inDimVec[1] != IN_DIM_DIM1 || maxseqlen != inDimVec[DIM3_VAL]) {
        ATB_LOG(ERROR) << "inTensor shape should be [batch,1,maxseqlen,maxseqlen]";
        return ERROR_INVALID_TENSOR_DIM;
    }

    int64_t outdims = 0;
    int headNum = param_.headNum;
    for (std::size_t i = 0; i < size; i++) {
        outdims += headNum * param_.seqLen.at(i) * param_.seqLen.at(i);
        if (param_.seqLen.at(i) > maxseqlen) {
            ATB_LOG(ERROR) << "seqLen shape should not be larger than maxseqlen";
            return ERROR_INVALID_PARAM;
        }
    }
    ATB_LOG(INFO) << outTensorDescs.at(0).shape.dims;
    outTensorDescs.at(0).shape.dimNum = 1;
    outTensorDescs.at(0).shape.dims[0] = outdims;
    ATB_LOG(INFO) << outTensorDescs.at(0).shape.dims[0];
    return NO_ERROR;
}

Status GenAttentionMaskOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                                 const SVector<Tensor> &outTensors) const
{
    int64_t outdims = 0;
    std::size_t size = param_.seqLen.size();
    int headNum = param_.headNum;
    for (std::size_t i = 0; i < size; i++) {
        outdims += headNum * param_.seqLen.at(i) * param_.seqLen.at(i);
    }
    if (outTensors.at(0).desc.shape.dims[0] != outdims) {
        ATB_LOG(ERROR) << "outTensor dim should be sum(headNum * seqLen.at(i) * seqLen.at(i))";
        return ERROR_INVALID_TENSOR_DIM;
    }
    ATB_LOG(DEBUG) << "inTensors size:" << inTensors.size();
    return NO_ERROR;
}

std::shared_ptr<Runner> GenAttentionMaskOperation::CreateRunner(Context &context) const
{
    ContextBase *contextBase = dynamic_cast<ContextBase *>(&context);
    if (!contextBase) {
        ATB_LOG(DEBUG) << "context cast to contextBase failed!";
        return nullptr;
    }
    RunnerPool &pool = contextBase->GetRunnerPool(RUNNER_TYPE_GENATTENTIONMASK);
    Runner *runner = pool.MallocRunner<GenAttentionMaskOpsRunner, train::GenAttentionMaskParam>(param_);
    if (!runner) {
        ATB_LOG(DEBUG) << "MallocRunner from pool failed!";
        return std::make_shared<GenAttentionMaskOpsRunner>(param_);
    }
    return std::shared_ptr<Runner>(runner, [&pool](Runner *runner) { pool.FreeRunner(runner); });
}

nlohmann::json GenAttentionMaskOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}

train::GenAttentionMaskParam GenAttentionMaskOperation::GetParam() const
{
    return param_;
}

void GenAttentionMaskOperation::SetParam(const train::GenAttentionMaskParam &param)
{
    param_ = param;
    runner_ = nullptr;
}
} // namespace atb