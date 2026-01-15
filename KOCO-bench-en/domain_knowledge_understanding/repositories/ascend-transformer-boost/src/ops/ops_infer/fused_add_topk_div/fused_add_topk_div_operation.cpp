/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fused_add_topk_div_operation.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils/tensor_check.h"
#include "fused_add_topk_div_ops_runner.h"

static constexpr uint32_t IN_TENSOR_NUM = 2;
static constexpr uint32_t IN_TENSOR_NUM_EXPERT_MAPPING = 4;
static constexpr uint32_t OUT_TENSOR_NUM = 2;
static constexpr uint32_t TENSOR_2 = 2;
static constexpr uint32_t TENSOR_3 = 3;

namespace {
bool ParamCheck(const atb::infer::FusedAddTopkDivParam opParam)
{
    if (opParam.groupNum == 0) {
        ATB_LOG(ERROR) << "groupNum should not be 0";
        return false;
    }
    if (opParam.groupTopk == 0) {
        ATB_LOG(ERROR) << "groupTopk should not be 0";
        return false;
    }
    if (opParam.groupTopk > opParam.groupNum) {
        ATB_LOG(ERROR) << "groupTopk should not be larger than groupNum";
        return false;
    }
    if (opParam.n == 0) {
        ATB_LOG(ERROR) << "n should not be 0";
        return false;
    }
    if (opParam.k == 0) {
        ATB_LOG(ERROR) << "k should not be 0";
        return false;
    }
    if (opParam.activationType != atb::infer::ACTIVATION_SIGMOID) {
        ATB_LOG(ERROR) << "activationType only support ACTIVATION_SIGMOID";
        return false;
    }
    return true;
}
} // namespace

namespace atb {
OPERATION_PARAM_FUNCS(FusedAddTopkDivOperation, infer::FusedAddTopkDivParam)

FusedAddTopkDivOperation::FusedAddTopkDivOperation(const infer::FusedAddTopkDivParam &param)
    : OperationBase("FusedAddTopkDivOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr(
        param_.enableExpertMapping ? "FusedAddTopkDivOperationExpertMapping" : "FusedAddTopkDivOperation");
}

FusedAddTopkDivOperation::~FusedAddTopkDivOperation() {}

uint32_t FusedAddTopkDivOperation::GetInputNum() const
{
    if (param_.enableExpertMapping) {
        return IN_TENSOR_NUM_EXPERT_MAPPING;
    }
    return IN_TENSOR_NUM;
}

uint32_t FusedAddTopkDivOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

infer::FusedAddTopkDivParam FusedAddTopkDivOperation::GetParam() const
{
    return param_;
}

void FusedAddTopkDivOperation::SetParam(const infer::FusedAddTopkDivParam &param)
{
    param_ = param;
    runner_ = nullptr;
}

Status FusedAddTopkDivOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                                SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    outTensorDescs.at(0).dtype = ACL_FLOAT;
    outTensorDescs.at(0).shape.dims[1] = param_.k;

    outTensorDescs.at(1) = inTensorDescs.at(0);
    outTensorDescs.at(1).dtype = ACL_INT32;
    outTensorDescs.at(1).shape.dims[1] = param_.k;

    return NO_ERROR;
}

Status FusedAddTopkDivOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    return InTensorDescsCheck(inTensorDescs);
}

Status FusedAddTopkDivOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                                const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs;
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    Status status = InTensorDescsCheck(inTensorDescs);
    if (status != NO_ERROR) {
        return status;
    }
    return OutTensorCheck(outTensors, inTensorDescs);
}

std::shared_ptr<Runner> FusedAddTopkDivOperation::CreateRunner(Context &context) const
{
    ContextBase *contextBase = dynamic_cast<ContextBase *>(&context);
    if (!contextBase) {
        ATB_LOG(DEBUG) << "context cast to contextBase failed!";
        return nullptr;
    }
    RunnerPool &pool = contextBase->GetRunnerPool(RUNNER_TYPE_FUSED_ADD_TOPK_DIV);
    Runner *runner = pool.MallocRunner<FusedAddTopkDivOpsRunner, infer::FusedAddTopkDivParam>(param_);
    if (!runner) {
        ATB_LOG(DEBUG) << "MallocRunner from pool failed!";
        return std::make_shared<FusedAddTopkDivOpsRunner>(param_);
    }
    return std::shared_ptr<Runner>(runner, [poolPtr = &pool](Runner *runner) { poolPtr->FreeRunner(runner); });
}

nlohmann::json FusedAddTopkDivOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}

Status FusedAddTopkDivOperation::InTensorDescsCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    if (!TensorCheck::IsTensorDescDimNumValid(inTensorDescs.at(0), 2)) { // 2: x dimNum
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    int64_t inTensor0Dim1 = inTensorDescs.at(0).shape.dims[1];
    if (inTensor0Dim1 % param_.groupNum != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor0 dim1 should be integer multiple of groupNum"
                       << ",inTensor0 dim1 = " << inTensor0Dim1 << ", groupNum = " << param_.groupNum;
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensor0Dim1 < param_.k) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor0 dim1 should not smaller than k"
                       << ", inTensor0 dim1 = " << inTensor0Dim1 << ", k = " << param_.k;
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensor0Dim1 < param_.groupNum * param_.n) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor0 dim1 should not smaller than groupNum * n"
                       << ", inTensor0 dim1 = " << inTensor0Dim1 << ", groupNum * n = " << param_.groupNum * param_.n;
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensor0Dim1 > param_.groupNum * 32) { // 32: maximum: groupNum * 32
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor0 dim1 should not larger than groupNum * 32"
                       << ", inTensor0 dim1 = " << inTensor0Dim1
                       << ", groupNum * 32 = " << (param_.groupNum * 32); // 32: maximum: groupNum * 32
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (param_.groupNum != 8 && inTensor0Dim1 > 32) { // 8: groupNum, 32: dim1 maximum
        ATB_LOG(ERROR) << GetLogPrefix() << "groupNum is 8, inTensor0 dim1 should not larger than 32"
                       << ", inTensor0 dim1 = " << inTensor0Dim1;
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (!TensorCheck::IsTensorDescDimNumValid(inTensorDescs.at(1), 1)) { // 1: addMum
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor1 dimNum and inTensor0 dimNum should be equal"
                       << ", inTensor1 dimNum = " << inTensorDescs.at(1).shape.dimNum
                       << ", inTensor0 dimNum = " << inTensorDescs.at(0).shape.dimNum;
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDescs.at(1).shape.dims[0] != inTensor0Dim1) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor1 dim0 and inTensor0 dim1 should be the same"
                       << ", inTensor1 dim0 = " << inTensorDescs.at(1).shape.dims[0]
                       << ", inTensor0 dim1 = " << inTensor0Dim1;
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (param_.enableExpertMapping) {
        return InTensorDescsCheckMapping(inTensorDescs);
    }
    return NO_ERROR;
}
Status FusedAddTopkDivOperation::InTensorDescsCheckMapping(const SVector<TensorDesc> &inTensorDescs) const
{
    if (!TensorCheck::IsTensorDescDimNumValid(inTensorDescs.at(TENSOR_2), 1)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor2 dimNum should be 1"
                       << ", inTensor2 dimNum = " << inTensorDescs.at(TENSOR_2).shape.dimNum;
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    int64_t inTensor0Dim1 = inTensorDescs.at(0).shape.dims[1];
    if (inTensorDescs.at(TENSOR_2).shape.dims[0] != inTensor0Dim1) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor2 dim0 and inTensor0 dim1 should be the same"
                       << ", inTensor2 dim0 = " << inTensorDescs.at(TENSOR_2).shape.dims[0]
                       << ", inTensor0 dim1 = " << inTensor0Dim1;
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (!TensorCheck::IsTensorDescDimNumValid(inTensorDescs.at(TENSOR_3), 2)) { // 2: mappingTable dimNum
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor3 dimNum should be 2"
                       << ", inTensor3 dimNum = " << inTensorDescs.at(TENSOR_3).shape.dimNum;
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    if (inTensorDescs.at(TENSOR_3).shape.dims[0] != inTensor0Dim1) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor3 dim0 and inTensor0 dim1 should be the same"
                       << ", inTensor3 dim0 = " << inTensorDescs.at(TENSOR_3).shape.dims[0]
                       << ", inTensor0 dim1 = " << inTensor0Dim1;
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (inTensorDescs.at(TENSOR_3).shape.dims[1] > 128) { // 128 : MAX support 128
        ATB_LOG(ERROR) << GetLogPrefix()
                       << "inTensor3 dim1 and inTensor0 dim1 should be less than or equal to MAX supprt 128"
                       << ", inTensor3 dim1 = " << inTensorDescs.at(TENSOR_3).shape.dims[1];
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status FusedAddTopkDivOperation::OutTensorCheck(const SVector<Tensor> &outTensors,
                                                const SVector<TensorDesc> &inTensorDescs) const
{
    const TensorDesc &yTensorDesc = outTensors.at(0).desc;
    const TensorDesc &indicesTensorDesc = outTensors.at(1).desc;
    if (!TensorCheck::IsTensorDescDimNumValid(yTensorDesc, 2)) { // 2: x, addNum
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }
    int64_t a = inTensorDescs.at(0).shape.dims[0];
    if (yTensorDesc.shape.dims[0] != a) {
        ATB_LOG(ERROR) << "outTensor0 dim0 and inTensor0 dim0 should be equal";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (yTensorDesc.shape.dims[1] != param_.k) {
        ATB_LOG(ERROR) << "outTensor0 dim1 should be k";
        return ERROR_INVALID_TENSOR_DIM;
    }

    Status status = TensorCheck::TensorDescsEqual(yTensorDesc, indicesTensorDesc);
    if (status == ERROR_INVALID_TENSOR_DIM_NUM) {
        ATB_LOG(ERROR) << "outTensor0 dimNum and outTensor1 dimNum should be equal";
        return ERROR_INVALID_TENSOR_DIM_NUM;
    } else if (status == ERROR_INVALID_TENSOR_DIM) {
        ATB_LOG(ERROR) << "outTensor0 dims and outTensor1 dims should be equal";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}
} // namespace atb