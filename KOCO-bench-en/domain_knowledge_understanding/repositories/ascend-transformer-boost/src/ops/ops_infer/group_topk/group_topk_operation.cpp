/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "group_topk_operation.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/config.h"
#include "atb/utils/log.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "group_topk_ops_runner.h"
#include "atb/operation/op_param_funcs.h"

static constexpr uint32_t IN_TENSOR_NUM = 2;
static constexpr uint32_t OUT_TENSOR_NUM = 1;
static constexpr uint32_t DIM_NUM_2 = 2;
static constexpr uint32_t IN_TENSOR_SHAPE_0 = 1024;
static constexpr uint32_t MAX_EXPERT = 1024;

namespace atb {
template <> Status CreateOperation(const infer::GroupTopkParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    ATB_LOG(INFO) << "CreateOperation GroupTopkParam groupNum: " << opParam.groupNum << ", k: " << opParam.k
                  << ", groupMultiFlag:" << opParam.groupMultiFlag << ", n:" << opParam.n;
    if (opParam.groupNum < 1) {
        ATB_LOG(ERROR) << "param groupNum should >= 1";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.k < 1) {
        ATB_LOG(ERROR) << "param k should >= 1";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.k > opParam.groupNum) {
        ATB_LOG(ERROR) << "param groupNum and param k should satisfy the relationship: k <= groupNum";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.groupMultiFlag > infer::GroupTopkParam::GroupMultiFlag::SUM_MULTI_MAX) {
        ATB_LOG(ERROR) << "param groupMultiFlag is invalid";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.groupMultiFlag == infer::GroupTopkParam::GroupMultiFlag::SUM_MULTI_MAX && opParam.n == 0) {
        ATB_LOG(ERROR) << "param n should >= 1 when groupMultiFlag is SUM_MULTI_MAX";
        return ERROR_INVALID_PARAM;
    }
    if (!GetSingleton<Config>().Is910B()) {
        ATB_LOG(ERROR) << "groupTopk op only support Atlas 800I A2 inference product";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) GroupTopkOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

GroupTopkOperation::GroupTopkOperation(const infer::GroupTopkParam &param)
    : OperationBase("GroupTopkOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("GroupTopkOperation");
}

GroupTopkOperation::~GroupTopkOperation() {}

uint32_t GroupTopkOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t GroupTopkOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status GroupTopkOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                          SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    return NO_ERROR;
}

Status GroupTopkOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    return InTensorDescsCheck(inTensorDescs);
}

Status GroupTopkOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs;
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    Status status = InTensorDescsCheck(inTensorDescs);
    if (status != NO_ERROR) {
        return status;
    }

    TensorDesc varTensorDesc = inTensors.at(0).desc;
    TensorDesc outTensorDesc = outTensors.at(0).desc;
    status = TensorCheck::TensorDescsEqual(varTensorDesc, outTensorDesc);
    if (status == ERROR_INVALID_TENSOR_DIM_NUM) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor0 dimNum [" << outTensorDesc.shape.dimNum
                       << "] and inTensor0 dimNum [" << varTensorDesc.shape.dimNum << "] should be equal";
        return status;
    }
    if (status == ERROR_INVALID_TENSOR_DIM) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor0 dims and inTensor0 dims should be the same";
        return status;
    }
    return OutTensorsCheck(outTensors);
}

nlohmann::json GroupTopkOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}

std::shared_ptr<Runner> GroupTopkOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<GroupTopkOpsRunner>(param_);
}

Status GroupTopkOperation::InTensorDescsCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    size_t inTensorDescId = 0;
    TensorDesc tokenTensorDesc = inTensorDescs.at(inTensorDescId++);
    TensorDesc idxArrTensorDesc = inTensorDescs.at(inTensorDescId++);

    if (!TensorCheck::IsTensorDescDimNumValid(tokenTensorDesc, DIM_NUM_2)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor0 dimNum should be " << DIM_NUM_2 << " but get "
                       << tokenTensorDesc.shape.dimNum;
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }

    if (!TensorCheck::IsTensorDescDimNumValid(idxArrTensorDesc, 1)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor1 dimNum should be 1 but get " << idxArrTensorDesc.shape.dimNum;
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }

    int64_t expertNum = tokenTensorDesc.shape.dims[1];

    // param_.groupNum取值范围判断 [1, expertNum]
    if (param_.groupNum < 1 || param_.groupNum > expertNum) {
        ATB_LOG(ERROR) << GetLogPrefix() << "groupNum should in scope [1, " << expertNum << "], but get "
                       << param_.groupNum;
        return ERROR_INVALID_PARAM;
    }

    // param_.k取值范围判断 [1, groupNum]
    if (param_.k < 1 || param_.k > param_.groupNum) {
        ATB_LOG(ERROR) << GetLogPrefix() << "k should in scope [1, " << param_.groupNum << "], but get " << param_.k;
        return ERROR_INVALID_PARAM;
    }

    // param_.n取值范围判断 [1, expertNum / groupNum], 1在创建时已校验
    if (param_.groupMultiFlag == infer::GroupTopkParam::GroupMultiFlag::SUM_MULTI_MAX &&
        param_.n > expertNum / param_.groupNum) {
        ATB_LOG(ERROR) << GetLogPrefix() << "n should <= " << expertNum / param_.groupNum << ", but get " << param_.n;
        return ERROR_INVALID_PARAM;
    }

    // expertNum能被groupNum整除判断
    if (expertNum % param_.groupNum != 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "expert num " << expertNum
                       << " should be divided with no remainder by groupNum " << param_.groupNum;
        return ERROR_INVALID_PARAM;
    }

    // inTensor1.shape.dims[0] == 1024 判断
    if (idxArrTensorDesc.shape.dims[0] != IN_TENSOR_SHAPE_0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor1 dim[0] should be " << IN_TENSOR_SHAPE_0 << ", but get "
                       << idxArrTensorDesc.shape.dims[0];
        return ERROR_INVALID_TENSOR_DIM;
    }

    // inTensor0.shape.dims[1]取值范围为 [1,1024]
    if (tokenTensorDesc.shape.dims[1] < 1 || tokenTensorDesc.shape.dims[1] > MAX_EXPERT) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor0 dim[1] should in scope [1, " << MAX_EXPERT << "], but get "
                       << tokenTensorDesc.shape.dims[1];
        return ERROR_INVALID_TENSOR_DIM;
    }

    return NO_ERROR;
}

Status GroupTopkOperation::OutTensorsCheck(const SVector<Tensor> &outTensors) const
{
    size_t outTensorId = 0;
    TensorDesc out0TensorDesc = outTensors.at(outTensorId++).desc;
    if (!TensorCheck::IsTensorDescDimNumValid(out0TensorDesc, DIM_NUM_2)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor0 dimNum should be " << DIM_NUM_2 << " but get "
                       << out0TensorDesc.shape.dimNum;
        return ERROR_INVALID_TENSOR_DIM_NUM;
    }

    return NO_ERROR;
}

} // namespace atb