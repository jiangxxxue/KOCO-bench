/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "index_add_operation.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/config.h"
#include "atb/utils/log.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "index_add_ops_runner.h"
#include "atb/operation/op_param_funcs.h"

static const int32_t IN_TENSOR_NUM = 4;
static const int32_t OUT_TENSOR_NUM = 1;
static const uint64_t DIM_NUM_2 = 2;
static const size_t DIM_2 = 2;
static const int64_t D_2_MAX = 8192;

namespace atb {
template <> Status CreateOperation(const infer::IndexAddParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    ATB_LOG(INFO) << "CreateOperation IndexAddParam indexType: " << opParam.indexType << ", axis: " << opParam.axis;
    if (opParam.indexType <= infer::IndexAddParam::INDEX_UNDEFINED ||
        opParam.indexType > infer::IndexAddParam::INDEX_ADD_VALID) {
        ATB_LOG(ERROR) << "param indexType should be INDEX_ADD/INDEX_ADD_VALID";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.indexType == infer::IndexAddParam::INDEX_ADD_VALID) {
        if (!GetSingleton<Config>().Is910B()) {
            ATB_LOG(ERROR) << "INDEX_ADD_VALID only support Atlas 800I A2 inference product";
            return ERROR_INVALID_PARAM;
        }
        if (opParam.axis != 0) {
            ATB_LOG(ERROR) << "param axis should be 0";
            return ERROR_INVALID_PARAM;
        }
    }

    *operation = new (std::nothrow) IndexAddOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

IndexAddOperation::IndexAddOperation(const infer::IndexAddParam &param)
    : OperationBase("IndexAddOperation"), param_(param)
{
    if (param_.indexType == infer::IndexAddParam::INDEX_ADD) {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("IndexAddOperationIndexAdd");
    } else if (param_.indexType == infer::IndexAddParam::INDEX_ADD_VALID) {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("IndexAddOperationIndexAddValid");
    }
}

IndexAddOperation::~IndexAddOperation() {}

uint32_t IndexAddOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t IndexAddOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status IndexAddOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                         SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    return NO_ERROR;
}

Status IndexAddOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    return InTensorDescsCheck(inTensorDescs);
}

Status IndexAddOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
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
    } else if (status == ERROR_INVALID_TENSOR_DIM) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor0 dims and inTensor0 dims should be the same";
    }
    return status;
}

nlohmann::json IndexAddOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}

std::shared_ptr<Runner> IndexAddOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<IndexAddOpsRunner>(param_);
}

Status IndexAddOperation::InTensorDescsCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    if (param_.indexType == infer::IndexAddParam::INDEX_ADD) {
        return IndexAddInTensorDescsCheck(inTensorDescs);
    } else if (param_.indexType == infer::IndexAddParam::INDEX_ADD_VALID) {
        return IndexAddValidInTensorDescsCheck(inTensorDescs);
    }
    return NO_ERROR;
}

Status IndexAddOperation::IndexAddInTensorDescsCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    size_t inTensorDescId = 0;
    TensorDesc varTensorDesc = inTensorDescs.at(inTensorDescId++);
    TensorDesc indicesTensorDesc = inTensorDescs.at(inTensorDescId++);
    TensorDesc updatesTensorDesc = inTensorDescs.at(inTensorDescId++);
    TensorDesc alphaTensorDesc = inTensorDescs.at(inTensorDescId++);

    auto minAxis = -static_cast<int64_t>(varTensorDesc.shape.dimNum);
    auto maxAxis = static_cast<int64_t>(varTensorDesc.shape.dimNum) - 1;
    if (param_.axis < minAxis || param_.axis > maxAxis) {
        ATB_LOG(ERROR) << GetLogPrefix() << "param axis should be in range [" << minAxis << ", " << maxAxis
                       << "] but get " << param_.axis;
        return ERROR_INVALID_PARAM;
    }

    if (!TensorCheck::IsTensorDescDimNumValid(indicesTensorDesc, 1)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor1 dimNum should be 1 but get " << indicesTensorDesc.shape.dimNum;
        return ERROR_INVALID_TENSOR_SIZE;
    }

    if (!TensorCheck::IsTensorDescDimNumValid(updatesTensorDesc, varTensorDesc.shape.dimNum)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor2 dimNum [" << updatesTensorDesc.shape.dimNum
                       << "] and inTensor0 dimNum [" << varTensorDesc.shape.dimNum << "] should be equal";
        return ERROR_INVALID_TENSOR_SIZE;
    }
    size_t axis = param_.axis < 0 ? param_.axis + varTensorDesc.shape.dimNum : param_.axis;
    for (size_t i = 0; i < updatesTensorDesc.shape.dimNum; i++) {
        if (i == axis) {
            if (updatesTensorDesc.shape.dims[i] != indicesTensorDesc.shape.dims[0]) {
                ATB_LOG(ERROR) << GetLogPrefix() << "if i == axis, inTensor2 dim_i [" << updatesTensorDesc.shape.dims[i]
                               << "] and inTensor1 dim0 [" << indicesTensorDesc.shape.dims[0] << "] should be equal";
                return ERROR_INVALID_TENSOR_DIM;
            }
        } else {
            if (updatesTensorDesc.shape.dims[i] != varTensorDesc.shape.dims[i]) {
                ATB_LOG(ERROR) << GetLogPrefix() << "inTensor2 dim_i [" << updatesTensorDesc.shape.dims[i]
                               << "] and inTensor0 dim_i [" << varTensorDesc.shape.dims[i] << "] should be equal";
                return ERROR_INVALID_TENSOR_DIM;
            }
        }
    }

    if (!TensorCheck::IsTensorDescDimNumValid(alphaTensorDesc, 1)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor3 dimNum should be 1 but get " << alphaTensorDesc.shape.dimNum;
        return ERROR_INVALID_TENSOR_SIZE;
    }
    if (alphaTensorDesc.shape.dims[0] != 1) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor3 dim0 should be 1 but get " << alphaTensorDesc.shape.dims[0];
        return ERROR_INVALID_TENSOR_DIM;
    }

    return NO_ERROR;
}

Status IndexAddOperation::IndexAddValidInTensorDescsCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    size_t inTensorDescId = 0;
    TensorDesc varTensorDesc = inTensorDescs.at(inTensorDescId++);
    TensorDesc indicesTensorDesc = inTensorDescs.at(inTensorDescId++);
    TensorDesc updatesTensorDesc = inTensorDescs.at(inTensorDescId++);
    TensorDesc validIndicesTensorDesc = inTensorDescs.at(inTensorDescId++);
    bool isdtypeBF16 = (varTensorDesc.dtype == ACL_BF16);
    if (!TensorCheck::IsTensorDescDimNumValid(varTensorDesc, DIM_NUM_2)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor0 dimNum should be 2 but get " << varTensorDesc.shape.dimNum;
        return isdtypeBF16 ? ERROR_INVALID_TENSOR_DIM : ERROR_INVALID_TENSOR_SIZE;
    }
    if (varTensorDesc.shape.dims[1] > D_2_MAX) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor0 dim1 should be in range (0, 8192] but get "
                       << varTensorDesc.shape.dims[1];
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (!TensorCheck::IsTensorDescDimNumValid(indicesTensorDesc, 1)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor1 dimNum should be 1 but get " << indicesTensorDesc.shape.dimNum;
        return isdtypeBF16 ? ERROR_INVALID_TENSOR_DIM : ERROR_INVALID_TENSOR_SIZE;
    }

    if (!TensorCheck::IsTensorDescDimNumValid(updatesTensorDesc, DIM_NUM_2)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor2 dimNum should be 2 but get " << updatesTensorDesc.shape.dimNum;
        return isdtypeBF16 ? ERROR_INVALID_TENSOR_DIM : ERROR_INVALID_TENSOR_SIZE;
    }
    if (updatesTensorDesc.shape.dims[0] != indicesTensorDesc.shape.dims[0]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor2 dim0 [" << updatesTensorDesc.shape.dims[0]
                       << "] and inTensor1 dim0 [" << indicesTensorDesc.shape.dims[0] << "] should be equal";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (updatesTensorDesc.shape.dims[1] != varTensorDesc.shape.dims[1]) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor2 dim1 [" << updatesTensorDesc.shape.dims[1]
                       << "] and inTensor0 dim1 [" << varTensorDesc.shape.dims[1] << "] should be equal";
        return ERROR_INVALID_TENSOR_DIM;
    }

    if (!TensorCheck::IsTensorDescDimNumValid(validIndicesTensorDesc, 1)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor3 dimNum should be 1 but get "
                       << validIndicesTensorDesc.shape.dimNum;
        return isdtypeBF16 ? ERROR_INVALID_TENSOR_DIM : ERROR_INVALID_TENSOR_SIZE;
    }
    if (validIndicesTensorDesc.shape.dims[0] != 1) {
        ATB_LOG(ERROR) << GetLogPrefix() << "inTensor3 dim0 should be 1 but get "
                       << validIndicesTensorDesc.shape.dims[0];
        return ERROR_INVALID_TENSOR_DIM;
    }

    return NO_ERROR;
}
} // namespace atb