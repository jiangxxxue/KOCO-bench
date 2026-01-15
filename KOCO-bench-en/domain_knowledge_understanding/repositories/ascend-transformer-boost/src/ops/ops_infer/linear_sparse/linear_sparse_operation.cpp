/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "linear_sparse_operation.h"
#include "atb/utils/config.h"
#include "atb/utils/tensor_check.h"
#include "linear_sparse_ops_runner.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

static constexpr size_t DIM_2 = 2;
static constexpr size_t DIM_3 = 3;

static const uint32_t IN_TENSOR_NUM_WITH_BIAS = 5;
static const uint32_t OUT_TENSOR_NUM = 1;

namespace atb {
template <> Status CreateOperation(const infer::LinearSparseParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (!GetSingleton<Config>().Is310P()) {
        ATB_LOG(ERROR) << "linear_sparse only support Atlas inference products";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.transposeA) {
        ATB_LOG(ERROR) << "transposeA [" << opParam.transposeA << "] should be false";
        return ERROR_INVALID_PARAM;
    }
    if (!opParam.transposeB) {
        ATB_LOG(ERROR) << "transposeB [" << opParam.transposeB << "] should be true";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.tilingK != 8) { // 8: fixed value
        ATB_LOG(ERROR) << "tilingK [" << opParam.tilingK << "] should be 8";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.tilingN != 8) { // 8: fixed value
        ATB_LOG(ERROR) << "tilingN [" << opParam.tilingN << "] should be 8";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) LinearSparseOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

LinearSparseOperation::LinearSparseOperation(const infer::LinearSparseParam &param)
    : OperationBase("LinearSparseOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("LinearSparseOperation");
    commonCheckParam_ = param_;
}

LinearSparseOperation::~LinearSparseOperation() {}

uint32_t LinearSparseOperation::GetInputNum() const
{
    return IN_TENSOR_NUM_WITH_BIAS;
}

uint32_t LinearSparseOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status LinearSparseOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                             SVector<TensorDesc> &outTensorDescs) const
{
    return OperationUtil::MatmulInferShape(inTensorDescs, outTensorDescs, commonCheckParam_);
}

Status LinearSparseOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    return OperationUtil::MatmulInTensorDescsCheck(inTensorDescs, GetLogPrefix(), commonCheckParam_) ?
               NO_ERROR :
               ERROR_INVALID_TENSOR_DIM;
}

Status LinearSparseOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs = {};
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    if (!OperationUtil::MatmulInTensorDescsCheck(inTensorDescs, GetLogPrefix(), commonCheckParam_)) {
        return ERROR_INVALID_TENSOR_DIM;
    }
    const TensorDesc &outTensorDesc = outTensors.at(0).desc;
    return OperationUtil::MatmulOutTensorCheck(outTensorDesc, inTensorDescs, GetLogPrefix(), commonCheckParam_) ?
               NO_ERROR :
               ERROR_INVALID_TENSOR_DIM;
}

std::shared_ptr<Runner> LinearSparseOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<LinearSparseOpsRunner>(param_);
}

nlohmann::json LinearSparseOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb