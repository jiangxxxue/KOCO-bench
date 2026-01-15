/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cohere_layernorm_operation.h"
#include <cmath>
#include "cohere_layernorm_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"
#include "atb/utils/config.h"
#include "atb/utils/operation_util.h"
#include "atb/utils/singleton.h"
#include "atb/utils/tensor_util.h"

namespace atb {
static const uint32_t IN_TENSOR_NUM = 2;
static const uint32_t OUT_TENSOR_NUM = 1;
static const uint32_t X_TENSOR_NUM = 0;
static const uint32_t GAMMA_TENSOR_NUM = 1;
static const uint32_t LAST_ONE_NUM = 1;
static const uint32_t LAST_TWO_NUM = 2;
static const uint32_t ALIGNMENT = 16;
static const uint32_t DIM_NUM_TWO = 2;
static const uint32_t DIM_NUM_THREE = 3;
static const uint32_t DIM_NUM_FOUR = 4;
static const float THRESHOLD = 2e-38; // The non-negative minimum value of float type is 1.17549e-038.

template <> Status CreateOperation(const infer::CohereLayerNormParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (!GetSingleton<Config>().Is910B()) {
        ATB_LOG(ERROR) << "only support Atlas 800I A2/A3 Processors";
        return ERROR_INVALID_PARAM;
    }
    if (opParam.epsilon < THRESHOLD) {
        ATB_LOG(ERROR) << "Invalid epsilon, it's recommended to init a positive value for eps.";
        return ERROR_INVALID_PARAM;
    }
    *operation = new (std::nothrow) CohereLayerNormOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

CohereLayerNormOperation::CohereLayerNormOperation(const infer::CohereLayerNormParam &param)
    : OperationBase("CohereLayerNormOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("CohereLayerNormOperation");
}

CohereLayerNormOperation::~CohereLayerNormOperation() {}

uint32_t CohereLayerNormOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t CohereLayerNormOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status CohereLayerNormOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                                SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    return NO_ERROR;
}

Status CohereLayerNormOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    TensorDesc xDesc = inTensorDescs.at(X_TENSOR_NUM);
    TensorDesc gammaDesc = inTensorDescs.at(GAMMA_TENSOR_NUM);
    uint64_t xTensorDimNum = xDesc.shape.dimNum;
    uint64_t gammaTensorDimNum = gammaDesc.shape.dimNum;
    if (xTensorDimNum != DIM_NUM_THREE && xTensorDimNum != DIM_NUM_FOUR) {
        ATB_LOG(ERROR) << "The dim number of intensor0 is not equal to 3 or 4";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (gammaTensorDimNum != DIM_NUM_TWO) {
        ATB_LOG(ERROR) << "The dim number of intensor1 is not 2";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (xDesc.shape.dims[xTensorDimNum - LAST_TWO_NUM] != gammaDesc.shape.dims[gammaTensorDimNum - LAST_TWO_NUM]) {
        ATB_LOG(ERROR) << "The penultimate dim of intensor0 is not equal to the penultimate dim of intensor1";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (xDesc.shape.dims[xTensorDimNum - LAST_ONE_NUM] != gammaDesc.shape.dims[gammaTensorDimNum - LAST_ONE_NUM]) {
        ATB_LOG(ERROR) << "The last dim of intensor0 is not equal to the last dim of intensor1";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (static_cast<uint64_t>(xDesc.shape.dims[xTensorDimNum - LAST_ONE_NUM]) %
            (ALIGNMENT / sizeof(xDesc.dtype)) !=
        0) {
        ATB_LOG(ERROR) << "The last dim of intensor0 is not 32 byte alignment";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status CohereLayerNormOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                                const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs = {};
    OperationUtil::InTensorsToInTensorDescs(inTensors, inTensorDescs);
    TensorDesc resultTensorDescs = outTensors.at(0).desc;
    for (size_t i = 0; i < resultTensorDescs.shape.dimNum; i++) {
        if (resultTensorDescs.shape.dims[i] != inTensorDescs.at(0).shape.dims[i]) {
            ATB_LOG(ERROR) << "The dim of outtensor0 is not equal to the dim of intensor0";
            return ERROR_INVALID_TENSOR_DIM;
        }
    }
    return InferShapeCheckImpl(inTensorDescs);
}

std::shared_ptr<Runner> CohereLayerNormOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<CohereLayerNormRunner>(param_);
}

nlohmann::json CohereLayerNormOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}

} // namespace atb