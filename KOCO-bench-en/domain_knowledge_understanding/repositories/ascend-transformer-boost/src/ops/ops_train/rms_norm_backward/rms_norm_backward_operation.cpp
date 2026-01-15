/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "rms_norm_backward_operation.h"
#include "atb/utils/config.h"
#include "atb/utils/log.h"
#include "rms_norm_backward_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/tensor_util.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

namespace {
static const uint32_t IN_TENSOR_NUM = 4;
static const uint32_t OUT_TENSOR_NUM = 2;
static const uint32_t IN_TENSOR_DY = 0;
static const uint32_t IN_TENSOR_X = 1;
static const uint32_t IN_TENSOR_RSTD = 2;
static const uint32_t IN_TENSOR_GAMMA = 3;
static const uint32_t OUT_TENSOR_DX = 0;
static const uint32_t OUT_TENSOR_DGAMMA = 1;

bool IsTensorDiffShape(const atb::TensorDesc &tensorDesc1, const atb::TensorDesc &tensorDesc2)
{
    if (tensorDesc1.format != tensorDesc2.format || tensorDesc1.shape.dimNum != tensorDesc2.shape.dimNum) {
        return true;
    }
    for (uint64_t i = 0; i < tensorDesc1.shape.dimNum; i++) {
        if (tensorDesc1.shape.dims[i] != tensorDesc2.shape.dims[i]) {
            return true;
        }
    }
    return false;
}
} // namespace

namespace atb {
template <> Status CreateOperation(const train::RmsNormBackwardParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    if (!GetSingleton<Config>().Is910B()) {
        ATB_LOG(ERROR) << "RmsNormBackwardOperation is only supported in Atlas 800I A2 inference product.";
        return ERROR_INVALID_PARAM;
    }

    *operation = new (std::nothrow) RmsNormBackwardOperation(opParam);
    if (*operation == nullptr) {
        ATB_LOG(ERROR) << "failed to new operation";
        return ERROR_OUT_OF_HOST_MEMORY;
    }
    return NO_ERROR;
}

RmsNormBackwardOperation::RmsNormBackwardOperation(const train::RmsNormBackwardParam &param)
    : OperationBase("RmsNormBackwardOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("RmsNormBackwardOperation");
}

RmsNormBackwardOperation::~RmsNormBackwardOperation() {}

uint32_t RmsNormBackwardOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t RmsNormBackwardOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status RmsNormBackwardOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                                SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(OUT_TENSOR_DX) = inTensorDescs.at(IN_TENSOR_X);
    outTensorDescs.at(OUT_TENSOR_DGAMMA) = inTensorDescs.at(IN_TENSOR_GAMMA);
    outTensorDescs.at(OUT_TENSOR_DGAMMA).dtype = ACL_FLOAT;
    return NO_ERROR;
}

Status RmsNormBackwardOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    int32_t xDimNum = inTensorDescs.at(IN_TENSOR_X).shape.dimNum;
    int32_t gammaDimNum = inTensorDescs.at(IN_TENSOR_GAMMA).shape.dimNum;
    int32_t startDim = xDimNum - gammaDimNum;
    if (startDim < 0) {
        ATB_LOG(ERROR) << GetLogPrefix() << "gamma shape invalid";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

Status RmsNormBackwardOperation::SetupCheckImpl(const SVector<Tensor> &inTensors,
                                                const SVector<Tensor> &outTensors) const
{
    if (IsTensorDiffShape(outTensors.at(OUT_TENSOR_DX).desc, inTensors.at(IN_TENSOR_X).desc)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor dX shape should be same with X";
        return ERROR_INVALID_TENSOR_DIM;
    }
    if (IsTensorDiffShape(outTensors.at(OUT_TENSOR_DGAMMA).desc, inTensors.at(IN_TENSOR_GAMMA).desc)) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor dGAMMA shape should be same with GAMMA";
        return ERROR_INVALID_TENSOR_DIM;
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> RmsNormBackwardOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<RmsNormBackwardOpsRunner>(param_);
}
} // namespace atb