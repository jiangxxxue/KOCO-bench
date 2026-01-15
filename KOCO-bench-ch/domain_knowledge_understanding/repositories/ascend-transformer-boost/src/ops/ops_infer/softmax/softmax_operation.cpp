/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "softmax_operation.h"
#include "atb/utils/log.h"
#include "softmax_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/utils/singleton.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_NUM = 1;

template <> Status CreateOperation(const infer::SoftmaxParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    *operation = new SoftmaxOperation(opParam);
    return NO_ERROR;
}

SoftmaxOperation::SoftmaxOperation(const infer::SoftmaxParam &param) : OperationBase("SoftmaxOperation"), param_(param)
{
    if (GetSingleton<Config>().Is310B()) {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("SoftmaxOperationAtlas200I500A2");
    } else {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("SoftmaxOperation");
    }
}

SoftmaxOperation::~SoftmaxOperation() {}

uint32_t SoftmaxOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t SoftmaxOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status SoftmaxOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                        SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);

    return NO_ERROR;
}

Status SoftmaxOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    return ParamCheck(inTensorDescs.at(0));
}

Status SoftmaxOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    ATB_LOG(DEBUG) << "outTensors Size:" << outTensors.size();
    return ParamCheck(inTensors.at(0).desc);
}

Status SoftmaxOperation::ParamCheck(const TensorDesc &inTensorDesc) const
{
    SVector<int64_t> axes = param_.axes;
    if (axes.size() == 0) {
        ATB_LOG(ERROR) << "empty axes.";
        return ERROR_INVALID_PARAM;
    }

    for (size_t i = 1; i < axes.size(); i++) {
        if (axes[i] != axes[i - 1] + 1 || axes[i] == 0) {
            ATB_LOG(ERROR) << "input axes is not consistent. invalid axes: " << axes[i] << ", " << axes[i - 1];
            return ERROR_INVALID_PARAM;
        }
    }

    for (int64_t dim : param_.axes) {
        if (dim < -1 || dim >= MAX_DIM || dim >= static_cast<int64_t>(inTensorDesc.shape.dimNum)) {
            ATB_LOG(ERROR) << "axes dim should not be less than -1, "
                           << " and maximum should not be more than 8 dimensions and x dimNum. The dim is : " << dim;
            return ERROR_INVALID_PARAM;
        }
    }
    return NO_ERROR;
}

std::shared_ptr<Runner> SoftmaxOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<SoftmaxOpsRunner>(param_);
}

nlohmann::json SoftmaxOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb