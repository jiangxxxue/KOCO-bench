/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onehot_operation.h"
#include <limits>
#include <algorithm>
#include "onehot_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const int32_t IN_TENSOR_NUM = 3;
static const int32_t OUT_TENSOR_NUM = 1;

template <> Status CreateOperation(const infer::OnehotParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    *operation = new OnehotOperation(opParam);
    return NO_ERROR;
}

OnehotOperation::OnehotOperation(const infer::OnehotParam &param) : OperationBase("OnehotOperation"), param_(param)
{
    operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("OnehotOperation");
}

OnehotOperation::~OnehotOperation() {}

uint32_t OnehotOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t OnehotOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status OnehotOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                       SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);
    // 增加一个维度depth0
    std::fill_n(outTensorDescs.at(0).shape.dims, MAX_DIM, 0);
    int64_t outSize = static_cast<int64_t>(inTensorDescs.at(0).shape.dimNum + 1);
    int64_t axis = param_.axis >= 0 ? param_.axis : (outSize + param_.axis);
    int32_t indicesIndex = 0;
    int64_t depth = param_.depth;
    for (int i = 0; i < outSize; ++i) {
        if (i == axis) {
            outTensorDescs.at(0).shape.dims[i] = depth;
        } else {
            outTensorDescs.at(0).shape.dims[i] = inTensorDescs.at(0).shape.dims[indicesIndex++];
        }
    }

    outTensorDescs.at(0).shape.dimNum = static_cast<uint64_t>(outSize);
    return NO_ERROR;
}

Status OnehotOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    Status st = DimCheck(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }
    return ParamCheck(inTensorDescs.at(0));
}

Status OnehotOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    SVector<TensorDesc> inTensorDescs;
    for (size_t i = 0; i < inTensors.size(); i++) {
        inTensorDescs.push_back(inTensors.at(i).desc);
    }
    Status st = DimCheck(inTensorDescs);
    if (st != NO_ERROR) {
        return st;
    }

    if (outTensors.at(0).desc.shape.dimNum != inTensors.at(0).desc.shape.dimNum + 1) {
        ATB_LOG(ERROR) << GetLogPrefix() << "outTensor size should one dim lager than input size.";
        return ERROR_INVALID_TENSOR_SIZE;
    }
    return ParamCheck(inTensors.at(0).desc);
}

std::shared_ptr<Runner> OnehotOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<OnehotOpsRunner>(param_);
}

Status OnehotOperation::ParamCheck(TensorDesc inTensorDesc) const
{
    int64_t axis = param_.axis;
    int64_t maxSize = static_cast<int64_t>(inTensorDesc.shape.dimNum + 1);
    if (axis >= maxSize || axis <= -1 * maxSize) {
        ATB_LOG(ERROR) << GetLogPrefix() << "wrong pram, axis: " << axis;
        return ERROR_INVALID_PARAM;
    }
    return NO_ERROR;
}

Status OnehotOperation::DimCheck(const SVector<TensorDesc> &inTensorDescs) const
{
    // 输出维度比输入维度多一维，输入维度限制
    if (inTensorDescs.at(0).shape.dimNum >= MAX_DIM) {
        ATB_LOG(ERROR) << GetLogPrefix() << "input dims should no more than 7.";
        return ERROR_INVALID_TENSOR_SIZE;
    }
    // 后两个输入辅助作用的scalar 0、1
    if ((inTensorDescs.at(1).shape.dimNum != 1 || inTensorDescs.at(1).shape.dims[0] != 1) ||
        (inTensorDescs.at(2).shape.dimNum != 1 || inTensorDescs.at(2).shape.dims[0] != 1)) { // index: 2
        ATB_LOG(ERROR) << GetLogPrefix() << "intensor 1/2 should be scalar.";
        return ERROR_INVALID_TENSOR_SIZE;
    }
    return NO_ERROR;
}

nlohmann::json OnehotOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb