/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_operation.h"
#include <algorithm>
#include "reduce_ops_runner.h"
#include "atb/utils/tensor_check.h"
#include "atb/utils/param_to_json.h"
#include "atb/operation/atb_operation_ir_cfg.h"
#include "atb/utils/singleton.h"
#include "atb/operation/op_param_funcs.h"

namespace atb {
static const uint32_t IN_TENSOR_NUM = 1;
static const uint32_t OUT_TENSOR_NUM = 1;

template <> Status CreateOperation(const infer::ReduceParam &opParam, Operation **operation)
{
    if (operation == nullptr) {
        return ERROR_INVALID_PARAM;
    }
    OP_PARAM_RSV_CHECK(opParam);
    *operation = new ReduceOperation(opParam);
    return NO_ERROR;
}

ReduceOperation::ReduceOperation(const infer::ReduceParam &param) : OperationBase("ReduceOperation"), param_(param)
{
    if (param.reduceType == infer::ReduceParam::REDUCE_SUM) {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("ReduceOperationSUM");
    } else {
        operationIr_ = GetSingleton<AtbOperationIrCfg>().GetOperationIr("ReduceOperation");
    }
}

ReduceOperation::~ReduceOperation() {}

uint32_t ReduceOperation::GetInputNum() const
{
    return IN_TENSOR_NUM;
}

uint32_t ReduceOperation::GetOutputNum() const
{
    return OUT_TENSOR_NUM;
}

Status ReduceOperation::InferShapeImpl(const SVector<TensorDesc> &inTensorDescs,
                                       SVector<TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(0) = inTensorDescs.at(0);

    // dim数组清空，并统计降维后剩下的维度
    uint64_t outDimId = 0;
    std::fill_n(outTensorDescs.at(0).shape.dims, MAX_DIM, 0);
    for (std::size_t inDimIndex = 0; inDimIndex < inTensorDescs.at(0).shape.dimNum; inDimIndex++) {
        bool isReduced = false;
        for (std::size_t j = 0; j < param_.axis.size(); j++) {
            if (static_cast<int64_t>(inDimIndex) == param_.axis.at(j)) {
                isReduced = true;
                break;
            }
        }

        if (!isReduced) {
            outTensorDescs.at(0).shape.dims[outDimId++] = inTensorDescs.at(0).shape.dims[inDimIndex];
        }
    }

    outTensorDescs.at(0).shape.dimNum = outDimId;
    return NO_ERROR;
}

Status ReduceOperation::InferShapeCheckImpl(const SVector<TensorDesc> &inTensorDescs) const
{
    return ParamCheck(inTensorDescs.at(0));
}

Status ReduceOperation::SetupCheckImpl(const SVector<Tensor> &inTensors, const SVector<Tensor> &outTensors) const
{
    ATB_LOG(INFO) << GetLogPrefix() << "outTensor Size:" << outTensors.size();
    return ParamCheck(inTensors.at(0).desc);
}

Status ReduceOperation::ParamCheck(TensorDesc inTensorDesc) const
{
    if (static_cast<int32_t>(param_.reduceType) <= infer::ReduceParam::REDUCE_UNDEFINED ||
        param_.reduceType > infer::ReduceParam::REDUCE_SUM) {
        ATB_LOG(ERROR) << "unsupport reduceType: " << param_.reduceType;
        return ERROR_INVALID_PARAM;
    }
    const SVector<int64_t> axis = param_.axis;
    Dims shape = inTensorDesc.shape;

    if (axis.size() == 0) {
        ATB_LOG(ERROR) << "inTensor axis is not support: empty vector";
        return ERROR_INVALID_PARAM;
    }

    if (axis.size() > shape.dimNum) {
        ATB_LOG(ERROR) << "axis size is out of dims range, axis size: " << axis.size();
        return ERROR_INVALID_PARAM;
    }

    for (std::size_t i = 0; i < axis.size(); i++) {
        if (axis[i] < 0 || axis[i] >= static_cast<int64_t>(shape.dimNum)) {
            ATB_LOG(ERROR) << "axis value is out of dims range, wrong axis: " << axis[i];
            return ERROR_INVALID_PARAM;
        }
    }

    return NO_ERROR;
}

std::shared_ptr<Runner> ReduceOperation::CreateRunner(Context &context) const
{
    (void)context;
    return std::make_shared<ReduceOpsRunner>(param_);
}

nlohmann::json ReduceOperation::GetParamJson() const
{
    return OpParamToJson(param_);
}
} // namespace atb